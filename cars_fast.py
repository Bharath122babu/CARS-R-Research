"""cars_fast.py — faster training + controlled accuracy experiments for OG CARS v0.1.5.

Standalone CARS-R Research runner. Reference assets, data, and outputs live
inside this checkout; the historical projects are not required.

Owner priorities: keep ~100% IID ownership, lower Wiki NLL, less training time.

What it does
  1. Fast path: the same CARS state law, executed with all input-only factors
     (Cayley 2x2 system and a batched ridge Cholesky solve) computed once per
     sequence instead of per token. The token loop uses two bmm operations
     and one baddbmm. Same state law; floating-point rounding order differs.
  2. Training arms, one variable at a time, original data/model/token budget:
        og       original recipe exactly (reproduction + timing)
        lr8      peak LR 4e-4 -> 8e-4
        state10  decay/ridge/read-scale LR multiplier 1 -> 10
                 (an experimental setting; the original decay drift alone
                  does not establish that these parameters were LR-limited)
        lr8_state10  both — only run after each one helps on its own
     Optional --steps N changes the horizon (fixed-time experiment).
  3. Evaluation identical to the screenshot table (same 1,024 Wiki windows,
     same 500 IID cases, same 256 five-transfer cases, same stress suites, the
     project's own score_cases) plus a FRESH IID set disjoint from all splits.
  4. A leaderboard with pass/fail against the targets.

From the repository root:
  python cars_fast.py qualify
  python cars_fast.py train --arm og --seed 1337
  python cars_fast.py evaluate --run runs\\fast\\og_seed1337
  python cars_fast.py evaluate --run references/original/cars --reference
  python cars_fast.py summary
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import shutil
import statistics
import sys
import tempfile
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# All paths are relative to this standalone checkout.
ORIGINAL_RUN = ROOT / "references" / "original"
DELTA_RUN = ORIGINAL_RUN
DATA = ROOT / "data" / "comparison_wikitext2"
FAST_RUNS = ROOT / "runs" / "fast"
EVAL_SEED = 20261005           # identical to the screenshot evaluation
FRESH_IID_SEED = 99173         # new; never used before in this project
FRESH_IID_CASES = 2000

BASELINE = {"wiki_nll": 1.54078, "iid": 1.0, "five": 1.0, "minutes": 29.76}
TARGET = {"wiki_nll": 1.53, "iid": 0.99, "five": 0.99, "minutes": 15.0}

ORIGINAL_TRAIN_CONFIG = {
    "seed": 1337, "batch_size": 32, "seq_len": 128, "max_steps": 1000, "grad_accum_steps": 1,
    "learning_rate": 0.0004, "state_lr_scale": 1.0, "min_lr_ratio": 0.1, "warmup_steps": 100,
    "weight_decay": 0.1, "grad_clip": 1.0, "beta1": 0.9, "beta2": 0.95, "mtp_loss_weight": 0.0,
    "eval_interval": 100, "eval_batches": 8, "save_interval": 100, "log_interval": 1,
    "precision": "bf16", "compile_model": False, "gradient_checkpointing": False,
    "document_mode": "lines", "health_interval": 100,
}
ARMS = {
    "og": {},
    "lr8": {"learning_rate": 0.0008},
    "state10": {"state_lr_scale": 10.0},
    "lr8_state10": {"learning_rate": 0.0008, "state_lr_scale": 10.0},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def execution_identity(data_dir=DATA, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "runner_sha256": sha256_file(Path(__file__)),
        "source": {p.name: sha256_file(p) for p in sorted((ROOT / "cars_r").glob("*.py"))},
        "model_config_sha256": sha256_file(_model_config_path()),
        "dataset": {p.name: sha256_file(p) for p in sorted(Path(data_dir).iterdir() if Path(data_dir).exists() else []) if p.is_file()},
        "reference_checkpoint_sha256": sha256_file(ORIGINAL_RUN / "cars/best.pt") if (ORIGINAL_RUN / "cars/best.pt").exists() else None,
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def require_experiment_output(out):
    out = Path(out).resolve()
    if not out.is_relative_to(FAST_RUNS.resolve()) or out == FAST_RUNS.resolve():
        raise SystemExit("Training outputs must be a child of this branch's runs/fast directory.")
    return out


# =============================================================================
# 1. Fast path for the ORIGINAL operator
# =============================================================================
def _inv2x2(m: Tensor) -> Tensor:
    a, b, c, d = m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1]
    det = a * d - b * c
    return torch.stack((torch.stack((d, -b), -1), torch.stack((-c, a), -1)), -2) / det[..., None, None]


def _supported(cfg) -> bool:
    # Experimental non-OG transport modes retain the exact original path,
    # including their bitwise cross-mode control comparisons.
    return (cfg.write_mode == "ridge" and cfg.cars_read_before_write
            and not cfg.cars_transport_read_only and cfg.cars_transport_enabled
            and cfg.cars_transport_max_strength != 0.0)


def _prepare_factors(mixer, projected: Tensor):
    cfg = mixer.config
    q, k, v, u, w, s = mixer._split_projected(projected)       # FP32, same normalisation as reference
    dt, r = q.dtype, mixer.rank
    if cfg.cars_transport_enabled and cfg.cars_transport_max_strength != 0.0:
        eta = torch.tanh(s[..., 0]) * cfg.cars_transport_max_strength
    else:
        eta = torch.zeros_like(s[..., 0])
    U2 = torch.stack((u, -w), -1) * eta[..., None, None]            # Omega = U2 V2^T
    V2 = torch.stack((w, u), -1)
    Gc = torch.eye(2, dtype=dt, device=q.device) - 0.5 * V2.transpose(-1, -2) @ U2   # det >= 1
    P = U2 @ _inv2x2(Gc)                                              # S T = S + (S P) V2^T
    lam = mixer.ridge.to(dt)[None, None, :, None, None]
    Gk = k.transpose(-1, -2) @ k + lam * torch.eye(r, dtype=dt, device=q.device)
    # One batched SPD factorization per sequence, retaining the reference solver.
    # This is more reliable for nearly collinear evidence than explicit inversion.
    R = torch.cholesky_solve(k.transpose(-1, -2), torch.linalg.cholesky(Gk))
    C = torch.cat((P, q[..., None], k), -1)                           # [B,T,H,dk,3+r]
    D = torch.cat((V2, R.transpose(-1, -2)), -1)                      # [B,T,H,dk,2+r]
    a = V2.transpose(-1, -2) @ torch.cat((q[..., None], k), -1)       # [B,T,H,2,1+r]
    return C, D, a, v


def fused_forward(mixer, x: Tensor, *, state: Tensor | None = None, attention_mask: Tensor | None = None):
    """Same outputs/state as CARSStateMixer.forward (no diagnostics). Returns (y, state_matrix)."""
    b, t, _ = x.shape
    h, dk, dv, r = mixer.h, mixer.dk, mixer.dv, mixer.rank
    projected = mixer.in_proj(x)
    with torch.autocast(device_type=x.device.type, enabled=False):
        C, D, a, v = _prepare_factors(mixer, projected)
        dt, bh = C.dtype, b * h
        C = C.transpose(0, 1).reshape(t, bh, dk, 3 + r)
        D = D.transpose(0, 1).reshape(t, bh, dk, 2 + r).transpose(-1, -2)
        a = a.transpose(0, 1).reshape(t, bh, 2, 1 + r)
        v = v.transpose(0, 1).reshape(t, bh, dv, r)
        delta = mixer.state_decay.to(dt)[None, :, None, None].expand(b, h, 1, 1).reshape(bh, 1, 1)
        S = (torch.zeros(bh, dv, dk, dtype=dt, device=x.device) if state is None
             else state.to(dt).reshape(bh, dv, dk))
        mask = None
        if attention_mask is not None:
            mask = attention_mask.bool()[:, None, :].expand(b, h, t).reshape(bh, t).transpose(0, 1)
        reads = []
        for i in range(t):
            Z = torch.bmm(S, C[i])                                    # S @ [P | q | K]
            ZP = Z[..., :2]
            Zr = delta * (Z[..., 2:] + torch.bmm(ZP, a[i]))           # S_bar @ [q | K]
            read = Zr[..., 0]
            E = v[i] - Zr[..., 1:]                                    # innovation
            # Valid even when forgetting underflows to zero; never divide by delta.
            S_new = torch.baddbmm(delta * S, torch.cat((delta * ZP, E), -1), D[i])
            if mask is not None:
                m = mask[i]
                S_new = torch.where(m[:, None, None], S_new, S)
                read = read * m[:, None].to(dt)
            S = S_new
            reads.append(read)
        reads = torch.stack(reads, 1).reshape(b, h, t, dv).transpose(1, 2)
        reads = mixer._process_read(reads)
    y = mixer.out_proj(reads.reshape(b, t, h * dv).to(x.dtype))
    return y, S.reshape(b, h, dv, dk)


_PATCHED = {"on": False}


def enable_fast_path():
    """Route CARSStateMixer.forward through the fast path (diagnostics keep the reference)."""
    from cars_r import cars_operator as O
    if _PATCHED["on"]:
        return
    original = O.CARSStateMixer.forward

    def forward(self, x, *, state=None, attention_mask=None, return_diagnostics=False):
        if return_diagnostics or not _supported(self.config):
            return original(self, x, state=state, attention_mask=attention_mask,
                            return_diagnostics=return_diagnostics)
        if (x.ndim != 3 or x.shape[1] == 0 or
                (attention_mask is not None and attention_mask.shape != x.shape[:2]) or
                (state is not None and (state.matrix.shape != (x.shape[0], self.h, self.dv, self.dk)
                                       or state.matrix.device != x.device))):
            return original(self, x, state=state, attention_mask=attention_mask)
        y, S = fused_forward(self, x, state=None if state is None else state.matrix,
                             attention_mask=attention_mask)
        return y, O.CARSState(S), None

    forward._cars_reference = original
    O.CARSStateMixer.forward = forward
    _PATCHED["on"] = True


def disable_fast_path():
    from cars_r import cars_operator as O
    ref = getattr(O.CARSStateMixer.forward, "_cars_reference", None)
    if ref is not None:
        O.CARSStateMixer.forward = ref
    _PATCHED["on"] = False


# =============================================================================
# 2. Qualification
# =============================================================================
def _model_config_path() -> Path:
    for p in (ORIGINAL_RUN / "cars_config.json", DELTA_RUN / "cars_config.json"):
        if p.exists():
            return p
    raise FileNotFoundError("cars_config.json not found under runs/comparison_laptop_*seed1337")


def _fp64_operator_copy() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="cars_fp64_"))
    shutil.copytree(ROOT / "cars_r", tmp / "cars_r", ignore=shutil.ignore_patterns("__pycache__"))
    op = tmp / "cars_r" / "cars_operator.py"
    op.write_text(op.read_text(encoding="utf-8").replace(".float()", ".double()")
                  .replace("torch.float32", "torch.float64"), encoding="utf-8")
    return tmp


def _import_operator(pkg_root: Path):
    for k in [k for k in sys.modules if k == "cars_r" or k.startswith("cars_r.")]:
        del sys.modules[k]
    sys.path.insert(0, str(pkg_root))
    try:
        import cars_r.config as C
        import cars_r.cars_operator as O
    finally:
        sys.path.pop(0)
    return C, O


def _mixer_case(C, O, cfg, dtype, device, stress):
    torch.manual_seed(0)
    m = O.CARSStateMixer(C.ModelConfig(**cfg)).to(device=device, dtype=dtype)
    with torch.no_grad():
        torch.nn.init.normal_(m.in_proj.weight, std=0.02)
        torch.nn.init.normal_(m.out_proj.weight, std=0.02)
        m.apply_architecture_initialization()
        ph = m.per_head
        rows = (torch.arange(m.h) * ph + ph - 1).to(device)
        m.in_proj.weight[rows] = torch.randn(m.h, m.config.d_model, dtype=dtype, device=device) * 0.05
        for hh in range(m.h):
            if stress == "collinear_keys":
                base = hh * ph + m.dk
                kr = m.in_proj.weight[base:base + m.dk * m.rank].view(m.dk, m.rank, -1)
                kr[:, 1] = kr[:, 0] + 1e-3 * torch.randn_like(kr[:, 0])
            if stress == "near_collinear_keys":
                base = hh * ph + m.dk
                kr = m.in_proj.weight[base:base + m.dk * m.rank].view(m.dk, m.rank, -1)
                kr[:, 1] = kr[:, 0] + 1e-5 * torch.randn_like(kr[:, 0])
            if stress == "u_equals_w":
                base = hh * ph + m.dk + m.dk * m.rank + m.dv * m.rank
                m.in_proj.weight[base + m.dk:base + 2 * m.dk] = m.in_proj.weight[base:base + m.dk]
    torch.manual_seed(1)
    x = torch.randn(4, 64, m.config.d_model, dtype=dtype, device=device, requires_grad=True)
    mask = torch.ones(4, 64, dtype=torch.bool, device=device)
    mask[0, 40:] = False; mask[1, 10:13] = False; mask[2, :5] = False
    if stress == "zero_decay":
        with torch.no_grad(): m.raw_decay_rate.fill_(1000)
    initial = torch.randn(4, m.h, m.dv, m.dk, dtype=dtype, device=device) * 0.1
    y1, s1, _ = m(x, state=O.CARSState(initial), attention_mask=mask)
    g = torch.randn_like(y1)
    params = [x] + list(m.parameters())
    g1 = torch.autograd.grad((y1 * g).sum() + 1e-3 * s1.matrix.square().sum(), params)
    y2, s2 = fused_forward(m, x, state=initial, attention_mask=mask)
    g2 = torch.autograd.grad((y2 * g).sum() + 1e-3 * s2.square().sum(), params)
    rel = lambda a, b: ((a - b).abs().max() / (b.abs().max() + 1e-30)).item()
    return rel(y2, y1), rel(s2, s1.matrix), max(rel(a, b) for a, b in zip(g2, g1))


def _time_train_step(model, device, steps=10):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 256, (32, 129), generator=gen).to(device)
    from cars_r.train import autocast_context

    def step():
        with autocast_context(device, "bf16" if device.type == "cuda" else "fp32"):
            logits = model(ids[:, :-1]).logits
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)

    for _ in range(3):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = (time.perf_counter() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
    return sec, peak


def cmd_qualify(args):
    cfg = json.loads(_model_config_path().read_text(encoding="utf-8"))
    report = {"fast_path_file_sha256": sha256_file(Path(__file__)), "checks": [], "torch": torch.__version__}
    ok = True

    print("[1/4] operator: fast path vs reference (outputs, final state, all gradients; masks with holes)")
    runs = [("fp64-cpu", _fp64_operator_copy(), torch.float64, "cpu", 1e-10),
            ("fp32-cpu", ROOT, torch.float32, "cpu", 5e-5)]
    if torch.cuda.is_available():
        runs.append(("fp32-cuda", ROOT, torch.float32, "cuda", 5e-5))
    for label, pkg, dtype, dev, tol in runs:
        C, O = _import_operator(pkg)
        for stress in ("random", "collinear_keys", "near_collinear_keys", "u_equals_w", "zero_decay"):
            ry, rs, rg = _mixer_case(C, O, cfg, dtype, torch.device(dev), stress)
            # Near-dependent key gradients suffer cancellation in FP32 even
            # with the same SPD solver; use the same 0.1% relative allowance
            # for both collinear stress levels, while retaining tight FP64.
            limit = tol * (20 if "collinear_keys" in stress else 1)
            passed = max(ry, rs, rg) < limit
            ok &= passed
            report["checks"].append({"check": f"operator {label} {stress}", "out": ry, "state": rs, "grad": rg, "relative_limit": limit, "pass": passed})
            print(f"   {'PASS' if passed else 'FAIL'} {label:9s} {stress:15s} rel err out {ry:.1e} state {rs:.1e} grad {rg:.1e}")
    _import_operator(ROOT)  # restore the real package

    from cars_r.checkpoint import load_checkpoint
    from cars_r.model import CARSRLanguageModel
    from cars_r.config import ModelConfig
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[2/4] full-model NLL: reference vs fast path")
    export = ORIGINAL_RUN / "cars" / "best.pt"
    wiki_bin = DATA / "wiki_test.bin"
    from cars_r.data import TokenStream
    from cars_r.train import autocast_context
    import numpy as np
    trained_reference = export.exists()
    if trained_reference:
        model, _ = load_checkpoint(export, device=device)
    else:
        torch.manual_seed(0)
        model = CARSRLanguageModel(ModelConfig(**cfg)).to(device)
        print("   No historical weights: checking a seeded untrained model (numerical check only).")
    model.eval()
    if wiki_bin.exists():
        src = TokenStream.from_bin(wiki_bin, split="all").tokens
        blocks = np.random.default_rng(EVAL_SEED).choice((len(src) - 1) // 128, 1024, replace=False)[:args.qualify_windows]
        batch = torch.tensor(np.stack([src[s * 128:s * 128 + 129] for s in blocks]).astype("int64"), device=device)
    else:
        batch = torch.randint(0, 256, (min(args.qualify_windows, 16), 129), generator=torch.Generator().manual_seed(0)).to(device)
    res = {}
    for mode in ("reference", "fast"):
        (enable_fast_path if mode == "fast" else disable_fast_path)()
        nll = []
        with torch.inference_mode():
            for i in range(0, len(batch), 16):
                bb = batch[i:i + 16]
                with autocast_context(device, "bf16" if device.type == "cuda" else "fp32"):
                    lg = model(bb[:, :-1]).logits.float()
                nll.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), bb[:, 1:].reshape(-1), reduction="sum").item())
        res[mode] = sum(nll) / (len(batch) * 128)
    disable_fast_path()
    diff = abs(res["fast"] - res["reference"])
    passed = diff < 2e-3
    ok &= passed
    report["checks"].append({"check": "full-model NLL", "historical_weights": trained_reference,
                             "wiki_data": wiki_bin.exists(), **res, "abs_diff": diff, "pass": passed})
    print(f"   {'PASS' if passed else 'FAIL'} {len(batch)} windows: reference {res['reference']:.5f}  fast {res['fast']:.5f}  |diff| {diff:.1e}")

    print("[3/4] full-model training step time, batch 32 x context 128 (same weights)")
    torch.manual_seed(0)
    base = CARSRLanguageModel(ModelConfig(**cfg)).to(device)
    timings = {}
    for mode in ("reference", "fast"):
        (enable_fast_path if mode == "fast" else disable_fast_path)()
        model = CARSRLanguageModel(ModelConfig(**cfg)).to(device)
        model.load_state_dict(base.state_dict())
        model.train()
        timings[mode] = _time_train_step(model, device, steps=args.timing_steps)
        print(f"   {mode:9s} {timings[mode][0]:.3f} s/step   peak {timings[mode][1]:.0f} MiB")
    disable_fast_path()
    sp = timings["reference"][0] / timings["fast"][0]
    report["timing"] = {"device": str(device), "reference_s_per_step": timings["reference"][0],
                        "fast_s_per_step": timings["fast"][0], "speedup": sp,
                        "projected_1000_step_minutes_fast": timings["fast"][0] * 1000 / 60}
    print(f"   speedup {sp:.2f}x  -> about {timings['fast'][0] * 1000 / 60:.1f} min of step compute per 1,000 steps"
          " (excludes validation/checkpoints; check against the real run's metrics.jsonl)")

    print("[4/4] host synchronisations inside one fast-path CARS layer forward+backward")
    if device.type == "cuda":
        from cars_r import cars_operator as O
        m = O.CARSStateMixer(ModelConfig(**cfg)).to(device)
        xx = torch.randn(32, 128, cfg["d_model"], device=device, requires_grad=True)
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                y, _ = fused_forward(m, xx)
                y.float().square().mean().backward()
            finally:
                torch.cuda.set_sync_debug_mode("default")
        n = sum("synchroniz" in str(w.message).lower() for w in caught)
        report["fast_path_syncs"] = n
        print(f"   {n} synchronizing warnings (partial diagnostic, not an exhaustive count)")
    else:
        print("   SKIP (needs CUDA)")

    report["verdict"] = "PASS" if ok else "FAIL"
    report["execution_identity"] = execution_identity()
    FAST_RUNS.mkdir(parents=True, exist_ok=True)
    (FAST_RUNS / "QUALIFICATION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nQUALIFICATION {report['verdict']} -> {FAST_RUNS / 'QUALIFICATION.json'}")
    if not ok:
        sys.exit(1)


# =============================================================================
# 3. Training arms
# =============================================================================
def run_dir(arm: str, seed: int, steps: int | None) -> Path:
    return FAST_RUNS / (f"{arm}_seed{seed}" + (f"_steps{steps}" if steps else ""))


def cmd_train(args):
    if args.arm not in ARMS:
        raise SystemExit(f"unknown arm {args.arm}; choose from {sorted(ARMS)}")
    if not args.reference and not (FAST_RUNS / "QUALIFICATION.json").exists():
        raise SystemExit("run `python cars_fast.py qualify` first (or pass --reference)")
    if not args.reference:
        q = json.loads((FAST_RUNS / "QUALIFICATION.json").read_text(encoding="utf-8"))
        if q.get("verdict") != "PASS" or q.get("execution_identity") != execution_identity(args.data_dir, args.device):
            raise SystemExit("qualification missing/failed or this file changed since; rerun `python cars_fast.py qualify`")
    base_cfg_path = ORIGINAL_RUN / "train_config.json"
    train_cfg = json.loads(base_cfg_path.read_text(encoding="utf-8")) if base_cfg_path.exists() else dict(ORIGINAL_TRAIN_CONFIG)
    if train_cfg != ORIGINAL_TRAIN_CONFIG:
        print("WARNING: runs/comparison_laptop_seed1337/train_config.json differs from the recorded original; using the file.")
    train_cfg.update(ARMS[args.arm])
    train_cfg["seed"] = args.seed
    if args.steps:
        train_cfg["max_steps"] = args.steps
    for item in args.override or []:              # testing only, e.g. batch_size=4
        k, v = item.split("=", 1)
        train_cfg[k] = type(train_cfg[k])(json.loads(v)) if not isinstance(train_cfg[k], bool) else v.lower() == "true"
    out = run_dir(args.arm, args.seed, args.steps)
    if args.out_dir:
        out = Path(args.out_dir)
    out = require_experiment_output(out)
    identity = {"identity": execution_identity(args.data_dir, args.device),
                "execution": "reference" if args.reference else "fast_path",
                "train_config": train_cfg, "arm": args.arm}
    contract_path = out / "execution_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != identity:
            raise SystemExit("Execution/data/config contract changed; use a new output directory.")
    elif out.exists() and (list(out.glob("*.pt")) or (out / "metrics.jsonl").exists()):
        raise SystemExit("Existing run has no immutable execution contract; use a new output directory.")
    out.mkdir(parents=True, exist_ok=True)
    if not contract_path.exists():
        contract_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    cfg_path = out / "train_config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text(encoding="utf-8")) != train_cfg:
        raise SystemExit(f"{out} was started with different settings; use a new arm/seed/--out-dir")
    if (out / "final.pt").exists():
        print(f"already complete: {out}")
        return
    cfg_path.write_text(json.dumps(train_cfg, indent=2), encoding="utf-8")
    model_cfg_path = out / "model_config.json"
    shutil.copy2(_model_config_path(), model_cfg_path)
    (out / "fast_path.json").write_text(json.dumps({
        "execution": "reference" if args.reference else "fast_path (cars_fast.py)",
        "cars_fast_sha256": sha256_file(Path(__file__)),
        "cars_r_untouched": True, "arm": args.arm, "arm_changes": ARMS[args.arm],
        "seed": args.seed, "max_steps": train_cfg["max_steps"],
    }, indent=2), encoding="utf-8")
    argv = ["cars_r.train", "--model-config", str(model_cfg_path), "--train-config", str(cfg_path),
            "--bin", str(args.data_dir / "train.bin"), "--val-bin", str(args.data_dir / "validation.bin"),
            "--out-dir", str(out)]
    if args.device:
        argv += ["--device", args.device]
    if (out / "latest.pt").exists():
        argv += ["--resume", str(out / "latest.pt")]
    if args.stop_after:
        argv += ["--stop-after", str(args.stop_after)]
    if not args.reference:
        enable_fast_path()
    print(f"TRAIN {out.name}: {'reference' if args.reference else 'fast path'}; changes vs original: "
          f"{ARMS[args.arm] or 'none'}; seed {args.seed}; steps {train_cfg['max_steps']}", flush=True)
    import cars_r.train as T
    sys.argv = argv
    started = time.perf_counter()
    try:
        T.main()
    finally:
        disable_fast_path()
        with (out / "invocation_times.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"wall_seconds": time.perf_counter() - started,
                                 "completed": (out / "final.pt").exists()}) + "\n")


# =============================================================================
# 4. Evaluation (same suites as the screenshot + fresh IID)
# =============================================================================
def _load_evaluator():
    spec = importlib.util.spec_from_file_location("eval_completed", ROOT / "scripts" / "evaluate_completed_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fresh_iid_cases(seed=FRESH_IID_SEED, count=FRESH_IID_CASES, data_dir=DATA):
    """Unseen cases from the original generator, conditioned on split exclusion.

    Finite-space exclusion makes this set almost entirely three-transfer cases.
    """
    from cars_r.comparison_data import relational_case
    used = set()
    for split in ("train", "validation", "test"):
        for line in (data_dir / f"relations_{split}.jsonl").read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            used.add(c["prompt"] + c["target"])
    rng, out, seen = random.Random(seed), [], set()
    tries = 0
    while len(out) < count:
        tries += 1
        if tries > 2_000_000:
            raise RuntimeError("could not find enough unseen IID cases")
        prompt, target = relational_case(rng, rng.choice((2, 3)))
        key = prompt + target
        if key in used or key in seen:
            continue
        seen.add(key)
        out.append({"prompt": prompt, "target": target})
    return out


def training_minutes(run: Path) -> float:
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    steps = {r["step"]: r["training_seconds"] for r in rows if "training_seconds" in r}
    return steps[max(steps)] / 60 if steps else float("nan")


def cmd_evaluate(args):
    import numpy as np
    run = Path(args.run)
    if not run.is_absolute():
        run = ROOT / run
    E = _load_evaluator()
    from cars_r.checkpoint import load_checkpoint
    from cars_r.comparison_data import relational_case
    from cars_r.data import TokenStream
    from cars_r.train import autocast_context
    if not args.reference:
        enable_fast_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = run / "best.pt"
    model, payload = load_checkpoint(ckpt, device=device)
    model.eval()
    src = TokenStream.from_bin(args.data_dir / "wiki_test.bin", split="all").tokens
    blocks = np.random.default_rng(EVAL_SEED).choice((len(src) - 1) // 128, 1024, replace=False)[:args.windows]
    wiki = np.stack([src[s * 128:s * 128 + 129] for s in (blocks * 1).tolist()]).astype(np.int64)
    losses = []
    with torch.inference_mode():
        for i in range(0, len(wiki), 16):
            bb = torch.tensor(wiki[i:i + 16], device=device)
            with autocast_context(device, "bf16" if device.type == "cuda" else "fp32"):
                lg = model(bb[:, :-1]).logits.float()
            losses.extend(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), bb[:, 1:].reshape(-1),
                                          reduction="none").reshape(len(bb), 128).mean(-1).tolist())
    rng = random.Random(EVAL_SEED)
    five = [dict(zip(("prompt", "target"), relational_case(rng, 5))) for _ in range(256)]
    suites = {
        "iid_original_500": [json.loads(l) for l in (args.data_dir / "relations_test.jsonl").read_text(encoding="utf-8").splitlines()],
        "iid_fresh": fresh_iid_cases(data_dir=args.data_dir),
        "five_transfers": five,
        "two_objects_distractor_last": E.stress_cases(EVAL_SEED + 1, 256),
        "two_objects_unchanged_query": E.stress_cases(EVAL_SEED + 2, 256, unchanged=True),
    }
    scores = {k: E.score_cases(model, v, device, 16) for k, v in suites.items()}
    result = {
        "run": run.name, "checkpoint": "best.pt", "selected_step": payload["step"],
        "execution": "reference" if args.reference else "fast_path",
        "wiki_nll": float(np.mean(losses)), "wiki_windows": len(losses),
        "training_minutes": training_minutes(run),
        "accuracy": {k: v["candidate_accuracy"] for k, v in scores.items()},
        "unrestricted_greedy": {k: v["unrestricted_greedy_space_and_owner_exact"] for k, v in scores.items()},
        "fresh_iid_seed": FRESH_IID_SEED, "eval_seed": EVAL_SEED,
        "settings": json.loads((run / "train_config.json").read_text(encoding="utf-8")) if (run / "train_config.json").exists() else None,
        "provenance": {"checkpoint_sha256": sha256_file(ckpt),
                       "execution_identity": execution_identity(args.data_dir),
                       "suites_sha256": hashlib.sha256(json.dumps(suites, sort_keys=True).encode()).hexdigest(),
                       "wiki_blocks": blocks.tolist()},
        "fresh_transfer_counts": {str(n): sum(c["prompt"].count(" gives ") == n for c in suites["iid_fresh"]) for n in (2, 3)},
    }
    try:
        run.resolve().relative_to(FAST_RUNS.resolve())
        dest = run
    except ValueError:   # never write into original/protected run folders
        dest = FAST_RUNS / f"baseline_{run.parent.name}_{run.name}"
        dest.mkdir(parents=True, exist_ok=True)
        result["run"] = dest.name
    (dest / "fast_eval.json").write_text(json.dumps({**result, "predictions": {k: v["predictions"] for k, v in scores.items()}}, indent=2), encoding="utf-8")
    (dest / "evaluation_cases.json").write_text(json.dumps(suites, indent=2), encoding="utf-8")
    a = result["accuracy"]
    print(f"{run.name}: Wiki NLL {result['wiki_nll']:.5f} ({len(losses)} windows) | IID {a['iid_original_500']:.1%} | "
          f"fresh IID {a['iid_fresh']:.1%} | 5-transfer {a['five_transfers']:.1%} | train {result['training_minutes']:.2f} min")
    _write_leaderboard()


def _write_leaderboard():
    rows = []
    for f in sorted(FAST_RUNS.glob("*/fast_eval.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        a = r["accuracy"]
        rows.append({"run": r["run"], "wiki_nll": round(r["wiki_nll"], 5), "windows": r["wiki_windows"],
                     "iid_500": round(a["iid_original_500"], 4), "iid_fresh": round(a["iid_fresh"], 4),
                     "five_transfers": round(a["five_transfers"], 4), "train_minutes": round(r["training_minutes"], 2),
                     "selected_step": r["selected_step"]})
    if rows:
        FAST_RUNS.mkdir(parents=True, exist_ok=True)
        with (FAST_RUNS / "LEADERBOARD.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    return rows


def cmd_summary(args):
    rows = _write_leaderboard()
    if not rows:
        print("no evaluated runs yet")
        return
    print(f"original: Wiki NLL {BASELINE['wiki_nll']}, IID 100%, 5-transfer 100%, {BASELINE['minutes']} min")
    print(f"targets : Wiki NLL < {TARGET['wiki_nll']}, IID/fresh IID/5-transfer >= 99%, < {TARGET['minutes']:.0f} min\n")
    print(f"{'run':28s} {'NLL':>8s} {'IID':>7s} {'fresh':>7s} {'5-tr':>7s} {'min':>6s}  verdict")
    groups: dict[str, list] = {}
    for r in rows:
        ok_own = r["iid_500"] >= TARGET["iid"] and r["iid_fresh"] >= TARGET["iid"] and r["five_transfers"] >= TARGET["five"]
        tags = ["ownership OK" if ok_own else "OWNERSHIP FAIL",
                "NLL target" if r["wiki_nll"] < TARGET["wiki_nll"] else ("NLL better" if r["wiki_nll"] < BASELINE["wiki_nll"] else "NLL not better"),
                "time target" if r["train_minutes"] < TARGET["minutes"] else "time over"]
        note = "" if r["windows"] == 1024 else f" (only {r['windows']} windows — not comparable)"
        print(f"{r['run']:28s} {r['wiki_nll']:8.5f} {r['iid_500']:7.1%} {r['iid_fresh']:7.1%} {r['five_transfers']:7.1%} {r['train_minutes']:6.2f}  {', '.join(tags)}{note}")
        arm = r["run"].split("_seed")[0] + ("_steps" + r["run"].split("_steps")[1] if "_steps" in r["run"] else "")
        # An incomplete evaluation or failed ownership/time target cannot win.
        if ok_own and r["windows"] == 1024 and r["train_minutes"] < TARGET["minutes"]:
            groups.setdefault(arm, []).append(r["wiki_nll"])
    print("\nper arm (mean NLL over seeds; an arm wins only if its gain is > 0.01 AND > 2x the seed spread of 'og'):")
    og = groups.get("og", [])
    spread = statistics.pstdev(og) if len(og) >= 2 else None
    for arm, v in groups.items():
        mean = statistics.mean(v)
        line = f"  {arm:20s} n={len(v)} mean {mean:.5f}"
        if og and arm != "og":
            gain = statistics.mean(og) - mean
            line += f"  gain vs og {gain:+.5f}"
            if len(v) >= 3 and len(og) >= 3 and spread is not None:
                line += "  -> candidate for matched-seed review" if gain > max(0.01, 2 * spread) else "  -> not accepted"
            else:
                line += "  (need 3 seeds in both arms)"
        print(line)
    if spread is not None:
        print(f"  og seed spread (pop. std): {spread:.5f}")


# =============================================================================
def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qualify", help="prove fast path == reference; measure speed on this machine")
    q.add_argument("--qualify-windows", type=int, default=128)
    q.add_argument("--timing-steps", type=int, default=10)
    t = sub.add_parser("train", help="train one arm/seed (resumable: rerun the same command)")
    t.add_argument("--arm", default="og", choices=sorted(ARMS))
    t.add_argument("--seed", type=int, default=1337)
    t.add_argument("--steps", type=int, default=None, help="change horizon (fixed-time experiment)")
    t.add_argument("--reference", action="store_true", help="use the original slow operator")
    t.add_argument("--device", choices=("cpu", "cuda"), default=None)
    t.add_argument("--data-dir", type=Path, default=DATA)
    t.add_argument("--out-dir", default=None)
    t.add_argument("--stop-after", type=int, default=None)
    t.add_argument("--override", nargs="*", help=argparse.SUPPRESS)
    e = sub.add_parser("evaluate", help="score best.pt of a run on the screenshot suites + fresh IID")
    e.add_argument("--run", required=True)
    e.add_argument("--windows", type=int, default=1024)
    e.add_argument("--data-dir", type=Path, default=DATA)
    e.add_argument("--reference", action="store_true")
    sub.add_parser("summary", help="leaderboard + accept/reject per arm")
    args = p.parse_args()
    if args.cmd in ("train", "evaluate"):
        from cars_r.comparison_data import prepare_dataset
        prepare_dataset(getattr(args, "data_dir", DATA))
    {"qualify": cmd_qualify, "train": cmd_train, "evaluate": cmd_evaluate, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()

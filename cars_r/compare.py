"""One-command, resumable CARS-R versus conventional Transformer pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from .checkpoint import read_checkpoint_payload
from .comparison_data import prepare_dataset
from .comparison_eval import evaluate_checkpoint
from .config import ModelConfig, TrainConfig
from .export_model import export_model
from .model import CARSRLanguageModel

ROOT = Path.cwd().resolve()


ARCHITECTURES = ("cars", "transformer", "kda", "gdn2")


def source_sha256():
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def model_configs(profile="laptop"):
    if profile == "micro":
        cars = ModelConfig(d_model=64, n_layers=4, n_heads=2, d_ff=192,
            max_seq_len=256, cars_key_dim=16, cars_value_dim=16,
            cars_transport_max_strength=0.25, cars_half_life_min=32.0,
            cars_half_life_max=512.0, mla_q_lora_rank=32, mla_kv_lora_rank=32,
            mla_qk_nope_dim=12, mla_qk_rope_dim=4, mla_v_dim=16, mtp_steps=1)
        baseline_ff = 184
    elif profile in ("pilot", "laptop"):
        width = 192 if profile == "laptop" else 128
        cars = ModelConfig(d_model=width, n_layers=4, n_heads=4, d_ff=3*width,
            max_seq_len=512, cars_key_dim=32, cars_value_dim=32,
            cars_transport_max_strength=0.25, cars_half_life_min=64.0,
            cars_half_life_max=2048.0, mla_q_lora_rank=64, mla_kv_lora_rank=64,
            mla_qk_nope_dim=24, mla_qk_rope_dim=8, mla_v_dim=32, mtp_steps=1)
        baseline_ff = 608 if profile == "laptop" else 512
    else:
        raise ValueError(f"unknown comparison profile: {profile}")
    transformer = ModelConfig(**cars.to_dict())
    transformer.architecture = "transformer"
    transformer.d_ff = baseline_ff
    models = {"cars": cars, "transformer": transformer}
    target = CARSRLanguageModel(cars).num_parameters()
    for architecture in ("kda", "gdn2"):
        delta = ModelConfig(**cars.to_dict())
        delta.architecture = architecture
        delta.delta_key_dim = cars.cars_key_dim
        delta.delta_value_dim = cars.cars_value_dim
        delta.d_ff = 8
        fixed = CARSRLanguageModel(delta).num_parameters()-3*delta.d_model*delta.n_layers*delta.d_ff
        multiple = 2 if profile == "micro" else 8
        delta.d_ff = max(multiple, multiple*round((target-fixed)/(3*delta.d_model*delta.n_layers*multiple)))
        models[architecture] = delta
    return models


def validate_reuse(directory, models, cfg, manifest, device):
    """Reuse completed CARS/Transformer experiments, never partial checkpoints.

    The two original model paths retain their arithmetic in this extension;
    artifact provenance records their original source hash rather than replacing it.
    """
    directory = Path(directory).resolve()
    original = json.loads((directory/"experiment.json").read_text(encoding="utf-8"))
    verified_sources = json.loads(Path(__file__).with_name("baseline_sources.json").read_text(encoding="utf-8")).get("compatible_completed_source_sha256", [])
    if original["source_sha256"] not in [source_sha256(), *verified_sources]:
        raise ValueError("reused source revision has not been checked for model compatibility")
    if original["training"] != cfg.to_dict() or original["dataset_sha256"] != manifest["prepared_sha256"] or original["device"] != device:
        raise ValueError("reused run must have identical train settings, seed, device, and prepared data")
    records = {}
    for name in ("cars", "transformer"):
        if name not in original["models"]:
            continue
        config = ModelConfig(**original["models"][name])
        if config.to_dict() != models[name].to_dict():
            raise ValueError(f"reused {name} model settings differ")
        final = directory/name/"final.pt"
        best = directory/name/"best.pt"
        if not final.exists() or not best.exists():
            raise ValueError(f"reused {name} must have completed the entire configured horizon")
        for checkpoint in (final, best):
            payload = read_checkpoint_payload(checkpoint)
            if ModelConfig(**payload["model_config"]).to_dict() != models[name].to_dict() or payload["train_config"] != cfg.to_dict():
                raise ValueError(f"reused {name} checkpoint settings differ from the experiment")
            if not 0 < payload["step"] <= cfg.max_steps or (checkpoint == final and payload["step"] != cfg.max_steps):
                raise ValueError(f"reused {name} must have completed the entire configured horizon")
        record = {"directory": str(directory), "original_source_sha256": original["source_sha256"]}
        for path in (final, best, directory/name/"metrics.jsonl"):
            record[path.name+"_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        records[name] = record
    if not records:
        raise ValueError("reused run contains no completed CARS/Transformer models")
    return records


def run_logged(command, log_path):
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with Path(log_path).open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND " + subprocess.list2cmdline(command) + "\n")
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
                if code:
                    raise RuntimeError(f"command failed ({code}); see {log_path}. Rerun the same comparison command to resume from latest.pt.")
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
                raise


def write_results(out, results, plan):
    (out / "comparison.json").write_text(json.dumps({"plan": plan, "models": results}, indent=2), encoding="utf-8")
    rows = []
    for name, result in results.items():
        rows.append({"architecture": name, "parameters": result["parameters"],
            "final_training_step": plan["training"]["max_steps"], "selected_best_step": result["checkpoint_step"],
            "training_tokens": plan["training_tokens_per_model"],
            "wiki_test_token_nll": result["wiki_test"]["mean_token_nll"],
            "wiki_byte_token_ppl": result["wiki_test"]["byte_token_perplexity"],
            "ownership_iid_accuracy": result["ownership_iid"]["owner_candidate_accuracy"],
            "ownership_longer_chain_accuracy": result["ownership_longer_chain"]["owner_candidate_accuracy"],
            "training_seconds": result["training_seconds"], "peak_cuda_memory_mb": result["peak_cuda_memory_mb"]})
    if rows:
        with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# CARS-R / Transformer / KDA / GDN-2 pilot", "", "Same data, tokenizer, token budget, schedule, and AdamW settings. Parameter sizes differ by less than 1%. Recurrent mixers use identical state dimensions and the same 3:1 mixer/MLA schedule. This is not a FLOP-matched comparison.", "", "| Model | Parameters | Wiki test token NLL | Ownership IID | Longer chain | Training hours |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['architecture']} | {row['parameters']:,} | {row['wiki_test_token_nll']:.4f} | {row['ownership_iid_accuracy']:.1%} | {row['ownership_longer_chain_accuracy']:.1%} | {row['training_seconds']/3600:.2f} |")
    lines += ["", "Models are selected by validation AR loss before testing. Test metrics use identical seeded windows/cases. Wiki metrics include byte/special tokens and are not comparable with published word perplexity. Ownership accuracy scores eight candidate owners after an answer-space token; final-recipient copying can solve these cases. One short seed is a pilot; repeat across seeds and use stronger state/order tests before claiming a quality advantage.", "", "KDA and GDN-2 are portable FLA-recipe PyTorch reference adaptations, with sigmoid output gating and shared MLA anchors. Their runtime does not represent upstream fused-kernel performance. Reused completed artifacts retain their original source provenance in experiment.json."]
    (out / "COMPARISON.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    # Portable browser-viewable plot, with no plotting dependency.
    series = {}
    for name in results:
        metrics = out / name / "metrics.jsonl"
        series[name] = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
    losses = [(name, m["step"], m["train_ar_loss"]) for name, items in series.items() for m in items if "train_ar_loss" in m]
    if losses:
        values = [entry[2] for entry in losses]
        minimum, maximum = min(values), max(values)
        span = max(maximum-minimum, 0.1)
        max_step = max(entry[1] for entry in losses)
        paths = []
        colors = {"cars": "#2563eb", "transformer": "#d97706", "kda": "#16a34a", "gdn2": "#9333ea"}
        for name in results:
            color = colors[name]
            points = " ".join(f"{60+step/max_step*680:.2f},{320-(value-minimum)/span*250:.2f}" for arch, step, value in losses if arch == name)
            paths.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400"><rect width="800" height="400" fill="white"/><text x="60" y="28" font-family="sans-serif" font-size="15">AR loss: CARS blue; Transformer orange; KDA green; GDN-2 purple</text><path d="M60 60V330H750" fill="none" stroke="#666"/>' + ''.join(paths) + f'<text x="60" y="370" font-family="sans-serif">Steps 0–{max_step}; loss range {minimum:.3f}–{maximum:.3f}</text></svg>'
        (out / "training_curve.svg").write_text(svg, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--data-dir", default="data/comparison_wikitext2")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--profile", choices=("micro", "pilot", "laptop"), default="laptop")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--reuse-run", default=None, help="import completed matching CARS/Transformer artifacts, train only additional models")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="validate settings/data/reused checkpoints and write the plan without training")
    parser.add_argument("--smoke", action="store_true", help="four steps/model and tiny eval/export; execution check only")
    args = parser.parse_args()
    if len(set(args.architectures)) != len(args.architectures):
        raise ValueError("architectures must not contain duplicates")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("steps must be positive")
    os.chdir(ROOT)
    data = Path(args.data_dir).resolve()
    manifest = prepare_dataset(data)
    if args.prepare_only:
        print(f"DATA READY: {data}")
        return
    device = args.device or ("cpu" if args.profile == "micro" else "cuda")
    # Match the training subprocess policy for model setup and CPU exports.
    torch.set_num_threads(1)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use a CUDA-enabled PyTorch build or --device cpu --smoke")
    steps = 4 if args.smoke else (args.steps or {"micro": 300, "pilot": 2000, "laptop": 1000}[args.profile])
    small = args.profile == "micro"
    laptop = args.profile == "laptop"
    cfg = TrainConfig(seed=args.seed, batch_size=1 if args.smoke else (8 if small else (32 if laptop else 16)), seq_len=32 if args.smoke else (80 if small else 128),
        max_steps=steps, grad_accum_steps=1,
        learning_rate=4e-4, state_lr_scale=1.0, weight_decay=0.1, grad_clip=1.0,
        beta1=0.9, beta2=0.95, warmup_steps=min(100, max(1, steps//10)),
        mtp_loss_weight=0.0, precision="bf16" if device == "cuda" else "fp32",
        log_interval=1, eval_interval=2 if args.smoke else (50 if small else 100),
        eval_batches=1 if args.smoke else (4 if small else (8 if laptop else 16)), save_interval=2 if args.smoke else (50 if small else 100),
        health_interval=2 if args.smoke else (50 if small else 100), gradient_checkpointing=not laptop, compile_model=False)
    cfg.validate()
    models = model_configs(args.profile)
    reused = validate_reuse(args.reuse_run, models, cfg, manifest, device) if args.reuse_run else {}
    selected = [name for name in ARCHITECTURES if name in args.architectures or name in reused]
    models = {name: models[name] for name in selected}
    counts = {name: CARSRLanguageModel(model).num_parameters() for name, model in models.items()}
    if max(counts.values())/min(counts.values()) > 1.01:
        raise ValueError("parameter matching exceeded 1% tolerance")
    tokens = steps * cfg.seq_len * cfg.batch_size * cfg.grad_accum_steps
    train_meta = json.loads((data / "train.bin.json").read_text(encoding="utf-8"))
    plan = {"training": cfg.to_dict(), "models": {name: model.to_dict() for name, model in models.items()},
        "parameter_counts": counts, "training_tokens_per_model": tokens,
        "effective_training_stream_passes": tokens/train_meta["tokens"], "dataset_revision": manifest["revision"],
        "dataset_sha256": manifest["prepared_sha256"], "source_sha256": source_sha256(),
        "comparison": "approximately parameter matched; same tokens/objective/optimizer; not FLOP matched",
        "device": device, "profile": args.profile, "smoke": args.smoke,
        "architectures": selected, "reused_completed_runs": reused,
        "baseline_sources": json.loads(Path(__file__).with_name("baseline_sources.json").read_text(encoding="utf-8"))}
    out = Path(args.out_dir or f"runs/comparison_{args.profile}_delta_seed{args.seed}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "experiment.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("existing experiment settings/code/data differ; use a new --out-dir rather than changing a resumed experiment")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    train_cfg_path = out / "train_config.json"
    train_cfg_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    print("PLAN: " + "; ".join(f"{name} {count:,} params" for name, count in counts.items()) + f"; {tokens:,} identical training tokens each ({plan['effective_training_stream_passes']:.3f} stream passes).", flush=True)
    print("PLAN: selected models run sequentially; completed imports skip training; evaluation, compilation, and exports follow automatically. Rerun this exact command to resume.", flush=True)
    if args.plan_only:
        print(f"PLAN VERIFIED: {plan_path}; reused completed models: {', '.join(reused) or 'none'}; training has not started.", flush=True)
        return
    results = {}
    for name, model in models.items():
        folder = out / name
        folder.mkdir(exist_ok=True)
        model_cfg_path = out / f"{name}_config.json"
        model.save_json(model_cfg_path)
        command = [sys.executable, "-u", "-m", "cars_r.train", "--model-config", str(model_cfg_path),
            "--train-config", str(train_cfg_path), "--bin", str(data/"train.bin"), "--val-bin", str(data/"validation.bin"),
            "--device", device, "--out-dir", str(folder)]
        final = folder / "final.pt"
        best = folder / "best.pt"
        if name in reused:
            imported = Path(reused[name]["directory"])
            final, best = imported/name/"final.pt", imported/name/"best.pt"
            shutil.copy2(imported/name/"metrics.jsonl", folder/"metrics.jsonl")
            print(f"REUSING completed {name} from {imported}", flush=True)
        elif not final.exists():
            latest = folder / "latest.pt"
            if latest.exists():
                command += ["--resume", str(latest)]
            print(f"START TRAINING: {name}", flush=True)
            run_logged(command, out / f"{name}_console.log")
        final_payload = read_checkpoint_payload(final)
        if final_payload["step"] != steps:
            raise ValueError("final checkpoint does not match experiment horizon")
        if not best.exists():
            raise ValueError("validation-selected best checkpoint is missing")
        print(f"EVALUATING: {name} validation-selected best.pt on held-out WikiText and ownership cases", flush=True)
        samples = 2 if args.smoke else (16 if small else (64 if laptop else 128))
        evaluation = evaluate_checkpoint(best, data, device=device, windows=samples, cases=samples)
        logs = [json.loads(line) for line in (folder/"metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        evaluation["training_seconds"] = final_payload.get("extra", {}).get("run_state", {}).get("training_seconds", 0.0)
        memories = [record["peak_cuda_memory_mb"] for record in logs if record.get("peak_cuda_memory_mb") is not None]
        evaluation["peak_cuda_memory_mb"] = max(memories) if memories else None
        (out / f"{name}_evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        print(f"COMPILING / EXPORTING: {name} for new prompts", flush=True)
        evaluation["export"] = export_model(best, out / "exports" / name, context=cfg.seq_len)
        results[name] = evaluation
        write_results(out, results, plan)
    print(f"COMPLETE: results {out / 'COMPARISON.md'}; CSV {out / 'comparison.csv'}; models {out / 'exports'}", flush=True)


if __name__ == "__main__":
    main()

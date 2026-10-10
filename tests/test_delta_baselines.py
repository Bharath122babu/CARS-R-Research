import ast
import math
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from cars_r import CARSRLanguageModel
from cars_r.delta_attention import DeltaMixer, delta_recurrence
from test_model import tiny

UPSTREAM = Path(__file__).resolve().parents[1]/"third_party/fla"


def upstream_definition(path, name, namespace):
    # Execute only the pinned PyTorch definition. Importing the FLA package
    # would pull in Linux/Triton dependencies, irrelevant to reference math.
    parsed = ast.parse((UPSTREAM/path).read_text(encoding="utf-8"))
    definition = next(x for x in parsed.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), definition], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(UPSTREAM/path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("architecture", ["kda", "gdn2"])
def test_recurrence_outputs_states_and_gradients_match_pinned_upstream(architecture):
    torch.manual_seed(25)
    q, k = [torch.randn(2, 9, 2, 4, requires_grad=True) for _ in range(2)]
    v = torch.randn(2, 9, 2, 3, requires_grad=True)
    g = (-torch.rand_like(q)*0.3).requires_grad_()
    initial = torch.randn(2, 2, 4, 3, requires_grad=True)
    if architecture == "kda":
        beta = torch.rand(2, 9, 2, requires_grad=True)
        erase, write = beta[..., None], beta[..., None]
        fn = upstream_definition("fla/ops/kda/naive.py", "naive_recurrent_kda", {"torch": torch})
        reference, reference_state = fn(q, k, v, g, beta, initial_state=initial, output_final_state=True)
        inputs = (q, k, v, g, beta, initial)
    else:
        erase = torch.rand_like(q, requires_grad=True)
        write = torch.rand_like(v, requires_grad=True)
        fn = upstream_definition("fla/ops/gdn2/naive.py", "naive_recurrent_gdn2", {"torch": torch})
        reference, reference_state = fn(q, k, v, g, erase, write, initial_state=initial, output_final_state=True)
        inputs = (q, k, v, g, erase, write, initial)
    actual, state = delta_recurrence(q, k, v, g, erase, write, initial_state=initial)
    torch.testing.assert_close(actual, reference, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(state, reference_state, rtol=3e-5, atol=3e-6)
    grad_a = torch.autograd.grad(actual.square().mean()+state.square().mean(), inputs, retain_graph=True)
    grad_b = torch.autograd.grad(reference.square().mean()+reference_state.square().mean(), inputs)
    for a, b in zip(grad_a, grad_b):
        torch.testing.assert_close(a, b, rtol=5e-5, atol=5e-6)


class ReferenceShortConv(nn.Conv1d):
    def __init__(self, hidden_size, kernel_size, bias=False, **kwargs):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, padding=kernel_size-1, bias=bias)

    def forward(self, x, **kwargs):
        y = super().forward(x.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        return F.silu(y), None


class ReferenceGatedRMS(nn.Module):
    def __init__(self, size, activation="sigmoid", eps=1e-5):
        super().__init__()
        assert activation == "sigmoid"
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x, gate):
        return x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+self.eps)*self.weight*gate.float().sigmoid()


@pytest.mark.parametrize("architecture", ["kda", "gdn2"])
def test_complete_mixer_recipe_matches_pinned_fla_layer(architecture):
    # Run the actual pinned layer class; replace only fused dispatch, short
    # convolution and fused normalization with their portable math definitions.
    cfg = tiny(architecture=architecture, delta_key_dim=8, delta_value_dim=6, mtp_steps=1)
    mixer = DeltaMixer(cfg)
    name = "KimiDeltaAttention" if architecture == "kda" else "GatedDeltaNet2"
    recurrent = upstream_definition(f"fla/ops/{architecture}/naive.py", f"naive_recurrent_{architecture}", {"torch": torch})

    def kernel(q, k, v, g, **kwargs):
        q = q.float()*torch.rsqrt(q.float().square().sum(-1, keepdim=True)+1e-6)
        k = k.float()*torch.rsqrt(k.float().square().sum(-1, keepdim=True)+1e-6)
        if architecture == "kda":
            log_decay = -kwargs["A_log"].float().exp()[None, None, :, None]*F.softplus(g.float()+kwargs["dt_bias"].reshape(cfg.n_heads, cfg.delta_key_dim))
            return recurrent(q, k, v, log_decay, kwargs["beta"].float().sigmoid())
        return recurrent(q, k, v, g, kwargs["b"], kwargs["w"])

    def rearrange(x, pattern, *, d=None):
        return x.reshape(*x.shape[:-1], x.shape[-1]//d, d) if "-> ... h d" in pattern else x.flatten(-2)

    namespace = {"torch": torch, "nn": nn, "math": math, "F": F, "rearrange": rearrange,
        "ShortConvolution": ReferenceShortConv, "FusedRMSNormGated": ReferenceGatedRMS,
        "get_layer_cache": lambda *args: None, "unpad_hidden_states": lambda x, *args: (x, None, None),
        "repad_hidden_states": lambda x, *args: x, "update_layer_cache": lambda *args, **kwargs: None,
        f"chunk_{architecture}": kernel, f"fused_recurrent_{architecture}": kernel}
    cls = upstream_definition(f"fla/layers/{architecture}.py", name, namespace)
    reference = cls(hidden_size=cfg.d_model, num_heads=cfg.n_heads, head_dim=cfg.delta_key_dim,
        expand_v=cfg.delta_value_dim/cfg.delta_key_dim)
    mapping = {"q_conv1d": "q_conv", "k_conv1d": "k_conv", "v_conv1d": "v_conv"}
    state = {mapping.get(key.split('.')[0], key.split('.')[0])+('.'+key.split('.', 1)[1] if '.' in key else ''): value for key, value in reference.state_dict().items()}
    mixer.load_state_dict(state)
    x = torch.randn(2, 11, cfg.d_model)
    torch.testing.assert_close(mixer(x), reference(x)[0], rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("architecture", ["kda", "gdn2"])
def test_delta_model_causal_all_gradients_and_masked_conv(architecture):
    torch.manual_seed(71)
    cfg = tiny(architecture=architecture, delta_key_dim=8, delta_value_dim=6, mtp_steps=1)
    model = CARSRLanguageModel(cfg)
    ids = torch.randint(0, 259, (2, 12))
    changed = ids.clone(); changed[:, 7:] = torch.randint(0, 259, (2, 5))
    logits = model(ids).logits
    torch.testing.assert_close(logits[:, :7], model(changed).logits[:, :7], rtol=2e-5, atol=2e-5)
    logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    mixer = model.blocks[0].attn
    x = torch.randn(1, 9, cfg.d_model)
    mask = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 1, 1]], dtype=torch.bool)
    masked = mixer(x, attention_mask=mask)
    torch.testing.assert_close(masked[:, mask[0]], mixer(x[:, mask[0]]), rtol=2e-5, atol=2e-5)
    assert masked[:, ~mask[0]].abs().sum() == 0


@pytest.mark.parametrize("architecture", ["kda", "gdn2"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_delta_cuda_mixed_precision_backward(architecture, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = CARSRLanguageModel(tiny(architecture=architecture, delta_key_dim=8, delta_value_dim=8, mtp_steps=1)).cuda()
    with torch.autocast("cuda", dtype=dtype):
        logits = model(torch.randint(0, 259, (2, 12), device="cuda")).logits
        loss = logits.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

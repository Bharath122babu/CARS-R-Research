from pathlib import Path
import random

import numpy as np
import pytest
import torch

from cars_r import ModelConfig, CARSRLanguageModel
from cars_r.checkpoint import (
    capture_runtime_state,
    load_checkpoint,
    read_checkpoint_payload,
    restore_runtime_state,
    save_checkpoint,
)
from cars_r.losses import causal_lm_loss


def tiny(**kw):
    values = dict(
        vocab_size=259, max_seq_len=64, d_model=64, n_layers=4, n_heads=2,
        d_ff=128, anchor_interval=4, cars_key_dim=16, cars_value_dim=16,
        cars_evidence_rank=2, cars_half_life_min=32.0, cars_half_life_max=512.0,
        mla_q_lora_rank=32, mla_kv_lora_rank=32,
        mla_qk_nope_dim=12, mla_qk_rope_dim=4, mla_v_dim=16, mtp_steps=2,
    )
    values.update(kw)
    return ModelConfig(**values)


def test_model_forward_backward_and_all_core_paths_have_gradients():
    torch.manual_seed(10)
    model = CARSRLanguageModel(tiny())
    x = torch.randint(0, 259, (2, 20))
    y = torch.randint(0, 259, (2, 20))
    out = model(x)
    loss = causal_lm_loss(out, y, mtp_weight=0.1)
    loss.total.backward()
    assert out.logits.shape == (2, 20, 259)
    must_have = [
        "blocks.0.mixer.in_proj.weight",
        "blocks.0.mixer.out_proj.weight",
        "blocks.3.attn.q_a.weight",
        "blocks.3.attn.kv_a.weight",
        "blocks.3.ff.in_proj.weight",
    ]
    params = dict(model.named_parameters())
    for name in must_have:
        assert params[name].grad is not None, name
        assert torch.isfinite(params[name].grad).all(), name
        assert int(torch.count_nonzero(params[name].grad)) > 0, name


def test_suffix_invariance():
    torch.manual_seed(11)
    model = CARSRLanguageModel(tiny()).eval()
    a = torch.randint(0, 259, (1, 30))
    b = a.clone()
    b[:, 21:] = torch.randint(0, 259, (1, 9))
    with torch.no_grad():
        ya = model(a).logits
        yb = model(b).logits
    torch.testing.assert_close(ya[:, :21], yb[:, :21], rtol=2e-5, atol=2e-5)


def test_static_schedule_is_3_to_1():
    c = tiny(n_layers=8, anchor_interval=4)
    assert [c.layer_type(i) for i in range(8)] == [
        "cars", "cars", "cars", "mla", "cars", "cars", "cars", "mla"
    ]


def test_pure_mla_and_pure_cars_ablation_configs_run():
    x = torch.randint(0, 259, (1, 12))
    pure_mla = CARSRLanguageModel(tiny(anchor_interval=1, n_layers=3))
    pure_cars = CARSRLanguageModel(tiny(anchor_interval=99, n_layers=3))
    assert torch.isfinite(pure_mla(x).logits).all()
    assert torch.isfinite(pure_cars(x).logits).all()


def test_hybrid_rejects_partial_cars_state_continuation():
    model = CARSRLanguageModel(tiny()).eval()
    x = torch.randint(0, 259, (1, 12))
    with torch.no_grad():
        first = model(x[:, :6])
    with pytest.raises(ValueError, match="MLA KV/position cache"):
        model(x[:, 6:], states=first.states)


def test_checkpoint_roundtrip_and_atomic_write(tmp_path: Path):
    torch.manual_seed(12)
    model = CARSRLanguageModel(tiny()).eval()
    x = torch.randint(0, 259, (1, 16))
    with torch.no_grad():
        expected = model(x).logits
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, None, step=7)
    loaded, payload = load_checkpoint(path)
    loaded.eval()
    with torch.no_grad():
        actual = loaded(x).logits
    assert payload["step"] == 7
    assert payload["checkpoint_version"] == 2
    assert not list(tmp_path.glob("*.tmp"))
    torch.testing.assert_close(actual, expected)


def _one_step(model, optimizer, gen):
    batch = torch.randint(0, 259, (2, 13), generator=gen)
    out = model(batch[:, :-1])
    loss = causal_lm_loss(out, batch[:, 1:], mtp_weight=0.1).total
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def test_resume_restores_training_trajectory(tmp_path: Path):
    # Continuous 4-step reference.
    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    cfg = tiny()
    model_a = CARSRLanguageModel(cfg)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    gen_a = torch.Generator().manual_seed(123)
    for _ in range(4):
        _one_step(model_a, opt_a, gen_a)

    # Same first two steps, checkpoint complete runtime state, reconstruct, then resume.
    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    model_b = CARSRLanguageModel(cfg)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    gen_b = torch.Generator().manual_seed(123)
    for _ in range(2):
        _one_step(model_b, opt_b, gen_b)
    runtime = capture_runtime_state(train_generator=gen_b)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, model_b, opt_b, step=2, runtime_states={"0": runtime})

    payload = read_checkpoint_payload(path)
    model_c = CARSRLanguageModel(cfg)
    model_c.load_state_dict(payload["model_state"])
    opt_c = torch.optim.AdamW(model_c.parameters(), lr=1e-3)
    opt_c.load_state_dict(payload["optimizer_state"])
    gen_c = torch.Generator().manual_seed(0)
    restore_runtime_state(payload["runtime_states"]["0"], train_generator=gen_c)
    for _ in range(2):
        _one_step(model_c, opt_c, gen_c)

    for pa, pc in zip(model_a.parameters(), model_c.parameters()):
        torch.testing.assert_close(pa, pc, rtol=0, atol=0)


def test_gradient_checkpointing_backward():
    torch.manual_seed(13)
    model = CARSRLanguageModel(tiny()).train()
    model.set_gradient_checkpointing(True)
    x = torch.randint(0, 259, (1, 12))
    y = torch.randint(0, 259, (1, 12))
    out = model(x)
    loss = causal_lm_loss(out, y, mtp_weight=0.1)
    loss.total.backward()
    assert model.blocks[0].mixer.in_proj.weight.grad is not None
    assert model.blocks[3].attn.q_a.weight.grad is not None
    assert torch.isfinite(model.blocks[0].mixer.in_proj.weight.grad).all()
    assert torch.isfinite(model.blocks[3].attn.q_a.weight.grad).all()


@pytest.mark.parametrize("masked", [False, True])
def test_sdpa_matches_explicit_causal_attention(masked, monkeypatch):
    import math
    import torch.nn.functional as functional
    from cars_r.attention import CausalMLA
    torch.manual_seed(30)
    attention = CausalMLA(tiny()).eval()
    x = torch.randn(2, 9, 64)
    mask = torch.tensor([[0, 1, 1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1, 1, 0]], dtype=torch.bool) if masked else None
    actual = attention(x, attention_mask=mask)
    def explicit(q, k, v, attn_mask=None, is_causal=False, dropout_p=0.0):
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        allowed = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril() if is_causal else attn_mask
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.nan_to_num(scores.softmax(-1), nan=0.0)
        return weights @ v
    monkeypatch.setattr(functional, "scaled_dot_product_attention", explicit)
    expected = attention(x, attention_mask=mask)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

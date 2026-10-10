import torch
import pytest
import copy

from cars_r import ModelConfig
from cars_r.cars_operator import CARSStateMixer


def cfg(**kw):
    d = dict(
        d_model=64, n_layers=4, n_heads=2, d_ff=128, max_seq_len=128,
        cars_key_dim=16, cars_value_dim=16, cars_evidence_rank=2,
        cars_half_life_min=32.0, cars_half_life_max=512.0,
        mla_q_lora_rank=32, mla_kv_lora_rank=32,
        mla_qk_nope_dim=12, mla_qk_rope_dim=4, mla_v_dim=16,
        mtp_steps=1,
    )
    d.update(kw)
    return ModelConfig(**d)


def test_operator_forward_backward_is_finite():
    torch.manual_seed(1)
    mixer = CARSStateMixer(cfg())
    x = torch.randn(2, 12, 64, requires_grad=True)
    y, state, diag = mixer(x, return_diagnostics=True)
    assert y.shape == x.shape
    assert state.matrix.shape == (2, 2, 16, 16)
    assert state.matrix.dtype == torch.float32
    assert diag is not None
    loss = y.square().mean() + state.matrix.square().mean() * 1e-3
    loss.backward()
    assert torch.isfinite(y).all()
    assert torch.isfinite(state.matrix).all()
    assert mixer.in_proj.weight.grad is not None
    assert torch.isfinite(mixer.in_proj.weight.grad).all()
    assert mixer.raw_ridge.grad is not None
    assert mixer.raw_decay_rate.grad is not None


def test_chunked_operator_equals_full_operator():
    torch.manual_seed(2)
    mixer = CARSStateMixer(cfg()).eval()
    x = torch.randn(1, 17, 64)
    with torch.no_grad():
        full, full_state, _ = mixer(x)
        a, state, _ = mixer(x[:, :7])
        b, state, _ = mixer(x[:, 7:12], state=state)
        c, state, _ = mixer(x[:, 12:], state=state)
    torch.testing.assert_close(torch.cat((a, b, c), dim=1), full, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state.matrix, full_state.matrix, rtol=1e-5, atol=1e-5)


def test_cayley_transport_is_norm_preserving():
    torch.manual_seed(3)
    mixer = CARSStateMixer(cfg()).eval()
    state = torch.randn(2, 2, 16, 16)
    u = torch.randn(2, 2, 16)
    w = torch.randn(2, 2, 16)
    strength = torch.randn(2, 2, 1)
    with torch.no_grad():
        moved, _ = mixer._transport(state, u, w, strength)
    torch.testing.assert_close(
        moved.flatten(2).norm(dim=-1),
        state.flatten(2).norm(dim=-1),
        rtol=3e-5,
        atol=3e-5,
    )


def test_read_before_write_has_no_current_write_shortcut():
    torch.manual_seed(4)
    mixer = CARSStateMixer(cfg(cars_read_before_write=True)).eval()
    x = torch.randn(1, 1, 64)
    state = mixer.initial_state(1, device=x.device, dtype=x.dtype)
    with torch.no_grad():
        y, next_state, _ = mixer(x, state=state)
    torch.testing.assert_close(y, torch.zeros_like(y), atol=1e-6, rtol=0)
    assert int(torch.count_nonzero(next_state.matrix)) > 0


def test_collapsed_transport_plane_is_finite_and_has_escape_gradient():
    torch.manual_seed(5)
    mixer = CARSStateMixer(cfg()).eval()
    state = torch.randn(1, 2, 16, 16)
    u = torch.randn(1, 2, 16, requires_grad=True)
    # exact collapse: w == u numerically, but keep an independent leaf so we can
    # verify the smooth Cayley parametrization provides an escape gradient.
    w = u.detach().clone().requires_grad_(True)
    strength = torch.ones(1, 2, 1, requires_grad=True)
    moved, _ = mixer._transport(state, u, w, strength)
    assert torch.isfinite(moved).all()
    torch.testing.assert_close(
        moved.flatten(2).norm(dim=-1),
        state.flatten(2).norm(dim=-1),
        rtol=3e-5, atol=3e-5,
    )
    loss = (moved * torch.randn_like(moved)).sum()
    loss.backward()
    assert w.grad is not None
    assert torch.isfinite(w.grad).all()
    assert float(w.grad.norm()) > 0.0


def test_transport_strength_rows_start_near_identity():
    torch.manual_seed(6)
    mixer = CARSStateMixer(cfg(cars_transport_init_std=1e-5))
    mixer.apply_architecture_initialization()
    rows = torch.arange(mixer.h) * mixer.per_head + (mixer.per_head - 1)
    angle_rows = mixer.in_proj.weight[rows]
    assert float(angle_rows.detach().std()) < 5e-5


def test_heads_start_with_diverse_half_lives():
    mixer = CARSStateMixer(cfg(cars_half_life_min=16.0, cars_half_life_max=1024.0))
    hl = mixer.state_half_life.detach()
    assert torch.all(hl[1:] > hl[:-1])
    assert abs(float(hl[0]) - 16.0) < 0.1
    assert abs(float(hl[-1]) - 1024.0) < 1.0


@pytest.mark.parametrize("mode", ["raw", "rmsnorm", "rmsclip"])
def test_batched_projections_match_token_steps_and_gradients(mode):
    torch.manual_seed(21)
    batched = CARSStateMixer(cfg(cars_read_mode=mode))
    sequential = copy.deepcopy(batched)
    x = torch.randn(2, 9, 64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    mask = torch.tensor([[1, 1, 0, 1, 1, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1, 1, 1, 0]], dtype=torch.bool)
    actual, state, _ = batched(x, attention_mask=mask)
    reference_state = sequential.initial_state(2, device=x.device, dtype=x.dtype)
    outputs = []
    for i in range(x.shape[1]):
        y, reference_state, _ = sequential.step(x_ref[:, i], reference_state, active=mask[:, i])
        outputs.append(y)
    expected = torch.stack(outputs, dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(state.matrix, reference_state.matrix, rtol=2e-5, atol=2e-5)
    (actual.square().mean() + state.matrix.square().mean()).backward()
    (expected.square().mean() + reference_state.matrix.square().mean()).backward()
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=3e-4, atol=3e-5)
    for p, reference in zip(batched.parameters(), sequential.parameters()):
        torch.testing.assert_close(p.grad, reference.grad, rtol=3e-4, atol=3e-5)


def test_rmsclip_preserves_small_reads_and_bounds_large_reads():
    mixer = CARSStateMixer(cfg(cars_read_mode="rmsclip", cars_read_clip=0.5))
    small = torch.full((1, 2, 16), 1e-4)
    torch.testing.assert_close(mixer._process_read(small), small * mixer.read_scale[:, None])
    large = torch.full((1, 2, 16), 1e4, requires_grad=True)
    clipped = mixer._process_read(large)
    bound = 0.5 * mixer.read_scale
    assert torch.all(clipped.square().mean(-1).sqrt() <= bound[None] + 1e-6)
    clipped.sum().backward()
    assert torch.isfinite(large.grad).all()


@pytest.mark.parametrize("transport", [False, True])
def test_additive_control_keeps_live_memory(transport):
    mixer = CARSStateMixer(cfg(cars_write_mode="additive", cars_transport_enabled=transport))
    y, state, _ = mixer(torch.randn(1, 5, 64))
    assert torch.count_nonzero(state.matrix) > 0
    assert torch.count_nonzero(y[:, 1:]) > 0
    y.square().mean().backward()
    assert torch.isfinite(mixer.in_proj.weight.grad).all()


def test_no_write_is_explicitly_a_zero_memory_control():
    mixer = CARSStateMixer(cfg(cars_write_mode="none"))
    y, state, _ = mixer(torch.randn(1, 5, 64))
    assert torch.count_nonzero(y) == 0
    assert torch.count_nonzero(state.matrix) == 0


def test_read_only_transport_changes_reads_without_changing_stored_state():
    torch.manual_seed(25)
    control = CARSStateMixer(cfg(cars_transport_read_only=True, cars_transport_init_std=0.1))
    no_transport = copy.deepcopy(control)
    no_transport.config = cfg(cars_transport_enabled=False)
    x = torch.randn(1, 8, 64)
    a, a_state, _ = control(x)
    b, b_state, _ = no_transport(x)
    torch.testing.assert_close(a_state.matrix, b_state.matrix, rtol=0, atol=0)
    assert not torch.allclose(a, b)


def test_low_rank_transport_matches_dense_cayley():
    torch.manual_seed(26)
    mixer = CARSStateMixer(cfg())
    state = torch.randn(2, 2, 16, 16)
    u = torch.nn.functional.normalize(torch.randn(2, 2, 16), dim=-1)
    w = torch.nn.functional.normalize(torch.randn(2, 2, 16), dim=-1)
    raw = torch.randn(2, 2, 1)
    eta = raw[..., 0].tanh() * mixer.config.cars_transport_max_strength
    skew = eta[..., None, None] * (u[..., :, None] * w[..., None, :] - w[..., :, None] * u[..., None, :])
    eye = torch.eye(16)
    dense = torch.linalg.solve(eye - skew / 2, eye + skew / 2)
    actual, _ = mixer._transport(state, u, w, raw)
    torch.testing.assert_close(actual, state @ dense, rtol=2e-5, atol=2e-5)


def test_ridge_update_satisfies_regularized_least_squares_optimum():
    torch.manual_seed(27)
    mixer = CARSStateMixer(cfg())
    state = torch.randn(2, 2, 16, 16)
    k = torch.nn.functional.normalize(torch.randn(2, 2, 16, 2), dim=-2)
    v = torch.randn(2, 2, 16, 2)
    updated, _ = mixer._correct(state, k, v)
    gradient = ((updated @ k - v) @ k.transpose(-1, -2)
                + mixer.ridge[None, :, None, None] * (updated - state))
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=2e-6)


def test_interference_probe_no_distractors_preserves_historical_reads():
    from cars_r.eval_memory_interference import probe
    report = probe(CARSStateMixer(cfg()), memories=4, distractors=0, trials=2)
    assert report["before_distractors"] == report["after_distractors"]
    assert report["historical_read_change_rms_per_head"] == [0.0, 0.0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_autocast_keeps_recurrence_math_fp32(dtype, monkeypatch):
    mixer = CARSStateMixer(cfg()).cuda()
    original = mixer._transport
    def checked_transport(*args):
        assert not torch.is_autocast_enabled("cuda")
        transported, strength = original(*args)
        assert transported.dtype == strength.dtype == torch.float32
        return transported, strength
    monkeypatch.setattr(mixer, "_transport", checked_transport)
    x = torch.randn(2, 12, 64, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=dtype):
        y, state, diag = mixer(x, return_diagnostics=True)
        loss = y.float().square().mean() + state.matrix.square().mean()
    assert state.matrix.dtype == diag.innovation_norm.dtype == torch.float32
    assert torch.isfinite(state.matrix).all()
    loss.backward()
    assert torch.isfinite(x.grad).all()

"""Regression checks for reviewed runner failures and public operator behavior."""
import contextlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cars_fast as fast
from cars_r.cars_operator import CARSState, CARSStateMixer
from cars_r.config import ModelConfig


def test_zero_decay_and_nonzero_state_mask_preservation():
    was_enabled = fast._PATCHED['on']
    fast.disable_fast_path()
    torch.manual_seed(42)
    m = CARSStateMixer(ModelConfig(**json.loads(fast._model_config_path().read_text())))
    with torch.no_grad(): m.raw_decay_rate.fill_(1000)
    x = torch.randn(2, 5, m.config.d_model, requires_grad=True)
    initial = torch.randn(2, m.h, m.dv, m.dk)
    mask = torch.tensor([[False] * 5, [True, False, True, False, True]])
    y0, s0, _ = m(x, state=CARSState(initial), attention_mask=mask)
    if was_enabled: fast.enable_fast_path()
    y1, s1 = fast.fused_forward(m, x, state=initial, attention_mask=mask)
    torch.testing.assert_close(y1, y0, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(s1, s0.matrix, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(s1[0], initial[0], rtol=0, atol=0)
    y1.square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_invalid_mask_uses_reference_validation():
    m = CARSStateMixer(ModelConfig(**json.loads(fast._model_config_path().read_text())))
    was_enabled = fast._PATCHED['on']
    fast.enable_fast_path()
    try:
        with pytest.raises(ValueError, match='attention_mask'):
            m(torch.randn(2, 5, m.config.d_model), attention_mask=torch.ones(2, 4))
    finally:
        if not was_enabled:
            fast.disable_fast_path()


def test_failed_targets_cannot_produce_winning_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(fast, 'FAST_RUNS', tmp_path)
    for arm, nll, acc, minutes in (('og', 1.54, 1.0, 10), ('lr8', 1.52, .5, 60)):
        for seed in (1337, 1338, 1339):
            name = f'{arm}_seed{seed}'
            folder = tmp_path / name
            folder.mkdir()
            (folder / 'fast_eval.json').write_text(json.dumps({
                'run': name, 'wiki_nll': nll, 'wiki_windows': 1024, 'selected_step': 1000,
                'training_minutes': minutes,
                'accuracy': {'iid_original_500': acc, 'iid_fresh': acc, 'five_transfers': acc}}))
    output = io.StringIO()
    with contextlib.redirect_stdout(output): fast.cmd_summary(SimpleNamespace())
    assert 'OWNERSHIP FAIL' in output.getvalue()
    assert '-> ACCEPT' not in output.getvalue()
    assert 'candidate for matched-seed review' not in output.getvalue()


def test_resume_refuses_changed_execution_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.setattr(fast, 'FAST_RUNS', tmp_path)
    out = tmp_path / 'og_seed1337'
    out.mkdir()
    # A completed run must also be checked before any early return.
    (out / 'final.pt').write_bytes(b'fixture')
    contract = out / 'execution_contract.json'
    contract.write_text('{"execution":"previous-different-implementation"}')
    before = contract.read_bytes()
    args = SimpleNamespace(arm='og', reference=True, seed=1337, steps=None,
                           override=None, out_dir=None, data_dir=fast.DATA, device='cpu')
    with pytest.raises(SystemExit, match='contract changed'):
        fast.cmd_train(args)
    assert contract.read_bytes() == before


def test_protected_project_cannot_be_training_destination():
    with pytest.raises(SystemExit, match='outputs must'):
        fast.require_experiment_output(fast.ORIGINAL_RUN / 'cars')

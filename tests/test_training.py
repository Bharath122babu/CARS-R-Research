import json
import sys

import pytest
import torch

from cars_r.config import ModelConfig, TrainConfig
from cars_r.checkpoint import capture_runtime_state, restore_runtime_state, read_checkpoint_payload
from cars_r.train import main
from test_model import tiny


@pytest.mark.parametrize("architecture", ["cars", "kda", "gdn2"])
def test_cli_training_resume_is_exact_with_dropout_accumulation_and_health(tmp_path, monkeypatch, architecture):
    source = tmp_path / "corpus.txt"
    source.write_text("Ava gives the key to Ben. Ben gives it to Cora.\n" * 200, encoding="utf-8")
    model_path = tmp_path / "model.json"
    tiny(architecture=architecture, dropout=0.1, cars_read_mode="rmsclip").save_json(model_path)
    train_path = tmp_path / "train.json"
    train = TrainConfig(batch_size=1, seq_len=12, max_steps=4, grad_accum_steps=2,
                        precision="fp32", warmup_steps=1, eval_interval=2, eval_batches=1,
                        save_interval=2, log_interval=1, health_interval=2)
    train_path.write_text(json.dumps(train.to_dict()), encoding="utf-8")
    def run(directory, *extra):
        monkeypatch.setattr(sys, "argv", ["cars-train", "--model-config", str(model_path),
            "--train-config", str(train_path), "--text", str(source), "--device", "cpu",
            "--out-dir", str(directory), *extra])
        main()
    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    run(continuous)
    run(resumed, "--stop-after", "2")
    assert not (resumed / "final.pt").exists()
    run(resumed, "--resume", str(resumed / "latest.pt"))
    a = read_checkpoint_payload(continuous / "final.pt")
    b = read_checkpoint_payload(resumed / "final.pt")
    assert a["step"] == b["step"] == 4
    for key in a["model_state"]:
        torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)
    health = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert any("cars_health" in record for record in health)
    # A changed corpus must never be silently presented as exact resume.
    source.write_text(source.read_text() + "different document\n", encoding="utf-8")
    with pytest.raises(ValueError, match="run contract"):
        run(resumed, "--resume", str(resumed / "latest.pt"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_mapped_rng_tensors_restore_on_cpu_generators():
    generator = torch.Generator().manual_seed(92)
    state = capture_runtime_state(train_generator=generator)
    expected = torch.randint(0, 1000, (16,), generator=generator)
    state["torch_cpu_rng"] = state["torch_cpu_rng"].cuda()
    state["train_generator"] = state["train_generator"].cuda()
    state["torch_cuda_rng_all"] = [item.cuda() for item in state["torch_cuda_rng_all"]]
    restore_runtime_state(state, train_generator=generator)
    torch.testing.assert_close(torch.randint(0, 1000, (16,), generator=generator), expected)


@pytest.mark.parametrize("change", [{"cars_read_mode": "invalid"}, {"cars_write_mode": "invalid"},
                                  {"cars_read_clip": 0}, {"cars_ridge_init": float("nan")}])
def test_invalid_model_configs_are_rejected(change):
    with pytest.raises(ValueError):
        ModelConfig(**change).validate()

import argparse
import json

import pytest
import torch

from cars_r import ModelConfig, TrainConfig, CARSRLanguageModel
from cars_r.checkpoint import save_checkpoint, read_checkpoint_payload
from cars_r.compare import model_configs, source_sha256, validate_reuse
from cars_r.comparison_data import build_relational_splits
from cars_r.data import document_aware_byte_tokens
from cars_r.export_model import export_model
from cars_r.infer import compiled_generate
from cars_r.train import load_sources
from test_model import tiny


def test_conventional_transformer_causality_and_all_gradients():
    torch.manual_seed(80)
    model = CARSRLanguageModel(tiny(architecture="transformer", mtp_steps=1)).eval()
    ids = torch.randint(0, 259, (2, 12))
    alternative = ids.clone()
    alternative[:, 8:] = torch.randint(0, 259, (2, 4))
    output = model(ids)
    torch.testing.assert_close(output.logits[:, :8], model(alternative).logits[:, :8], rtol=2e-5, atol=2e-5)
    assert output.states == ()
    output.logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    with pytest.raises(ValueError, match="cache"):
        model(ids, states=[])


@pytest.mark.parametrize("profile", ["micro", "pilot", "laptop"])
def test_comparison_parameter_and_objective_matching(profile):
    configs = model_configs(profile)
    counts = [CARSRLanguageModel(config).num_parameters() for config in configs.values()]
    if profile == "laptop":
        assert all(1_000_000 <= count <= 4_000_000 for count in counts)
    assert max(counts)/min(counts) < 1.01
    assert all(config.mtp_steps == 1 for config in configs.values())
    for field in ("d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len", "dropout"):
        assert getattr(configs["cars"], field) == getattr(configs["transformer"], field)


def test_generated_task_splits_are_disjoint_and_reproducible(tmp_path):
    first, second = tmp_path/"first", tmp_path/"second"
    a = build_relational_splits(first, counts=(20, 10, 10), seed=4)
    b = build_relational_splits(second, counts=(20, 10, 10), seed=4)
    assert a == b
    sets = [{line for line in (first/f"relations_{split}.jsonl").read_text().splitlines()} for split in ("train", "validation", "test")]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    for line in sets[0]:
        example = json.loads(line)
        assert len(example["prompt"].encode()+example["target"].encode())+3 < 128


def test_separate_validation_uses_full_training_corpus(tmp_path):
    train, val = tmp_path/"train.txt", tmp_path/"val.txt"
    train.write_text("train document\n"*20, encoding="utf-8")
    val.write_text("held out document\n"*10, encoding="utf-8")
    args = argparse.Namespace(text=str(train), bin=None, val_text=str(val), val_bin=None)
    source, validation = load_sources(args, ModelConfig(), TrainConfig())
    assert len(source.tokens) == len(document_aware_byte_tokens(train.read_bytes()))
    assert len(validation.tokens) == len(document_aware_byte_tokens(val.read_bytes()))


def test_reuse_requires_complete_matching_checkpoint_settings_and_known_source(tmp_path):
    configs = {name: tiny(architecture=name, mtp_steps=1) for name in ("cars", "transformer")}
    cfg = TrainConfig(max_steps=4, precision="fp32")
    manifest = {"prepared_sha256": {"train.bin": "fixed-data"}}
    plan = {"training": cfg.to_dict(), "models": {n: c.to_dict() for n, c in configs.items()},
        "dataset_sha256": manifest["prepared_sha256"], "device": "cpu", "source_sha256": source_sha256()}
    (tmp_path/"experiment.json").write_text(json.dumps(plan), encoding="utf-8")
    for name, config in configs.items():
        folder = tmp_path/name; folder.mkdir()
        model = CARSRLanguageModel(config)
        save_checkpoint(folder/"final.pt", model, None, step=4, train_config=cfg)
        save_checkpoint(folder/"best.pt", model, None, step=2, train_config=cfg)
        (folder/"metrics.jsonl").write_text('{}\n', encoding="utf-8")
    assert set(validate_reuse(tmp_path, configs, cfg, manifest, "cpu")) == {"cars", "transformer"}
    changed = TrainConfig(**cfg.to_dict()); changed.seed += 1
    with pytest.raises(ValueError, match="identical train settings"):
        validate_reuse(tmp_path, configs, changed, manifest, "cpu")
    plan["source_sha256"] = "unverified-previous-revision"
    (tmp_path/"experiment.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="model compatibility"):
        validate_reuse(tmp_path, configs, cfg, manifest, "cpu")
    plan["source_sha256"] = source_sha256()
    (tmp_path/"experiment.json").write_text(json.dumps(plan), encoding="utf-8")
    save_checkpoint(tmp_path/"cars/final.pt", CARSRLanguageModel(configs["cars"]), None, step=3, train_config=cfg)
    with pytest.raises(ValueError, match="entire configured horizon"):
        validate_reuse(tmp_path, configs, cfg, manifest, "cpu")


@pytest.mark.parametrize("architecture", ["cars", "transformer", "kda", "gdn2"])
def test_compiled_export_matches_source_and_generates_on_other_input(tmp_path, architecture):
    model = CARSRLanguageModel(tiny(architecture=architecture, mtp_steps=1))
    checkpoint = tmp_path/"source.pt"
    save_checkpoint(checkpoint, model, torch.optim.AdamW(model.parameters()), step=3)
    bundle = tmp_path/"export"
    manifest = export_model(checkpoint, bundle, context=8)
    assert manifest["equivalence"].startswith("PASS")
    assert (bundle/"compiled_cpu.pt").exists()
    payload = read_checkpoint_payload(bundle/"model.pt")
    assert payload["optimizer_state"] is None and payload["extra"]["inference_only"]
    assert isinstance(compiled_generate(bundle, "abc", 2), str)

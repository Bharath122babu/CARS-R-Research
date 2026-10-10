# Reproducibility and workflows

The measured results predate the standalone cleanup. Model package source
`cars_r/*.py` is preserved byte-for-byte from OG v0.1.5, source SHA-256
`1c6eb29e6e3dd84a4e1d2f30ec560d04ced855084237dab9526239df36ab1914`.
Only standalone runners, metadata, examples, documentation and workflows were
added/adapted. The old project and experiment folders remain separate.

Qualification binds runner/package/config/data/reference-weight hashes and
software/GPU identity. A historical report under `reports/` cannot authorize
new training: this checkout creates its own `runs/fast/QUALIFICATION.json`.
The epoch runner checks its own source/sampler contract before resume.
Original atomic checkpoint writing, optimizer/scaler/RNG restoration and
original validation selection remain in use.

## Local validation

```bash
python -m pytest tests --basetemp=runs/test-temp
python scripts/verify_repository.py
python cars_epoch.py --epochs 1 --plan-only
```

Tests cover model causality, operator states/gradients, upstream baseline
recurrences, exact training resume, public validation, unsafe resume contracts,
sampler coverage/continuation and ignored padding gradients. CUDA tests skip
when a GPU is absent. The `--plan-only` command reads/verifies data and prints
the exact epoch plan; it does not update model weights.

## GitHub Actions

`ci.yml` configures Windows/Linux CPU tests, with both the original operator
and the opt-in faster adapter, plus repository-asset validation. GitHub jobs
do not need private checkpoints or the training data cache.
`docs.yml` checks the documentation on relevant changes. These are prepared
workflow definitions, not proof of an already completed GitHub run.

Workflow setup follows the official
[GitHub Python guide](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
and uses the tested Torch 2.6 CPU build listed in
[PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).
The GPU benchmark is tied to the local RTX 4070 Laptop setup; CPU CI does not
establish GPU throughput.

## Review before public posting

The folder has a local Git repository with source/config/docs/results tracked.
Data, weights, run outputs, caches and build artifacts are excluded through
`.gitignore`. No remote or fabricated repository URL is configured. Add the
actual GitHub remote when the user chooses to publish. The optional weights
and dataset can remain local or be published separately with their manifests.

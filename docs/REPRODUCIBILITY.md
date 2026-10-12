# Reproducibility and workflows

The historical 1,000-step results predate the standalone cleanup; a completed
matched epoch comparison is now included in [RESULTS.md](RESULTS.md) and
[`../reports/epoch1_comparison.json`](../reports/epoch1_comparison.json).
Model package source
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
`docs.yml` checks the documentation on relevant changes. Both workflows run
on the published repository; [Actions](https://github.com/Bharath122babu/CARS-R-Research/actions)
shows the current results for each commit.

Workflow setup follows the official
[GitHub Python guide](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
and uses the tested Torch 2.6 CPU build listed in
[PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).
The GPU benchmark is tied to the local RTX 4070 Laptop setup; CPU CI does not
establish GPU throughput.

## Published repository

The folder has a local Git repository with source/config/docs/results tracked.
Data, weights, run outputs, caches and build artifacts are excluded through
`.gitignore`. The public repository is
[Bharath122babu/CARS-R-Research](https://github.com/Bharath122babu/CARS-R-Research),
published on 10 October 2026. Optional weights and the prepared dataset remain
local; a source-only clone regenerates the pinned dataset as described in the
training guide.

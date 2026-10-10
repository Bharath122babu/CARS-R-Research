# Training and evaluation

![Workflow](assets/workflow.svg)

Run commands from the repository root. Install a CUDA-enabled PyTorch for GPU
training, then `python -m pip install -e ".[test,comparison]"`. Tested hardware
is an RTX 4070 Laptop GPU, Python 3.12, Torch 2.6.0+cu124. CPU unit tests are
available; the epoch launcher intentionally requires CUDA.

## One real epoch

```cmd
python -u cars_epoch.py --epochs 1
```

Or double-click `START_EPOCH.cmd`. This starts **fresh training**, with seed
1337, in `runs/fast/epoch1_seed1337`; it does not continue the earlier 1,000-step
weights. It resumes its own latest checkpoint if the run is interrupted.

The 11,824,795-token stream has **11,824,794 prediction targets**. Split into
chunks with 128 targets and a shared one-token boundary input, shuffle chunk
IDs once and consume every chunk exactly once. Batch size is 32, giving
**2,887 updates**. The last partial chunk/batch uses PAD labels ignored by
cross entropy; trailing PAD cannot affect earlier causal predictions.

`epoch_progress.jsonl` records real prediction-token coverage. The inherited
trainer's `tokens_seen` counts padded tensor positions, so use epoch progress
for the exact corpus coverage. Sampler progress is logged at batch sampling;
optimizer completion is recorded by `metrics.jsonl` and saved checkpoints.
After a restart from an older checkpoint, uncommitted log rows may repeat.

AdamW betas=(0.9,0.95), peak LR=4e-4, weight decay=0.1 for eligible matrix
parameters, gradient clip=1, state LR multiplier=1, warmup=100 steps and cosine
floor=4e-5 are retained. The cosine horizon is **2,887 steps**, so this is a new
budget/sampling experiment, not an identical repetition of the 1,000-step run.
Mixed validation is checked every 100 steps and at the final step. The best
validation AR loss selects the exported/evaluated checkpoint.

At completion the command evaluates 1,024 held-out Wiki windows, original
ownership IID, 2,000 unseen ownership cases, five-transfer cases and both
two-object stress suites. It exports `best.pt` using the original model code.

## Logs and artifacts

```text
runs/fast/epoch1_seed1337/
  epoch_contract.json       code/data/config/hardware + sampler identity
  epoch_progress.jsonl      real targets and epoch fraction
  metrics.jsonl             every update, validation and health probes
  latest.pt / best.pt        resumable and validation-selected checkpoints
  final.pt                  completed full-horizon checkpoint
  fast_eval.json            final evaluation and prediction arrays
  evaluation_cases.json     exact generated test cases
  export/model.pt           inference weights
  export/compiled_cpu.pt    verified fixed-context CPU TorchScript
```

Rerun the same command to resume. Changed code, data, horizon, seed or hardware
requires a distinct experiment. Use `--seed 1338` for a new independently
initialized run. An interrupted run is not marked complete without `final.pt`.

## Original random-window comparison

```bash
python cars_fast.py qualify
python cars_fast.py train --arm og --seed 1337
python cars_fast.py evaluate --run runs/fast/og_seed1337 --reference
python cars_fast.py summary
```

The original 1,000-step sampler draws random windows **with replacement**. Its
4.096M tokens are only 0.346 corpus-length equivalents and do not guarantee
that every training target has been visited. This explains why an epoch run
needs a different sampler rather than just renaming a step count.

Qualification checks math/gradients and performs short disposable optimizer
benchmarks; it does not start a research training run. A local historical
reference checkpoint and Wiki data enable a trained-checkpoint NLL check.
On a source-only clone, a seeded untrained model provides a numerical NLL
comparison instead. Reports explicitly identify which check was performed.

## Optional accuracy arms

`--arm lr8` doubles peak LR to 8e-4 **and** the proportional floor to 8e-5.
`--arm state10` scales decay/ridge/read-scale LR by 10. Neither has demonstrated
an accuracy improvement in the reported comparison. Train/evaluate every
matched seed before comparing, retain IID ownership and use validation to
choose settings before a fresh final test. The summary's seed-spread heuristic
is only a candidate screen, not statistical proof.

## Baselines and inference

```bash
python -m cars_r.compare --help
python -m cars_r.infer pretrained/fast_og_seed1337 "A owns x. A gives x to B. Owner of x:" --max-new-tokens 2 --device cpu
python -m cars_r.infer pretrained/fast_og_seed1337 "A owns x. A gives x to B. Owner of x:" --max-new-tokens 2 --compiled
```

Pretrained weights are present in this local folder but Git-ignored. A
source-only clone uses its own exported checkpoint after training. The
compiled option is CPU inference at fixed context, not GPU training speedup.

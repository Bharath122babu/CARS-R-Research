# Local assets versus Git source

This prepared folder is immediately usable on the existing laptop. It contains
the exact original prepared dataset, the original validation-selected CARS
inference weights in `references/original/cars/best.pt`, and the measured fast
model in `pretrained/fast_og_seed1337/`. These weights are small local artifacts,
not dependencies of model construction or CPU unit tests.

The pretrained folder also contains `compiled_cpu.pt`, `model_config.json`,
`tokenizer.json`, and `export.json`. Original reference configs and metrics are
tracked; weights and dataset caches are Git-ignored so ordinary `git add` does
not push them. [`local_assets_manifest.json`](local_assets_manifest.json)
records their relative filenames and exact SHA-256 values.

A future GitHub clone contains the source/config/docs/results. Its epoch
command recreates data automatically. Missing historical reference weights
cause qualification to use a seeded untrained model for numerical comparison;
that test is explicitly labeled and does not claim historical trained accuracy.
Training creates a fresh model and its own exported checkpoint. No weights
download URL is fabricated. Historical weights can be distributed as separate
release assets later if desired.

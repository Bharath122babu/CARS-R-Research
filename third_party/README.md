# Pinned baseline reference code

`fla/` retains MIT-licensed Flash Linear Attention reference files and its
upstream LICENSE/SOURCES.json. The tests execute selected naive PyTorch
definitions to verify the portable KDA/GDN-2 recurrences, without importing
Triton or installing the upstream package. Runtime adaptation/provenance is
also recorded in `cars_r/baseline_sources.json`.

Cached non-runtime NVIDIA source copies and repository tree inventories were
excluded during cleanup. No NVIDIA implementation is relicensed here. The
portable baselines use the MIT FLA recipes, including their sigmoid output
gate, and preserve the common CARS-R hybrid backbone. They are local reference
baselines, not full upstream training systems.

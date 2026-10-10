# Standalone build validation — 2026-10-10

The prepared copy is independent of the existing project and experiment
folders. At preparation time it was an unpacked local Git repository without a
remote. It was subsequently published as
[Bharath122babu/CARS-R-Research](https://github.com/Bharath122babu/CARS-R-Research)
on 10 October 2026; the checks below record the local preparation stage.
No new full research training or epoch run was started while preparing it.

| Check | Outcome |
|---|---|
| Full suite with fast adapter, local RTX laptop | 74 passed |
| Tracked-files-only checkout, CPU simulated before collection | 67 passed, 7 CUDA-dependent skips |
| GPU mathematical qualification | PASS; FP64/FP32 outputs, states and gradients |
| Trained original reference, 128 Wiki windows | Reference NLL 1.53213; fast 1.53215; difference 0.000021 |
| Epoch plan / verified local corpus | 2,887 updates; 11,824,794 real prediction targets |
| Repository docs/diagrams | 8 initial main documents; six SVG/PNG pairs; all links and XML valid |
| Local data/reference/pretrained assets | 23 SHA-256 hashes verified |
| Source-only assets | Tracked metadata verified; optional data/weights absent as expected |
| GitHub workflow YAML | Both workflow files parsed locally |
| Core package | Byte-identical OG v0.1.5 source hash preserved |

The CPU source-only check simulates GPU absence on the laptop; it is not an
already-completed hosted GitHub job. Workflow execution on Windows/Linux
remains for after public posting. The mathematical qualification includes
short disposable optimizer benchmarks, not a new trained model. The reported
3.78× speedup is from the completed historical 1,000-step experiment; current
short timings vary with system conditions and are not substituted for it.

The first fresh-directory test invocation exposed a missing `runs/` parent for
Pytest's temporary directory. A fixture now creates the parent automatically;
the complete rerun passed. Cache/build/temporary test outputs are excluded
from the prepared source and removed from this local clean copy after checks.

Logs are in `reports/standalone_fast_tests.log`,
`reports/standalone_gpu_qualification.log`,
`reports/standalone_source_only_tests.log` and
`reports/standalone_source_only_docs.log`. The verified historical
weights and binary corpus stay local and Git-ignored.

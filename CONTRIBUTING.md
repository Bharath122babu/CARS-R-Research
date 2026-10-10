# Contributing

Install `python -m pip install -e ".[test,comparison]"`, then run
`python -m pytest tests --basetemp=runs/test-temp` and
`python scripts/verify_repository.py`.

Keep the original reference recurrence available when changing execution.
Compare outputs, state and gradients with masks, nonzero state and difficult
key geometry before making throughput claims. Benchmark matched batches and
report logged training time separately from full wall time/evaluation.

For architecture/data changes, choose new output directories and retain
validation selection, exact run contracts, original/fresh ownership cases
and two-object stress tests. Report all seeds and failed categories. Do not
describe single-object candidate accuracy as general reasoning accuracy.

Regenerate diagrams using `python scripts/make_figures.py` after installing
`.[docs]`. Keep SVG text readable and PNG copies available. Preserve third-party
licenses and pinned provenance. Add regressions for observed failures.

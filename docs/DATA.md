# Dataset and task construction

WikiText-2 raw official splits are pinned to Salesforce/wikitext revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, subset `wikitext-2-raw-v1`.
The dataset card and prepared file hashes are preserved in
[`dataset_manifest.json`](dataset_manifest.json). This local copy includes
the prepared binary streams and ownership JSONL files under `data/`.

The official training text is concatenated with 12,000 generated ownership
documents; validation/test each add 500 disjoint documents. Generator seed is
314159. People are A–H; objects are x/y/z/u/v/w; each story has one object and
two or three transfers. UTF-8 bytes receive BOS/EOS around physical lines.
Neither official validation nor test data enters the training stream.

There is no guaranteed 50/50 mixture in the original random-window recipe:
Wiki and relational text are concatenated, and sampled windows can cross
document boundaries. The epoch runner shuffles complete contiguous chunks of
that same stream, visiting each prediction target once without changing the
dataset. It does not add two-object stories or fix the training-distribution gap.

## Recreate the data on a source-only clone

Install `.[comparison]` for PyArrow; then run:

```bash
python -c "from pathlib import Path; from cars_r.comparison_data import prepare_dataset; prepare_dataset(Path('data/comparison_wikitext2'))"
```

The epoch command prepares data automatically. Preparation downloads only the
pinned official files, creates disjoint synthetic splits and writes manifests.
An existing manifest is checksum-verified instead of silently replaced.
Binary files, downloaded raw files and caches are Git-ignored; keep metadata
and attribution when distributing data separately.

The local pinned DATASET_CARD.md retains the upstream license/source
information, including Wikipedia attribution. Project MIT licensing does not
replace the dataset's terms. Consult that original card for attribution.

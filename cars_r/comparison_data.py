"""Pinned WikiText-2 plus disjoint, reproducible relational training examples."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import urllib.request

from .data import prepare_byte_corpus

REPO = "Salesforce/wikitext"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SUBSET = "wikitext-2-raw-v1"
PEOPLE = list("ABCDEFGH")
OBJECTS = list("xyzuvw")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CARS-R-research/0.1.3"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            temporary.replace(path)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def relational_case(rng: random.Random, steps: int):
    owner = rng.choice(PEOPLE)
    obj = rng.choice(OBJECTS)
    lines = [f"{owner} owns {obj}."]
    for _ in range(steps):
        following = rng.choice([p for p in PEOPLE if p != owner])
        lines.append(f"{owner} gives {obj} to {following}.")
        owner = following
    return " ".join(lines) + f" Owner of {obj}:", " " + owner


def build_relational_splits(directory: Path, counts=(12000, 500, 500), seed=314159):
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    seen = set()
    metadata = {}
    for split, count in zip(("train", "validation", "test"), counts):
        examples = []
        # 2-3 transfers offer enough unique cases and fit the 128-token window.
        while len(examples) < count:
            prompt, target = relational_case(rng, rng.choice((2, 3)))
            text = prompt + target
            if text in seen:
                continue
            seen.add(text)
            examples.append({"prompt": prompt, "target": target})
        (directory / f"relations_{split}.jsonl").write_text(
            "".join(json.dumps(case) + "\n" for case in examples), encoding="utf-8")
        (directory / f"relations_{split}.txt").write_text(
            "".join(case["prompt"] + case["target"] + "\n" for case in examples), encoding="utf-8")
        metadata[split] = {"documents": count, "sha256": sha256(directory / f"relations_{split}.jsonl")}
    return {"generator": "ownership-transfer-v1", "seed": seed, "splits": metadata, "max_training_transfer_steps": 3}


def prepare_dataset(directory: Path):
    """Only official train + generated train enter the shared training stream."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("revision") != REVISION:
            raise ValueError("dataset cache revision differs; use a new data directory")
        for name, digest in manifest["prepared_sha256"].items():
            if not (directory / name).exists() or sha256(directory / name) != digest:
                raise ValueError(f"prepared dataset checksum mismatch: {name}")
        print("DATA: verified cached WikiText-2 + ownership challenge", flush=True)
        return manifest
    directory.mkdir(parents=True, exist_ok=True)
    relations = build_relational_splits(directory)
    sources = {}
    for split in ("train", "validation", "test"):
        filename = f"{split}-00000-of-00001.parquet"
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{SUBSET}/{filename}?download=true"
        parquet = directory / "raw" / filename
        if not parquet.exists():
            print(f"DATA: downloading official WikiText-2 {split}", flush=True)
            download(url, parquet)
        wiki_path = directory / f"wiki_{split}.txt"
        # Keep Arrow native libraries separate from Torch on Windows. Only
        # verified UTF-8 text crosses into the training process.
        worker = """import json,sys
from pathlib import Path
import pyarrow.parquet as pq
table=pq.ParquetFile(sys.argv[1]).read(columns=['text'],use_threads=False)
texts=[text for text in table['text'].to_pylist() if text and text.strip()]
wiki=''.join(text if text.endswith('\\n') else text+'\\n' for text in texts)
Path(sys.argv[2]).write_text(wiki,encoding='utf-8',newline='')
print(json.dumps({'official_rows':len(table),'nonempty_rows':len(texts)}))
"""
        extracted = subprocess.run([sys.executable, "-c", worker, str(parquet), str(wiki_path)], capture_output=True, text=True, check=True)
        row_info = json.loads(extracted.stdout)
        expected_rows = {"train": 36718, "validation": 3760, "test": 4358}
        if row_info["official_rows"] != expected_rows[split]:
            raise ValueError(f"WikiText-2 {split} row count differs from the pinned dataset card")
        sources[split] = {"url": url, "parquet_sha256": sha256(parquet), **row_info}
        mixed = directory / f"{split}.txt"
        mixed.write_bytes(wiki_path.read_bytes() + (directory / f"relations_{split}.txt").read_bytes())
        prepare_byte_corpus(mixed, directory / f"{split}.bin")
        if split == "test":
            prepare_byte_corpus(wiki_path, directory / "wiki_test.bin")
    card = directory / "DATASET_CARD.md"
    download(f"https://huggingface.co/datasets/{REPO}/raw/{REVISION}/README.md", card)
    prepared = [directory / f"{split}.{extension}" for split in ("train", "validation", "test") for extension in ("bin", "bin.json")]
    prepared += [directory / "wiki_test.bin", directory / "wiki_test.bin.json", card]
    prepared += list(directory.glob("relations_*.jsonl"))
    manifest = {"dataset": REPO, "subset": SUBSET, "revision": REVISION,
        "source_card": f"https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md",
        "license": "See the preserved pinned DATASET_CARD.md; retain attribution for WikiText/Wikipedia data.",
        "tokenizer": "UTF-8 bytes + BOS/EOS; physical line per document",
        "sources": sources, "synthetic": relations,
        "prepared_sha256": {p.relative_to(directory).as_posix(): sha256(p) for p in prepared}}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("DATA: ready; official splits and generated splits remain separate", flush=True)
    return manifest

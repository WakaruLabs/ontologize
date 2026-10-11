"""Backfill the line/sentence marks and the version record onto a sentence
cache whose split predates them (`mc4_sent16`), from a re-split under
another name with the current `encode_sentences.py`.

A re-split replays the same stream with the same seed and models, so it
chooses the same sentences; this checks that it did, field by field for
every entry of every document, before touching the cache. Then it writes
<name>.kind.npy in the cache's row order, replaces <name>.sentences.jsonl
with the marked file (whose first six fields are the old file's), and adds
the kind sidecar and the re-split's versions to <name>.meta.json. The
embeddings are not touched.

    uv run --with wtpsplit --with fasttext==0.9.3 python encode_sentences.py split --name mc4_sent16_marks
    uv run python experiments/sentence-split/backfill_marks.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="mc4_sent16", help="the cache to mark")
    ap.add_argument("--marks", default="mc4_sent16_marks", help="the re-split with marks")
    ap.add_argument("-o", "--out", default="data/sonar_embeddings")
    args = ap.parse_args()

    from encode_sentences import encode_versions, entry

    out = Path(args.out)
    meta_path = out / f"{args.name}.meta.json"
    meta = json.loads(meta_path.read_text())
    old_path = out / f"{args.name}.sentences.jsonl"
    new_path = out / f"{args.marks}.sentences.jsonl"
    if "kind" in meta and (out / f"{args.name}.kind.npy").exists():
        # a run that stopped after the sidecar and the metadata: only the
        # sentences file may be left to swap in
        if new_path.exists():
            os.replace(new_path, old_path)
        print(f"{args.name} is already marked")
        return
    marks_split = json.loads((out / f"{args.marks}.split.json").read_text())
    if not marks_split.get("complete"):
        sys.exit(f"{args.marks}'s split is not complete")
    old_settings = {k: v for k, v in meta["split"].items() if k != "versions"}
    new_settings = {k: v for k, v in marks_split["settings"].items() if k != "versions"}
    if old_settings != new_settings:
        sys.exit(f"the re-split's settings differ: {new_settings} against {old_settings}")
    n, n_docs = meta["n"], meta["n_docs"]
    starts = np.load(out / f"{args.name}.docstart.npy")
    counts = np.diff(np.append(starts, n))

    kind_part = out / f"{args.name}.kind.npy.part"
    K = np.lib.format.open_memmap(kind_part, mode="w+", dtype="U8", shape=(n,))
    rows = 0
    with open(old_path, encoding="utf-8") as fo, open(new_path, encoding="utf-8") as fn:
        for doc, (lo, ln) in enumerate(zip(fo, fn)):
            a, b = json.loads(lo), json.loads(ln)
            if (a["doc"], a["lang"]) != (b["doc"], b["lang"]) or a["doc"] != doc:
                sys.exit(f"line {doc}: documents differ ({a['doc']} {a['lang']}, {b['doc']} {b['lang']})")
            if [e[:6] for e in b["sents"]] != a["sents"]:
                sys.exit(f"document {doc}: the re-split chose different sentences")
            if len(a["sents"]) != counts[doc]:
                sys.exit(f"document {doc}: {len(a['sents'])} sentences, the cache has {counts[doc]} rows")
            for e in b["sents"]:
                K[rows] = entry(e)[6]
                rows += 1
            if doc % 500000 == 0:
                print(f"  {doc}/{n_docs} documents checked", flush=True)
    if (doc + 1, rows) != (n_docs, n):
        sys.exit(f"checked {doc + 1} documents and {rows} rows; the cache has {n_docs} and {n}")
    K.flush()
    kinds = {k: int(v) for k, v in zip(*np.unique(K, return_counts=True))}
    del K

    os.replace(kind_part, out / f"{args.name}.kind.npy")
    meta["split"]["versions"] = marks_split["settings"]["versions"]
    meta["split_versions_note"] = (f"recorded from the re-split {args.marks}, which chose the same "
                                   "sentences; the original split did not record versions")
    meta.setdefault("encode_versions", encode_versions())
    meta["kind"] = (f"{args.name}.kind.npy: sentence, line (a whole line without sentence-final "
                    "punctuation) or piece (made by the cap)")
    meta["kind_counts"] = kinds
    meta_path.write_text(json.dumps(meta, indent=2))
    os.replace(new_path, old_path)   # last: a rerun after this point finds it done
    print(f"✓ {args.name}: {rows} rows marked {kinds}; {old_path.name} now carries the marks")


if __name__ == "__main__":
    main()

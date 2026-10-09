"""Encode a genuinely held-out mC4 slice into a fresh embedding cache.

The Ontologizer runs trained on the FULL mc4_4M cache, so the cache tail
that sae.py holds out is held out only for the SAEs (pareto.py's caveat).
This harness closes that asymmetry: it replays encode_corpus.py's exact
deterministic stream (same interleave order, tokenizer, encoder dtype,
pooling, and the training cache's source-language tags, read from its
meta.json), SKIPS the first --skip samples (default 4,000,000 = everything
mc4_4M contains, trained on by every model), and encodes the next --n
samples into a fresh <name>.npy + <name>.langs.npy pair that no model has
ever seen. Downstream scripts accept it anywhere they take --cache.

While skipping, each stream item's language is checked against the
training cache's .langs.npy sidecar (the same drift guard autointerp.py's
texts stage uses): if the stream no longer matches what was encoded, the
"fresh rows are exactly stream rows skip..skip+n" claim is void and the
run aborts instead of writing a silently misaligned cache.

Crash-safe and resumable exactly like encode_corpus.py: .part memmaps, a
progress sidecar, rename-on-complete. Rerunning resumes (the skip is
replayed; cheap relative to encoding).

  uv run python experiments/fresh-eval/encode_fresh.py
  uv run python experiments/fresh-eval/encode_fresh.py --n 65536

Writes <out>/<name>.npy, <name>.langs.npy, <name>.meta.json.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))

import argparse
import json
import numpy as np

# reuse encode_corpus's machinery/constants so fresh embeddings are
# computed bit-identically to the training cache (its import also
# preloads CuDNN and picks the torch device)
import encode_corpus
from ontologize.data.langs import MC4_4M_TAGS, MC4_TO_SONAR
from ontologize.data.loaders import HFDataSource, TokenizeTransform
from ontologize.data.multilingual import mc4_data
from ontologize.data.pretrained import pretrained_transformer, encode


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip", type=int, default=4_000_000,
                   help="stream samples to skip: the size of the training "
                        "cache (mc4_4M -> 4,000,000)")
    p.add_argument("--n", type=int, default=131072,
                   help="fresh samples to encode")
    p.add_argument("--b", type=int, default=256,
                   help="encoding batch size (encode_corpus.py default)")
    p.add_argument("--name", default="mc4_fresh")
    p.add_argument("--out", default="experiments/fresh-eval/out",
                   help="output directory for the fresh cache")
    p.add_argument("--train-cache",
                   default="data/sonar_embeddings/mc4_4M.npy",
                   help="training cache whose .langs.npy verifies the "
                        "skipped stream ('' disables the drift check)")
    p.add_argument("--flush-each", type=int, default=50,
                   help="batches between memmap flush + progress checkpoint")
    return p.parse_args()


def main():
    from tqdm import tqdm
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    d = encode_corpus.d
    final = outdir / f"{args.name}.npy"
    part = outdir / f"{args.name}.npy.part"
    langs_final = outdir / f"{args.name}.langs.npy"
    langs_part = outdir / f"{args.name}.langs.npy.part"
    prog = outdir / f"{args.name}.progress.json"

    if final.exists():
        print(f"error: {final} already exists. exiting.")
        return

    # encode under the training cache's own source tags, so the fresh rows
    # come from the distribution the models trained on; a meta.json with
    # no src_tags entry marks a cache built with langs.MC4_4M_TAGS
    tags_name = "MC4_TO_SONAR"
    if args.train_cache:
        train_meta = Path(args.train_cache).with_name(
            Path(args.train_cache).name.replace(".npy", ".meta.json"))
        if train_meta.exists():
            tags_name = json.loads(train_meta.read_text()).get(
                "src_tags", "MC4_4M_TAGS")
    tags = {"MC4_TO_SONAR": MC4_TO_SONAR, "MC4_4M_TAGS": MC4_4M_TAGS}[tags_name]
    print(f"source tags: {tags_name}")

    done = 0
    if prog.exists() and part.exists():
        meta = json.loads((outdir / f"{args.name}.meta.json").read_text())
        assert meta.get("src_tags") == tags_name, \
            f"resume tag mismatch: {part} was encoded under " \
            f"{meta.get('src_tags')}, the training cache uses {tags_name}"
        j = json.loads(prog.read_text())
        assert j.get("skip") == args.skip, \
            f"resume skip mismatch: progress has {j.get('skip')}, " \
            f"CLI has {args.skip}"
        done = j["count"]
        E = np.lib.format.open_memmap(part, mode="r+")
        L = np.lib.format.open_memmap(langs_part, mode="r+")
        assert E.shape == (args.n, d), \
            f"{part} has shape {E.shape}, expected {(args.n, d)}"
        print(f"Resuming {args.name} from sample {done}/{args.n}")
    else:
        E = np.lib.format.open_memmap(
            part, mode="w+", dtype=np.float32, shape=(args.n, d))
        L = np.lib.format.open_memmap(
            langs_part, mode="w+", dtype="U8", shape=(args.n,))
        (outdir / f"{args.name}.meta.json").write_text(json.dumps({
            "n": args.n, "d": d, "dtype": "float32",
            "skip": args.skip,
            "encoder": encode_corpus.encoder_id,
            "encoder_dtype": encode_corpus.dtype_str,
            "maxlen": encode_corpus.maxlen,
            "pooling": "masked mean + l2 norm",
            "source": "allenai/c4 train, interleaved over MC4_TO_SONAR; "
                      f"stream rows {args.skip}..{args.skip + args.n} "
                      "(everything before --skip is training data)",
            "src_tags": tags_name,
        }, indent=2))

    # training-cache sidecar for the stream-drift guard during the skip
    L_train = None
    if args.train_cache:
        langs_path = Path(args.train_cache).with_name(
            Path(args.train_cache).name.replace(".npy", ".langs.npy"))
        if langs_path.exists():
            L_train = np.load(langs_path, mmap_mode="r")
        else:
            print(f"WARNING: {langs_path} missing; stream alignment "
                  "of the skipped prefix is unverified")

    dev = encode_corpus.dev
    model, tokenizer = pretrained_transformer(
        encode_corpus.encoder_id, encode_corpus.dtype_str, dev=dev)
    model.eval()
    tok = TokenizeTransform(tokenizer, maxlen=encode_corpus.maxlen, tags=tags)

    ds = mc4_data("allenai/c4", split="train", streaming=True)
    it = iter(HFDataSource(ds, text_key="text"))
    for row in tqdm(range(args.skip + done), desc="Skipping", unit="sample"):
        item = next(it)
        if L_train is not None and row < len(L_train) \
                and item["lang"] != L_train[row]:
            raise RuntimeError(
                f"stream drift at row {row}: stream lang {item['lang']!r}"
                f" != cached {L_train[row]!r}; the fresh slice would not be"
                " the samples after the training corpus -- match the"
                " dataset revision or re-encode")

    def checkpoint(n):
        E.flush()
        L.flush()
        prog.write_text(json.dumps({"skip": args.skip, "count": n}))

    n = done
    buf, langs = [], []
    with tqdm(total=args.n, initial=done, desc="Encoding",
              unit="sample") as pbar:
        while n < args.n:
            try:
                item = next(it)
            except StopIteration:
                break
            buf.append(tok.map(item))
            langs.append(item["lang"])
            if len(buf) == min(args.b, args.n - n):
                batch = {k: np.stack([s[k] for s in buf])
                         for k in ("input_ids", "attention_mask")}
                E[n:n + len(buf)] = np.asarray(encode(model, batch, dev))
                L[n:n + len(buf)] = langs
                n += len(buf)
                pbar.update(len(buf))
                buf, langs = [], []
                if (n // args.b) % args.flush_each == 0 or n >= args.n:
                    checkpoint(n)

    checkpoint(n)
    if n < args.n:
        # interleave exhausted (smallest language ran out past --skip):
        # shrink to what we actually got
        print(f"Stream exhausted at {n}/{args.n}; truncating cache")
        encode_corpus.truncate_npy(part, n)
        encode_corpus.truncate_npy(langs_part, n)

    os.replace(part, final)
    os.replace(langs_part, langs_final)
    prog.write_text(json.dumps({"skip": args.skip, "count": n,
                                "complete": True}))
    print(f"✓ {final}: {n} x {d} float32 fresh embeddings "
          f"(stream rows {args.skip}..{args.skip + n})")


if __name__ == "__main__":
    main()

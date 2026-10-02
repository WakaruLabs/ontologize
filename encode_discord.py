# Precompute SONAR embeddings for the `content` field of every message in a
# DiscordChatExporter JSON export (e.g. ../act-i-export/*.json), storing them
# as a .npy cache that the `srctype="embedding"` loader can read directly.
#
# Uses the same encoder, dtype, pooling, and max length as encode_corpus.py,
# so these embeddings live in the same space as the mC4 cache and can be fed
# to a model trained on it. Two differences, both deliberate:
#   * dynamic padding instead of pad-to-512. Padded positions are masked out
#     of the attention and of `l2_pooling`, so the pooled vector is
#     numerically equivalent (not bitwise -- different padded shapes pick
#     different kernels) while being several times faster on short messages.
#   * within each chunk, messages are encoded in length order and scattered
#     back to their original rows, so a batch is not padded to its single
#     longest member. Row order in the output is unaffected.
#
# Row i of <name>.npy corresponds to line i of <name>.jsonl, which carries the
# channel, message id, author, timestamp, and the exact text encoded. Messages
# with empty/whitespace-only content are skipped and absent from both files.
#
# Crash-safe and resumable at chunk granularity: embeddings are written into
# <name>.npy.part with a progress sidecar and renamed to <name>.npy only when
# complete, so a half-built cache can never be mistaken for a finished one.

# disable preallocation so jax and torch can share VRAM
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import ctypes
from pathlib import Path

# Force load the pip-installed CuDNN library before JAX initializes
try:
    import nvidia.cudnn
    cudnn_dir = Path(list(nvidia.cudnn.__path__)[0])
    cudnn_path = cudnn_dir / "lib" / "libcudnn.so"
    if cudnn_path.exists():
        ctypes.CDLL(str(cudnn_path), mode=os.RTLD_GLOBAL)
except (ImportError, IndexError):
    pass

import argparse
import glob
import json
import numpy as np
import torch as t

from tqdm import tqdm

from ontologize.data.pretrained import pretrained_transformer, encode

dev = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

src = "../act-i-export"
out = Path("data").resolve() / "sonar_embeddings"
name = "act_i"

encoder_id = "cointegrated/SONAR_200_text_encoder"
dtype_str = "float32" # must match sonar.py: reducing precision breaks the embeddings
d = 1024              # SONAR embedding dimension
maxlen = 512          # sequence length
lang = "eng_Latn"     # SONAR source-language token

b = 64          # encoding batch size (post-bucketing)
chunk = 32768   # rows per length-bucketing window / checkpoint interval


def build_index(srcdir: Path, path: Path) -> int:
    """One row per non-empty message, in sorted-filename then message order.
    Written once; the row order it fixes is what the .npy is aligned to."""
    n = 0
    with open(path, "w") as fh:
        for f in sorted(glob.glob(str(srcdir / "*.json"))):
            doc = json.load(open(f))
            ch = doc["channel"]
            for m in doc["messages"]:
                text = (m.get("content") or "").strip()
                if not text:
                    continue
                fh.write(json.dumps({
                    "channel": ch.get("name"),
                    "category": ch.get("category"),
                    "channel_id": ch.get("id"),
                    "id": m["id"],
                    "author": (m.get("author") or {}).get("name"),
                    "is_bot": (m.get("author") or {}).get("isBot"),
                    "timestamp": m.get("timestamp"),
                    "content": text,
                }, ensure_ascii=False) + "\n")
                n += 1
    return n


def read_texts(path: Path):
    with open(path) as fh:
        return [json.loads(l)["content"] for l in fh]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-s", "--src", default=src, help="export directory")
    ap.add_argument("-o", "--out", default=str(out), help="output directory")
    ap.add_argument("-m", "--name", default=name, help="cache name")
    ap.add_argument("-b", "--b", type=int, default=b, help="batch size")
    ap.add_argument("-c", "--chunk", type=int, default=chunk,
                    help="rows per bucketing window / checkpoint")
    ap.add_argument("-l", "--lang", default=lang, help="SONAR source language")
    ap.add_argument("--maxlen", type=int, default=maxlen)
    args = ap.parse_args()

    srcdir = Path(args.src).resolve()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    final = outdir / f"{args.name}.npy"
    part = outdir / f"{args.name}.npy.part"
    index = outdir / f"{args.name}.jsonl"
    prog = outdir / f"{args.name}.progress.json"

    if final.exists():
        print(f"error: {final} already exists. exiting.")
        return

    if index.exists():
        print(f"Reusing index {index}")
    else:
        print(f"Indexing {srcdir} ...")
        n_idx = build_index(srcdir, index)
        print(f"  {n_idx:,} non-empty messages")

    texts = read_texts(index)
    n = len(texts)

    done = 0
    if prog.exists() and part.exists():
        done = json.loads(prog.read_text())["count"]
        E = np.lib.format.open_memmap(part, mode="r+")
        assert E.shape == (n, d), f"{part} has shape {E.shape}, expected {(n, d)}"
        print(f"Resuming {args.name} from row {done:,}/{n:,}")
    else:
        E = np.lib.format.open_memmap(
                part, mode="w+", dtype=np.float32, shape=(n, d))
        (outdir / f"{args.name}.meta.json").write_text(json.dumps({
            "n": n, "d": d, "dtype": "float32",
            "encoder": encoder_id, "encoder_dtype": dtype_str,
            "maxlen": args.maxlen, "pooling": "masked mean + l2 norm",
            "padding": "dynamic (length-bucketed within chunk)",
            "src_lang": args.lang,
            "source": f"DiscordChatExporter JSON: {srcdir}",
            "field": "messages[].content (stripped; empty skipped)",
            "index": index.name,
            }, indent=2))

    model, tokenizer = pretrained_transformer(encoder_id, dtype_str, dev=dev)
    model.eval()
    tokenizer.src_lang = args.lang

    with tqdm(total=n, initial=done, desc="Encoding", unit="msg") as pbar:
        for c0 in range(done, n, args.chunk):
            c1 = min(c0 + args.chunk, n)
            # encode in length order, scatter back to original rows
            rows = sorted(range(c0, c1), key=lambda i: len(texts[i]))
            for j in range(0, len(rows), args.b):
                idx = rows[j:j + args.b]
                tok = tokenizer([texts[i] for i in idx], return_tensors="np",
                                padding=True, truncation=True,
                                max_length=args.maxlen)
                batch = {"input_ids": tok["input_ids"],
                         "attention_mask": tok["attention_mask"]}
                E[idx] = np.asarray(encode(model, batch, dev))
                pbar.update(len(idx))
            E.flush()
            prog.write_text(json.dumps({"count": c1}))

    E.flush()
    del E
    os.replace(part, final)
    prog.write_text(json.dumps({"count": n, "complete": True}))
    print(f"✓ {final}: {n} x {d} float32 embeddings")
    print(f"✓ {index}: row-aligned message metadata")


if __name__ == "__main__":
    main()

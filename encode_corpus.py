# Precompute SONAR embeddings for the interleaved mC4 corpus and store them
# as a .npy cache that sonar.py can train from directly (see `cache` there).
# Uses the exact same stream order, tokenization, encoder dtype, and pooling
# as sonar.py's on-the-fly path, so cached embeddings are bit-identical to
# what training would have computed.
#
# Crash-safe and resumable: embeddings are written into <name>.npy.part with
# a progress sidecar, flushed every `flush_each` batches, and renamed to
# <name>.npy only when complete -- so a half-built cache can never be
# mistaken for a finished one. Rerunning after a crash/reboot skips the
# already-encoded samples (cheap: the stream is re-read but not re-encoded).

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
   # Namespace packages use __path__ which is an iterable of directory locations
   cudnn_dir = Path(list(nvidia.cudnn.__path__)[0])
   cudnn_path = cudnn_dir / "lib" / "libcudnn.so"

   if cudnn_path.exists():
       ctypes.CDLL(str(cudnn_path), mode=os.RTLD_GLOBAL)
except (ImportError, IndexError):
   pass

import argparse
import json
import numpy as np
import torch as t

from tqdm import tqdm

from ontologize.data.loaders import HFDataSource, TokenizeTransform
from ontologize.data.multilingual import mc4_data
from ontologize.data.pretrained import pretrained_transformer, encode

dev = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

path = Path("data").resolve()
dtype_str = "float32" # must match sonar.py: reducing precision breaks the embeddings

# NLLB tokenizer and model
encoder_id = "cointegrated/SONAR_200_text_encoder"

d = 1024 # SONAR embedding dimension
maxlen = 512 # sequence length

N = 4_000_000 # corpus size; at b=256 this is 15625 steps/epoch
b = 256 # encoding batch size
flush_each = 50 # batches between memmap flush + progress checkpoint

out = path / "sonar_embeddings"
name = "mc4_4M"


def truncate_npy(p: Path, n: int, bufrows: int = 65536):
    """Rewrite the .npy at `p` with only its first `n` rows (chunked)."""
    src = np.load(p, mmap_mode="r")
    tmp = p.with_name(p.name + ".trunc")
    dst = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=src.dtype, shape=(n,) + src.shape[1:])
    for i in range(0, n, bufrows):
        j = min(i + bufrows, n)
        dst[i:j] = src[i:j]
    dst.flush()
    del src, dst
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n", type=int, default=N, help='Corpus size')
    ap.add_argument("-b", "--b", type=int, default=b, help='Batch size')
    ap.add_argument("-m","--name", default=name, help='Name of model')
    ap.add_argument("-o", "--out", default=str(out), help='Output directory')
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    final = outdir / f"{args.name}.npy"
    part = outdir / f"{args.name}.npy.part"
    langs_final = outdir / f"{args.name}.langs.npy"
    langs_part = outdir / f"{args.name}.langs.npy.part"
    prog = outdir / f"{args.name}.progress.json"

    if final.exists():
        # it is possible it isn't complete if you are generating with different cli paramaters
        print(f"error: {final} already exists. exiting.")
        return

    done = 0
    if prog.exists() and part.exists():
        meta = json.loads((outdir / f"{args.name}.meta.json").read_text())
        if meta.get("src_tags") != "MC4_TO_SONAR":
            # caches without the field were tagged by langs.MC4_4M_TAGS
            print(f"error: {part} was encoded under different source "
                  "tags (its meta.json has no src_tags: MC4_TO_SONAR), and "
                  "resuming would mix two tag maps in one cache. Delete "
                  "the .part, .progress.json and .meta.json files to "
                  "start over.")
            return
        done = json.loads(prog.read_text())["count"]
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
            "encoder": encoder_id, "encoder_dtype": dtype_str,
            "maxlen": maxlen, "pooling": "masked mean + l2 norm",
            "source": "allenai/c4 train, interleaved over MC4_TO_SONAR",
            "src_tags": "MC4_TO_SONAR",
            }, indent=2))

    model, tokenizer = pretrained_transformer(encoder_id, dtype_str, dev=dev)
    model.eval()
    tok = TokenizeTransform(tokenizer, maxlen=maxlen)

    ds = mc4_data("allenai/c4", split="train", streaming=True)
    it = iter(HFDataSource(ds, text_key="text"))
    for _ in tqdm(range(done), desc="Skipping encoded samples"):
        next(it)

    def checkpoint(n):
        E.flush()
        L.flush()
        prog.write_text(json.dumps({"count": n}))

    n = done
    buf, langs = [], []
    with tqdm(total=args.n, initial=done, desc="Encoding", unit="sample") as pbar:
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
                if (n // args.b) % flush_each == 0 or n >= args.n:
                    checkpoint(n)

    checkpoint(n)
    if n < args.n:
        # corpus exhausted early (interleave stops when the smallest
        # language runs out): shrink the arrays to what we actually got
        print(f"Stream exhausted at {n}/{args.n}; truncating cache")
        truncate_npy(part, n)
        truncate_npy(langs_part, n)

    os.replace(part, final)
    os.replace(langs_part, langs_final)
    prog.write_text(json.dumps({"count": n, "complete": True}))
    print(f"✓ {final}: {n} x {d} float32 embeddings")


if __name__ == "__main__":
    main()

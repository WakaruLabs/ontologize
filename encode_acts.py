"""Harvest transformer residual-stream activations into a `.npy` cache with
the same layout `encode_corpus.py` writes, so `sonar.py`/`train_ste.py`
(`srctype="embedding"`) and `sae.py` train on it unchanged via `--cache`.

The point is a model organism where sparse dictionary learning is known to
work, next to SONAR's dense sentence embeddings. Defaults to GPT-2 small
`resid_post` at layer 8, the most-replicated open SAE site: public SAEs and
public auto-interp explanations exist for it, so `autointerp.py`'s scorer
and `splitting.py`'s matched fractions can be calibrated against something
external rather than only against each other.

One row per token position. Rows are written in DOCUMENT ORDER and that is
load-bearing: `NpyDataSource(holdout=)` and `sae.py --eval-rows` both split
off a contiguous TAIL of rows, so with a shuffled cache the eval split
would hold other positions from documents whose remaining tokens were
trained on, and every reconstruction number would be optimistic. In
document order the tail is whole documents. Batch composition needs no help
from the on-disk order -- both trainers permute globally -- so ordering
costs nothing. `<name>.docstart.npy` records where each document begins;
`--holdout` reports the nearest boundary to snap to.

Position 0 is a prepended BOS and is never harvested: with nothing to
attend to, its residual is a norm outlier that would dominate the
whitening. Positions 1..L are the real tokens, so row `(doc, pos)` is the
stream after reading `tokens[pos]`.

Sidecars, all keyed by cache row:

  <name>.index.npy     (n, 3) int32 (doc, pos, token_id)
  <name>.docstart.npy  (n_docs,) int64 first row of each document
  <name>.docs.jsonl    one line per document: {doc, row0, text, tokens}
  <name>.meta.json     model, site, ctx, counts, holdout boundaries

Storing the documents outright replaces `autointerp.py texts`'s stream
replay: recovering the text for a row is a lookup, not 4M documents of C4
re-streamed behind a drift check that can fail.

Crash-safe and resumable on the same pattern as `encode_corpus.py`: arrays
grow in `<name>.npy.part` with a progress sidecar, renamed to `<name>.npy`
only when complete, so a half-built cache cannot be mistaken for a
finished one.

  uv run python encode_acts.py                        # gpt2 layer 8, 4M rows
  uv run python encode_acts.py --model EleutherAI/pythia-160m --layer 6
  uv run python encode_acts.py -n 200000 --name gpt2_l8_small
"""
# disable preallocation so jax and torch can share VRAM (as encode_corpus.py)
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, List, Tuple

import numpy as np
import torch as t
from jaxtyping import Float, Int
from tqdm import tqdm

from ontologize.data.loaders import doc_boundaries

dev = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

MODEL_ID = "gpt2"
LAYER = 8          # of 12; resid_post, the mid-stack site open SAEs target
CTX = 128          # tokens kept per document, BOS included
MIN_TOKENS = 32    # skip documents shorter than this (padding waste)
N = 4_000_000      # target rows
B = 64             # documents per forward pass
FLUSH_EACH = 50    # batches between memmap flush + progress checkpoint

OUT = Path("data").resolve() / "activations"


def blocks(model: t.nn.Module) -> t.nn.ModuleList:
    """The transformer's block list, across HF naming conventions.

    Hooking a block's output is preferred over `output_hidden_states`
    because the hidden-states tuple applies the final layer norm to its
    last element and not the others, so an index into it means different
    things depending on which layer is asked for.
    """
    for path in ("h", "layers", "transformer.h", "gpt_neox.layers",
                 "model.layers", "decoder.layers"):
        node: Any = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if isinstance(node, t.nn.ModuleList):
            return node
    raise ValueError(
        f"cannot find the block list on {type(model).__name__}; add its "
        f"attribute path to `blocks()`")


def documents(dataset: str, config: str, split: str, text_key: str
              ) -> Iterator[str]:
    from datasets import load_dataset
    kwargs = dict(split=split, streaming=True)
    ds = (load_dataset(dataset, config, **kwargs) if config
          else load_dataset(dataset, **kwargs))
    for rec in ds:
        text = rec[text_key]
        if text and text.strip():
            yield text


def truncate_npy(p: Path, n: int, bufrows: int = 65536) -> None:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default=MODEL_ID, help="HF model id")
    p.add_argument("--layer", type=int, default=LAYER,
                   help="block index whose OUTPUT is harvested (resid_post)")
    p.add_argument("--model-dtype", default="float32",
                   choices=("float32", "float16", "bfloat16"))
    p.add_argument("--dtype", default="float32", choices=("float32", "float16"),
                   help="cache dtype. float16 halves the file; residual "
                        "streams carry high-variance outlier dimensions, so "
                        "the default keeps full precision")
    p.add_argument("--ctx", type=int, default=CTX,
                   help="tokens per document including the prepended BOS, so "
                        "at most ctx-1 rows come from one document")
    p.add_argument("--min-tokens", type=int, default=MIN_TOKENS,
                   help="skip shorter documents; they are mostly padding")
    p.add_argument("-n", "--n", type=int, default=N, help="target rows")
    p.add_argument("-b", "--b", type=int, default=B,
                   help="documents per forward pass")
    p.add_argument("--dataset", default="allenai/c4")
    p.add_argument("--config", default="en", help="dataset config ('' for none)")
    p.add_argument("--split", default="train")
    p.add_argument("--text-key", default="text")
    p.add_argument("--holdout", type=int, default=32768,
                   help="report the document boundaries nearest a tail of "
                        "this many rows, for sonar.py's `holdout` and "
                        "sae.py's --eval-rows")
    p.add_argument("--flush-each", type=int, default=FLUSH_EACH)
    p.add_argument("-o", "--out", default=str(OUT))
    p.add_argument("--name", default=None,
                   help="cache basename (default <model>_l<layer>)")
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    from transformers import AutoModel, AutoTokenizer

    name = cfg.name or f"{cfg.model.split('/')[-1].replace('-', '_')}_l{cfg.layer}"
    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)
    final = outdir / f"{name}.npy"
    part = outdir / f"{name}.npy.part"
    idx_final = outdir / f"{name}.index.npy"
    idx_part = outdir / f"{name}.index.npy.part"
    docs_path = outdir / f"{name}.docs.jsonl"
    start_final = outdir / f"{name}.docstart.npy"
    start_part = outdir / f"{name}.docstart.npy.part"
    prog = outdir / f"{name}.progress.json"

    if final.exists():
        print(f"error: {final} already exists. exiting.")
        return

    tok = AutoTokenizer.from_pretrained(cfg.model)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    assert bos is not None, \
        f"{cfg.model} has neither bos_token_id nor eos_token_id; position 0 " \
        f"cannot be made a discardable prefix"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token       # GPT-2 ships without one
    # rows are addressed by absolute position, so a left-padded batch would
    # silently shift every index in the sidecar
    tok.padding_side = "right"

    model = AutoModel.from_pretrained(
        cfg.model, torch_dtype=getattr(t, cfg.model_dtype)).to(dev)
    model.eval()
    layers = blocks(model)
    assert 0 <= cfg.layer < len(layers), \
        f"--layer {cfg.layer} out of range for {len(layers)} blocks"
    d = int(model.config.hidden_size)

    # at most ctx-1 rows per document, at least min_tokens, so this bounds
    # the document count without knowing the length distribution
    max_docs = cfg.n // max(cfg.min_tokens, 1) + 1

    done_rows, done_docs, jsonl_bytes = 0, 0, 0
    if prog.exists() and part.exists():
        st = json.loads(prog.read_text())
        done_rows, done_docs = st["rows"], st["docs"]
        jsonl_bytes = st["jsonl_bytes"]
        E = np.lib.format.open_memmap(part, mode="r+")
        I = np.lib.format.open_memmap(idx_part, mode="r+")
        S = np.lib.format.open_memmap(start_part, mode="r+")
        assert E.shape == (cfg.n, d), \
            f"{part} has shape {E.shape}, expected {(cfg.n, d)}"
        # a crash can leave a partial line; the byte count is what was
        # flushed alongside the arrays, so cut back to it
        with open(docs_path, "r+b") as f:
            f.truncate(jsonl_bytes)
        print(f"Resuming {name} from row {done_rows}/{cfg.n} "
              f"({done_docs} documents)")
    else:
        E = np.lib.format.open_memmap(
            part, mode="w+", dtype=np.dtype(cfg.dtype), shape=(cfg.n, d))
        I = np.lib.format.open_memmap(
            idx_part, mode="w+", dtype=np.int32, shape=(cfg.n, 3))
        S = np.lib.format.open_memmap(
            start_part, mode="w+", dtype=np.int64, shape=(max_docs,))
        docs_path.write_text("")

    captured: List[t.Tensor] = []

    def hook(_mod, _inp, out):
        # blocks return (hidden_states, present, ...) in most HF decoders
        captured.append(out[0] if isinstance(out, tuple) else out)

    handle = layers[cfg.layer].register_forward_hook(hook)

    it = documents(cfg.dataset, cfg.config, cfg.split, cfg.text_key)
    for _ in tqdm(range(done_docs), desc="Skipping harvested documents"):
        next(it)

    def checkpoint(rows: int, docs: int, nbytes: int) -> None:
        E.flush()
        I.flush()
        S.flush()
        prog.write_text(json.dumps(
            {"rows": rows, "docs": docs, "jsonl_bytes": nbytes}))

    rows, docs = done_rows, done_docs
    batches = 0
    stream_done = False
    jf = open(docs_path, "a")
    with tqdm(total=cfg.n, initial=rows, desc="Harvesting", unit="row") as pbar:
        while rows < cfg.n and not stream_done:
            texts: List[str] = []
            while len(texts) < cfg.b:
                try:
                    texts.append(next(it))
                except StopIteration:
                    stream_done = True
                    break
            if not texts:
                break

            enc = tok(texts, truncation=True, max_length=cfg.ctx - 1,
                      padding=True, return_tensors="pt")
            ids = enc["input_ids"]
            mask = enc["attention_mask"]
            lens = mask.sum(1)
            keep = (lens >= cfg.min_tokens).nonzero(as_tuple=True)[0]
            if not len(keep):
                continue
            ids, mask, lens = ids[keep], mask[keep], lens[keep]
            # prepend BOS: position 0 is discarded, so every kept row is a
            # token that had a prefix to attend to
            ids = t.cat([t.full((len(ids), 1), bos, dtype=ids.dtype), ids], 1)
            mask = t.cat([t.ones((len(mask), 1), dtype=mask.dtype), mask], 1)

            captured.clear()
            with t.no_grad():
                model(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                      use_cache=False)
            H = captured[0].to(t.float32).cpu().numpy()

            for j, doc_len in enumerate(lens.tolist()):
                if rows >= cfg.n:
                    break
                take = min(doc_len, cfg.n - rows)
                # `take`, not `doc_len`: the row budget can run out inside a
                # document, and listing tokens that produced no row would
                # break `len(tokens) - 1 == rows in this document` for the
                # last one. No row refers past `take`, so nothing is lost.
                seq = ids[j, :take + 1].numpy()
                S[docs] = rows
                E[rows:rows + take] = H[j, 1:1 + take].astype(cfg.dtype)
                I[rows:rows + take, 0] = docs
                I[rows:rows + take, 1] = np.arange(1, take + 1, dtype=np.int32)
                I[rows:rows + take, 2] = seq[1:1 + take]
                jf.write(json.dumps(
                    {"doc": docs, "row0": rows, "text": texts[int(keep[j])],
                     "tokens": seq.tolist()}, ensure_ascii=False) + "\n")
                rows += take
                docs += 1
                pbar.update(take)

            batches += 1
            if batches % cfg.flush_each == 0 or rows >= cfg.n:
                jf.flush()
                os.fsync(jf.fileno())
                checkpoint(rows, docs, jf.tell())

    handle.remove()
    jf.flush()
    os.fsync(jf.fileno())
    nbytes = jf.tell()
    checkpoint(rows, docs, nbytes)
    jf.close()

    if rows < cfg.n:
        print(f"Stream exhausted at {rows}/{cfg.n}; truncating cache")
        truncate_npy(part, rows)
        truncate_npy(idx_part, rows)
    truncate_npy(start_part, docs)

    # per-dim inverse-variance MSE weights, mean-1 normalized: the same
    # target whitening sonar.py and sae.py take as --mse-weights, whose
    # SONAR file is the wrong width for this cache. Computed over rows
    # strictly before the earliest document boundary the requested holdout
    # could snap to, so no eval-split statistic reaches the objective.
    S = np.load(start_part, mmap_mode="r")
    _, wide = doc_boundaries(S, rows, cfg.holdout)
    train_rows = rows - wide
    A = np.load(part, mmap_mode="r")
    var = np.zeros(d, np.float64)
    mean = np.zeros(d, np.float64)
    seen = 0
    for i in range(0, train_rows, 1 << 16):
        chunk = np.asarray(A[i:min(i + (1 << 16), train_rows)], np.float64)
        # Welford by block, so a 4M-row cache never lands in memory at once
        n_c = len(chunk)
        mu_c, var_c = chunk.mean(0), chunk.var(0)
        delta = mu_c - mean
        var = ((seen * var + n_c * var_c
                + delta ** 2 * seen * n_c / (seen + n_c)) / (seen + n_c))
        mean += delta * n_c / (seen + n_c)
        seen += n_c
    w = 1.0 / np.maximum(var, np.finfo(np.float64).tiny)
    w /= w.mean()
    np.save(outdir / f"{name}.mse_weights.npy", w.astype(np.float32))
    del A, S

    (outdir / f"{name}.meta.json").write_text(json.dumps({
        "n": rows, "d": d, "dtype": cfg.dtype,
        "model": cfg.model, "model_dtype": cfg.model_dtype,
        "site": f"blocks.{cfg.layer}.resid_post",
        "n_blocks": len(layers), "ctx": cfg.ctx, "min_tokens": cfg.min_tokens,
        "bos_prepended": True, "position_0_dropped": True,
        "n_docs": docs, "order": "document",
        "source": f"{cfg.dataset}"
                  f"{'/' + cfg.config if cfg.config else ''} {cfg.split}",
        "index": f"{name}.index.npy", "docs": f"{name}.docs.jsonl",
        "docstart": f"{name}.docstart.npy",
        "mse_weights": f"{name}.mse_weights.npy",
        "mse_weights_train_rows": train_rows,
    }, indent=2))

    os.replace(part, final)
    os.replace(idx_part, idx_final)
    os.replace(start_part, start_final)
    prog.write_text(json.dumps(
        {"rows": rows, "docs": docs, "jsonl_bytes": nbytes, "complete": True}))

    lo, hi = doc_boundaries(np.load(start_final), rows, cfg.holdout)
    print(f"✓ {final}: {rows} x {d} {cfg.dtype}, {docs} documents, "
          f"{rows / docs:.1f} rows/doc")
    print(f"  site {cfg.model} blocks.{cfg.layer}.resid_post of "
          f"{len(layers)} blocks")
    print(f"  mse_weights over {train_rows} train rows: "
          f"{w.min():.3g} to {w.max():.3g}, spread {w.max() / w.min():.3g}x")
    if cfg.holdout in (lo, hi):
        print(f"  holdout {cfg.holdout} lands on a document boundary")
    else:
        print(f"  holdout {cfg.holdout} splits a document across train and "
              f"eval. Whole-document tails nearest it are {lo} and {hi} rows; "
              f"use one for sonar.py's `holdout` and sae.py's --eval-rows")


if __name__ == "__main__":
    main()

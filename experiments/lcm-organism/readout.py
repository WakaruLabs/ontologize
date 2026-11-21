"""An LCM state's linear readout from its row and its context's mean: the control for context averaging.

    JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/readout.py
    JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/readout.py --features row

Gate B (stateinput.py's `x_context`) found that a ridge from x_n, the mean of
the document's earlier rows and an indicator of n = 0 explains about 70% of
block 4's state. A dictionary's seeds may agree on the LCM's states for what
that simple mix carries, rather than for anything else the model computes.
This writes the ridge's prediction of the target harvest's state at every one
of its rows,

    y_n = W^T [x_n, mean(x_0 .. x_{n-1}), 1{n = 0}] + b,

where x is the source cache the LCM read and the mean is over the earlier
rows of n's document (0 at n = 0). W and b are fitted, with
`train_lcm.Ridge`'s penalty, on every row of `--fit-docs` documents drawn
from the harvest's prefix (seed `--seed`), against the harvest's own states.
The held-out R^2 on the scored tail (the last 32,768 rows, snapped to whole
documents) should read Gate B's.

`--features row` reads from x_n alone (Gate B's `x`): y_n = W^T x_n + b. It
keeps how the LCM reshapes a sentence's own directions and drops the context,
which separates the two parts of the readout above. Its default output is
`lcm_readout_x_b4.npy`.

The output is laid out as the target harvest (harvest_lcm.py), as
recolour.py lays out its rows: the same rows in the same order, with the
target's `.index`, `.kind`, `.langs` and `.docstart` sidecars linked, its own
`.mse_weights.npy` and `.mse_weights_matched.npy` computed as harvest_lcm.py
computes them (matched to the source under `--source-weights`), and a
`.meta.json` with the target's row ranges, the fit and its R^2. Rows are
written through `<out>.partial` and renamed when complete.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from jaxtyping import Float, Int

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from f16cache import link, to_f16  # noqa: E402
from harvest_lcm import SLICED, inverse_variance, whitened_base  # noqa: E402
from ontologize.data.loaders import doc_holdout  # noqa: E402
from spectrum import row_map, stem  # noqa: E402
from train_lcm import Ridge  # noqa: E402

LINKED = (*SLICED, ".docstart.npy")


def context_features(x: Float[np.ndarray, "m d"], first: Int[np.ndarray, "k"]
                     ) -> Float[np.ndarray, "m e"]:
    """[x_n, the mean of its document's earlier rows, 1{n = 0}] per row, for
    the rows of whole documents laid end to end; `first` holds each
    document's first row (sorted, beginning at 0). stateinput.row_features'
    `x_context`, without the padding."""
    m, d = x.shape
    x64 = np.asarray(x, np.float64)
    cs = np.zeros((m + 1, d))
    np.cumsum(x64, axis=0, out=cs[1:])
    at = np.arange(m)
    start = np.asarray(first, np.int64)[np.searchsorted(first, at, side="right") - 1]
    count = at - start
    mean = (cs[at] - cs[start]) / np.maximum(count, 1)[:, None]
    return np.concatenate([x64, mean, (count == 0)[:, None]], 1).astype(np.float32)


def row_features(x: Float[np.ndarray, "m d"], first: Int[np.ndarray, "k"]
                 ) -> Float[np.ndarray, "m d"]:
    """x_n alone: stateinput.row_features' `x`. `first` is unused."""
    return np.asarray(x, np.float32)


FEATURES = {"context": context_features, "row": row_features}
DEFAULT_OUT = {"context": "data/activations/lcm-organism/lcm_readout_b4.npy",
               "row": "data/activations/lcm-organism/lcm_readout_x_b4.npy"}
DESCRIBED = {"context": "x_n, earlier mean, 1{n = 0}", "row": "x_n"}


def doc_chunks(docstart: Int[np.ndarray, "k"], cuts: List[int], n: int, chunk: int
               ) -> List[Tuple[int, int]]:
    """[lo, hi) ranges that cover [0, n), each ending at a document
    boundary (or n) and none crossing a row in `cuts` (themselves
    boundaries): as many whole documents as fit in `chunk` rows, or one
    document that is longer."""
    bounds = np.unique(np.r_[np.asarray(docstart, np.int64), cuts, n])
    bounds = bounds[(bounds > 0) & (bounds <= n)]
    out, lo = [], 0
    while lo < n:
        stop = min([c for c in cuts if c > lo] + [n])
        fits = bounds[(bounds > lo) & (bounds <= min(lo + chunk, stop))]
        hi = int(fits[-1]) if len(fits) else int(bounds[bounds > lo][0])
        out.append((lo, hi))
        lo = hi
    return out


def source_span(lo: int, hi: int, P: int, T: int) -> slice:
    """The source cache rows behind harvest rows [lo, hi), which lie on one
    side of the prefix end P."""
    return slice(lo, hi) if hi <= P else slice(T + lo - P, T + hi - P)


def fit_readout(x_s: np.ndarray, h: np.ndarray, docstart: Int[np.ndarray, "k"],
                docs: Int[np.ndarray, "f"], alpha: float, b: int = 512,
                features=context_features
                ) -> Tuple[Float[np.ndarray, "e d"], Float[np.ndarray, "d"], int]:
    """The ridge from `features` (context_features or row_features) to the
    harvest's states, on every row of the documents numbered `docs` (indices
    into `docstart`, all in the prefix). Returns (W, b, rows fitted)."""
    ends = np.r_[docstart[1:], len(h)]
    ridge = None
    for i in range(0, len(docs), b):
        k = docs[i:i + b]
        lens = ends[k] - docstart[k]
        k, lens = k[lens > 0], lens[lens > 0]
        rows = np.concatenate([np.arange(docstart[j], docstart[j] + n) for j, n in zip(k, lens)])
        first = np.r_[0, np.cumsum(lens)[:-1]]
        F = features(np.asarray(x_s[rows], np.float32), first)
        if ridge is None:
            ridge = Ridge(F.shape[1], h.shape[1])
        ridge.add(F, np.asarray(h[rows], np.float32))
    W, b0 = ridge.solve(alpha)
    return W, b0, int(ridge.n)


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="data/sonar_embeddings/mc4_sent16_f16.npy",
                   help="the cache the LCM read")
    p.add_argument("--source-weights", default="data/out/sonar/sentcache/mse_weights_no837.npy",
                   help="the control's weights on it, which the loss scale is matched to")
    p.add_argument("--target", default="data/activations/lcm-organism/lcm_b4.npy",
                   help="a harvest: its rows and the states the ridge predicts")
    p.add_argument("--features", choices=sorted(FEATURES), default="context",
                   help="context: x_n, the earlier mean and 1{n = 0}; row: x_n alone")
    p.add_argument("--out", default=None, help="default: lcm_readout_b4.npy, or "
                   "lcm_readout_x_b4.npy for --features row, in data/activations/lcm-organism")
    p.add_argument("--fit-docs", type=int, default=20000, help="prefix documents for the fit")
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--holdout", type=int, default=32768, help="the scored tail, snapped")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=65536)
    cfg = p.parse_args(argv)
    resolve = lambda s: Path(s) if Path(s).is_absolute() else ROOT / s  # noqa: E731
    src, src_w, tgt, dst = (resolve(s) for s in (cfg.source, cfg.source_weights, cfg.target,
                                                 cfg.out or DEFAULT_OUT[cfg.features]))
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    features = FEATURES[cfg.features]

    mapping = row_map(tgt)
    if mapping is None:
        raise SystemExit(f"{tgt} has no harvest meta.json with row ranges")
    P, T = mapping
    x_s = np.load(src, mmap_mode="r")
    h = np.load(tgt, mmap_mode="r")
    m = h.shape[0]
    n = T + m - P
    if x_s.shape[0] != n:
        raise SystemExit(f"{src} has {x_s.shape[0]} rows; {tgt}'s rows end at cache row {n}")
    idx_s = np.load(f"{stem(src)}.index.npy", mmap_mode="r")
    idx_t = np.load(f"{stem(tgt)}.index.npy", mmap_mode="r")
    if not (np.array_equal(idx_t[:P], idx_s[:P]) and np.array_equal(idx_t[P:], idx_s[T:n])):
        raise SystemExit(f"{tgt}'s index is not {src}'s on the harvest's rows")
    ds = np.load(f"{stem(tgt)}.docstart.npy").astype(np.int64)
    if P not in ds:
        raise SystemExit(f"the prefix end {P} is not a document boundary of {tgt}")

    prefix_docs = np.flatnonzero(ds < P)
    docs = np.sort(np.random.default_rng(cfg.seed).choice(
        prefix_docs, size=min(cfg.fit_docs, len(prefix_docs)), replace=False))
    W, b0, fit_rows = fit_readout(x_s, h, ds, docs, cfg.alpha, features=features)
    print(f"fit: {len(docs)} documents, {fit_rows} rows", flush=True)

    W32, b32 = W.astype(np.float32), b0.astype(np.float32)
    held = doc_holdout(tgt, cfg.holdout)
    part = dst.with_name(dst.name + ".partial")
    out = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16, shape=h.shape)
    conv = {"max_abs": 0.0, "subnormal": 0, "to_zero": 0, "nonfinite": 0}
    sse = {"tail": 0.0, "scored": 0.0}
    for lo, hi in doc_chunks(ds, [P, m - held], m, cfg.chunk):
        first = np.unique(ds[(ds >= lo) & (ds < hi)] - lo)
        F = features(np.asarray(x_s[source_span(lo, hi, P, T)], np.float32), first)
        y = F @ W32 + b32
        if lo >= P:
            e = float(((y.astype(np.float64) - np.asarray(h[lo:hi], np.float64)) ** 2).sum())
            sse["tail"] += e
            sse["scored"] += e if lo >= m - held else 0.0
        y16, st = to_f16(y)
        conv["max_abs"] = max(conv["max_abs"], st["max_abs"])
        for k in ("subnormal", "to_zero", "nonfinite"):
            conv[k] += st[k]
        out[lo:hi] = y16
    out.flush()
    del out
    if conv["nonfinite"]:
        raise SystemExit(f"{conv['nonfinite']} values are not finite")

    def sst(a: np.ndarray) -> float:
        a = np.asarray(a, np.float64)
        return float(((a - a.mean(0)) ** 2).sum())

    r2 = {"tail": 1 - sse["tail"] / sst(h[P:]), "scored_tail": 1 - sse["scored"] / sst(h[m - held:])}
    print(f"held-out R^2: {json.dumps(r2)}", flush=True)

    y = np.load(part, mmap_mode="r")
    w = inverse_variance(y, P)
    matched = whitened_base(x_s, np.load(src_w)) / whitened_base(y, w)
    side = lambda suf: dst.with_name(dst.stem + suf)  # noqa: E731
    np.save(side(".mse_weights.npy"), w.astype(np.float32))
    np.save(side(".mse_weights_matched.npy"), (w * matched).astype(np.float32))
    for suf in LINKED:
        link(Path(f"{stem(tgt)}{suf}"), side(suf))
    del y
    os.replace(part, dst)
    if doc_holdout(dst, cfg.holdout) != held:
        raise SystemExit(f"the scored tail is {doc_holdout(dst, cfg.holdout)} rows here, "
                         f"{held} in {tgt}")

    meta = {"source": src.name, "target": tgt.name, "n": int(m), "d": int(h.shape[1]),
            "dtype": "float16",
            "rows": {"prefix": [0, P], "tail": [T, n],
                     "note": "harvest row i is cache row i below P, cache row T + i - P after"},
            "holdout": {"rows": cfg.holdout, "harvest": held},
            "fit": {"docs": int(len(docs)), "rows": fit_rows, "seed": cfg.seed,
                    "alpha": cfg.alpha, "features": DESCRIBED[cfg.features]},
            "r2": r2, "conversion": conv,
            "mse_weights": {"rows": P, "max": float(w.max()), "median": float(np.median(w))},
            "mse_weights_matched_scale": matched}
    side(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"out": str(dst), "r2": r2, "conversion": conv}, indent=2))
    return dst


if __name__ == "__main__":
    main()

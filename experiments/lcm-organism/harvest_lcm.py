"""Harvest a Base-LCM's residual stream into a cache the dictionary tools read.

    uv run python experiments/lcm-organism/harvest_lcm.py --run data/out/lcm-organism/lcm_l8_d1024_s42 --block 4
    uv run python experiments/lcm-organism/harvest_lcm.py --run <run> --block 4 --step 0   # the untrained init

**Rows.** Two ranges of the source cache, each in whole documents:
- the prefix `[0, P)`, with P the first document boundary at or after
  `--prefix-rows`: where the dictionaries train (below 4M) and
  `closure.py` fits (4.0M-4.4M);
- the tail `[T, n)`, the rows the LCM never trained on (`train_lcm.py
  --tail-rows`, snapped the same way): where everything is scored.

`--prefix-rows 0` harvests the tail alone, which is all `stateagree.py`
needs (about 0.5 GB at 1024 wide). Such a harvest has no MSE weights, since
there are no training rows to weight by.

Harvest row i is cache row i for i < P, and cache row T + (i - P) after,
so `--rows 4000000`, `--fit-rows 4000000:4400000` and "the last 32,768
rows" mean on the harvest what they mean on the cache. Each row's state is
the residual stream after `--block` blocks at that row's position, given
its document's rows up to it.

**Files**, written through `<out>.partial` and renamed when complete:
- `<out>.npy`: the states, float16 by default (values beyond float16's
  range are refused);
- `.index.npy`, `.kind.npy`, `.langs.npy`: the cache's sidecars for the same
  rows;
- `.docstart.npy`: each document's first row, in the harvest's numbering;
- `.mse_weights.npy`: inverse variance over the prefix rows, scaled to mean
  1, as `encode_acts.py` writes it;
- `.mse_weights_matched.npy`: the same weights rescaled so the whitened
  error of predicting the mean equals SONAR's under the control's weights
  (`gpt2_l8.matched_weights`), so the recipe's loss weights transfer;
- `.meta.json`: the run, step, block, row ranges, precision and the
  conversion's statistics.

States are computed at matmul precision `--precision` (`highest`).
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, Optional, Tuple  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import jax  # noqa: E402
import numpy as np  # noqa: E402
from jaxtyping import Float, Int  # noqa: E402

from f16cache import to_f16  # noqa: E402
from lcm import BaseLCM, batches, doc_spans, gather  # noqa: E402
from ontologize.data.loaders import doc_holdout  # noqa: E402
from train_lcm import WEIGHTS, load_run, manager, split  # noqa: E402

ACTS = ROOT / "data" / "activations"
SLICED = (".index.npy", ".kind.npy", ".langs.npy")
MATCH_ROWS = 1 << 16             # rows sampled for the loss scale, as gpt2_l8


def ranges(docstart: Int[np.ndarray, "n_docs"], n: int, prefix_rows: int,
           train_hi: int) -> Tuple[int, int]:
    """(P, T): the prefix ends at the first document boundary at or after
    `prefix_rows`; the tail starts where the LCM's training documents end."""
    i = int(np.searchsorted(docstart, prefix_rows))
    P = int(docstart[i]) if i < len(docstart) else n
    if P > train_hi:
        raise ValueError(f"the prefix ({P} rows) runs into the tail at {train_hi}")
    return P, train_hi


def harvest_rows(P: int, T: int, n: int) -> Int[np.ndarray, "m"]:
    """The cache row behind each harvest row."""
    return np.r_[np.arange(P), np.arange(T, n)]


def first_rows(doc: Int[np.ndarray, "m"]) -> Int[np.ndarray, "k"]:
    """The first row of each run of equal document ids."""
    doc = np.asarray(doc)
    return np.flatnonzero(np.r_[True, doc[1:] != doc[:-1]]).astype(np.int64)


def inverse_variance(a: np.ndarray, rows: int, chunk: int = 1 << 16
                     ) -> Float[np.ndarray, "d"]:
    """`encode_acts.py`'s weights: 1 / variance over the first `rows` rows,
    by blocks (Welford), scaled to mean 1."""
    d = a.shape[1]
    mean, var, seen = np.zeros(d), np.zeros(d), 0
    for i in range(0, rows, chunk):
        c = np.asarray(a[i:min(i + chunk, rows)], np.float64)
        n_c, mu, v = len(c), c.mean(0), c.var(0)
        delta = mu - mean
        var = (seen * var + n_c * v + delta ** 2 * seen * n_c / (seen + n_c)) / (seen + n_c)
        mean += delta * n_c / (seen + n_c)
        seen += n_c
    w = 1.0 / np.maximum(var, np.finfo(np.float64).tiny)
    return w / w.mean()


def whitened_base(a: np.ndarray, w: Float[np.ndarray, "d"]) -> float:
    """The whitened error of predicting the mean, over the first
    `MATCH_ROWS` rows: `gpt2_l8._whitened_base`."""
    X = np.asarray(a[:MATCH_ROWS], np.float64)
    return float((X.var(0) * np.asarray(w, np.float64)).mean())


def default_out(run: Path, block: int, step: int, latest: int) -> Path:
    suffix = "_init" if step == 0 else ("" if step == latest else f"_step{step}")
    return ACTS / f"{run.name}_b{block}{suffix}.npy"


def harvest(model: BaseLCM, variables, arr, index, docstart, block: int,
            spans, out: np.ndarray, b: int, t: int, precision: str,
            f16: bool) -> Dict[str, float]:
    """Write the states of the documents in `spans` (a list of row ranges)
    into `out`, in row order, and return the conversion's statistics."""
    fwd = jax.jit(lambda v, x, pos, mask: model.apply(
        v, x, pos, mask, block, method=BaseLCM.states))
    stats = {"max_abs": 0.0, "subnormal": 0, "to_zero": 0, "nonfinite": 0}
    at = 0
    with jax.default_matmul_precision(precision):
        for lo, hi in spans:
            starts, lengths = doc_spans(docstart, len(arr), lo, hi)
            if lengths.sum() != hi - lo:
                raise ValueError(f"rows [{lo}, {hi}) are not whole documents")
            for s, n in batches(starts, lengths, b):
                x, pos, mask, _ = gather(arr, index, s, n, t)
                h = np.asarray(fwd(variables, x, pos, mask))[mask]
                if f16:
                    h, st = to_f16(h)
                    stats["max_abs"] = max(stats["max_abs"], st["max_abs"])
                    for k in ("subnormal", "to_zero", "nonfinite"):
                        stats[k] += st[k]
                elif not np.isfinite(h).all():
                    stats["nonfinite"] += int((~np.isfinite(h)).sum())
                out[at:at + len(h)] = h
                at += len(h)
    if at != len(out):
        raise ValueError(f"wrote {at} rows of {len(out)}")
    if stats["nonfinite"]:
        raise ValueError(f"{stats['nonfinite']} states are not finite")
    return stats


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="a train_lcm.py run directory")
    p.add_argument("--step", type=int, default=-1, help="-1: the latest; 0: the untrained init")
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--cache", default=None, help="default: the run's own cache")
    p.add_argument("--prefix-rows", type=int, default=4_400_000)
    p.add_argument("--out", default=None,
                   help="default: data/activations/<run>_b<block>[_init|_step<k>].npy")
    p.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--precision", default="highest")
    p.add_argument("--b", type=int, default=256, help="documents per batch")
    p.add_argument("--match-cache", default=None,
                   help="the SONAR cache the loss scale is matched to; default: the run's cache")
    p.add_argument("--match-weights", default=str(WEIGHTS),
                   help="the control's weights on it (no837)")
    cfg = p.parse_args(argv)

    run = Path(cfg.run)
    model, variables, step, spec = load_run(run, cfg.step)
    tr = spec["train"]
    cache = Path(cfg.cache or tr["cache"])
    arr = np.load(cache, mmap_mode="r")
    index = np.load(cache.with_name(cache.stem + ".index.npy"), mmap_mode="r")
    docstart = np.load(cache.with_name(cache.stem + ".docstart.npy"))
    sp = split(cache, tr["tail_rows"], tr["holdout"])
    if sp != spec["split"]:
        raise SystemExit(f"{cache} splits as {sp}, the run trained on {spec['split']}")
    n = sp["n"]
    P, T = ranges(docstart, n, cfg.prefix_rows, sp["train_hi"])
    src_rows = harvest_rows(P, T, n)

    latest = manager(run, argparse.Namespace(**tr)).latest_step()
    dst = Path(cfg.out) if cfg.out else default_out(run, cfg.block, step, latest)
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".partial")
    out = np.lib.format.open_memmap(part, mode="w+", dtype=np.dtype(cfg.dtype),
                                    shape=(len(src_rows), model.d_model))
    stats = harvest(model, variables, arr, index, docstart, cfg.block,
                    [(0, P), (T, n)], out, cfg.b, tr["t"], cfg.precision,
                    cfg.dtype == "float16")
    out.flush()
    del out

    stem = lambda suf: dst.with_name(dst.stem + suf)  # noqa: E731
    for suf in SLICED:
        side = cache.with_name(cache.stem + suf)
        if side.exists():
            a = np.load(side, mmap_mode="r")
            np.save(stem(suf), np.concatenate([np.asarray(a[:P]), np.asarray(a[T:n])]))
    doc = np.load(stem(".index.npy"), mmap_mode="r")[:, 0]
    np.save(stem(".docstart.npy"), first_rows(doc))

    h = np.load(part, mmap_mode="r")
    matched: Optional[float] = None
    w = inverse_variance(h, P) if P else None
    if w is not None:
        np.save(stem(".mse_weights.npy"), w.astype(np.float32))
    mcache = Path(cfg.match_cache or cache)
    if w is None:
        print("--prefix-rows 0, the tail alone: no prefix to weight by, so no MSE weights")
    elif mcache.exists() and Path(cfg.match_weights).exists():
        matched = (whitened_base(np.load(mcache, mmap_mode="r"), np.load(cfg.match_weights))
                   / whitened_base(h, w))
        np.save(stem(".mse_weights_matched.npy"), (w * matched).astype(np.float32))
    else:
        print(f"no {mcache} or {cfg.match_weights}: .mse_weights_matched.npy not written")
    del h
    os.replace(part, dst)
    held = {"rows": tr["holdout"], "harvest": doc_holdout(dst, tr["holdout"]),
            "cache": doc_holdout(cache, tr["holdout"])}
    if held["harvest"] != held["cache"]:
        raise SystemExit(f"the harvest's scored tail is {held['harvest']} rows, the "
                         f"cache's {held['cache']}: the tail range does not cover it")

    meta = {"source": cache.name, "run": run.name, "step": int(step), "block": cfg.block,
            "model": spec["model"], "n": int(len(src_rows)), "d": model.d_model,
            "dtype": cfg.dtype, "precision": cfg.precision,
            "rows": {"prefix": [0, P], "tail": [T, n],
                     "note": "harvest row i is cache row i below P, cache row T + i - P after"},
            "holdout": held,
            "conversion": stats,
            "mse_weights": None if w is None else
            {"rows": P, "max": float(w.max()), "median": float(np.median(w))},
            "mse_weights_matched_scale": matched}
    stem(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"out": str(dst), **{k: meta[k] for k in ("n", "rows", "conversion")}},
                     indent=2))
    return dst


if __name__ == "__main__":
    main()

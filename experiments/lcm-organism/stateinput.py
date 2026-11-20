"""Gate B: is an LCM block's state more than its own input row?

    uv run python experiments/lcm-organism/stateinput.py --run data/out/lcm-organism/lcm_l8_d1024_s42 --blocks 4 8

For the state h_n at every row of the scored tail's documents, the share of
its variance (held-out R^2) that a ridge explains from:
- `x`: x_n alone;
- `x_context`: x_n, the mean of the run's earlier rows, and an indicator of
  n = 0 (Gate A's context baseline, as an input);
- `alone`: the block's state for x_n read as a one-row document. That is
  everything the model computes from x_n without context, however
  nonlinear, which the two linear readings would otherwise count as context.
  What it leaves is what the context adds.

The ridges are fitted on every row of `--fit-docs` training documents
(drawn with seed 0) and scored on the tail, at matmul precision `highest`.
plan.md's reading: if x_n alone explains nearly all of a block's state, a
dictionary on it is a dictionary on SONAR with extra steps.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, Tuple  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import jax  # noqa: E402
import numpy as np  # noqa: E402
from jaxtyping import Bool, Float  # noqa: E402

from lcm import BaseLCM, batches, doc_spans, gather  # noqa: E402
from train_lcm import Ridge, load_run, split  # noqa: E402

READINGS = ("x", "x_context", "alone")


def row_features(x: Float[np.ndarray, "b t d"], mask: Bool[np.ndarray, "b t"]
                 ) -> Tuple[Float[np.ndarray, "r d"], Float[np.ndarray, "r e"]]:
    """For every real row n, in row order: x_n, and x_n with the mean of the
    run's rows before n and an indicator of n = 0."""
    t = x.shape[1]
    xm = x * mask[..., None]
    count = np.arange(t, dtype=np.float32)[None, :, None]
    mean = (np.cumsum(xm, axis=1) - xm) / np.maximum(count, 1)
    first = np.broadcast_to((count == 0).astype(np.float32), x.shape[:2] + (1,))
    xn = x[mask]
    return xn, np.concatenate([xn, mean[mask], first[mask]], -1)


def r2(pred: Float[np.ndarray, "r d"], y: Float[np.ndarray, "r d"]) -> float:
    """Held-out R^2 over all dimensions together (variance-weighted)."""
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


class States:
    """A block's states for documents in context, and for each row read
    alone as a one-row document, at the precision in force when called."""

    def __init__(self, model: BaseLCM, variables, blocks):
        self.blocks = tuple(blocks)
        f = lambda v, x, pos, mask: tuple(  # noqa: E731
            model.apply(v, x, pos, mask, b, method=BaseLCM.states) for b in self.blocks)
        self.fwd = jax.jit(f)
        self.v = variables

    def __call__(self, x, pos, mask) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """({block: states of the real rows in context}, {block: alone})."""
        b, t, d = x.shape
        ctx = self.fwd(self.v, x, pos, mask)
        alone = self.fwd(self.v, x.reshape(b * t, 1, d), np.zeros((b * t, 1), np.int32),
                         mask.reshape(b * t, 1))
        flat = mask.reshape(-1)
        return ({k: np.asarray(c)[mask] for k, c in zip(self.blocks, ctx)},
                {k: np.asarray(a)[:, 0][flat] for k, a in zip(self.blocks, alone)})


def measure(states: States, arr, index, fit_spans, test_spans, b: int, t: int,
            alpha: float) -> Dict[int, Dict[str, float]]:
    """Fit each reading's ridge on `fit_spans`' rows and score it on
    `test_spans`' rows, per block."""
    d_s = arr.shape[1]
    ridges = None
    for s, n in batches(*fit_spans, b):
        x, pos, mask, _ = gather(arr, index, s, n, t)
        h, alone = states(x, pos, mask)
        xn, xc = row_features(x, mask)
        if ridges is None:
            d_m = next(iter(h.values())).shape[1]
            ridges = {k: {"x": Ridge(d_s, d_m), "x_context": Ridge(2 * d_s + 1, d_m),
                          "alone": Ridge(d_m, d_m)} for k in states.blocks}
        for k in states.blocks:
            ridges[k]["x"].add(xn, h[k])
            ridges[k]["x_context"].add(xc, h[k])
            ridges[k]["alone"].add(alone[k], h[k])
    fits = {k: {r: ridges[k][r].solve(alpha) for r in READINGS} for k in states.blocks}
    sse = {k: {r: 0.0 for r in READINGS} for k in states.blocks}
    ys = {k: [] for k in states.blocks}
    for s, n in batches(*test_spans, b):
        x, pos, mask, _ = gather(arr, index, s, n, t)
        h, alone = states(x, pos, mask)
        xn, xc = row_features(x, mask)
        inputs = {"x": xn, "x_context": xc}
        for k in states.blocks:
            y = h[k].astype(np.float64)
            ys[k].append(y)
            for r in READINGS:
                W, b0 = fits[k][r]
                z = alone[k] if r == "alone" else inputs[r]
                sse[k][r] += float(((z.astype(np.float64) @ W + b0 - y) ** 2).sum())
    out = {}
    for k in states.blocks:
        y = np.concatenate(ys[k])
        sst = float(((y - y.mean(0)) ** 2).sum())
        out[k] = {"rows_fit": int(ridges[k]["x"].n), "rows_test": int(len(y)),
                  **{f"r2_{r}": 1 - sse[k][r] / sst for r in READINGS}}
    return out


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--step", type=int, default=-1, help="-1: the latest; 0: the untrained init")
    p.add_argument("--blocks", type=int, nargs="+", default=[4, 8])
    p.add_argument("--cache", default=None, help="default: the run's own cache")
    p.add_argument("--fit-docs", type=int, default=20000)
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--b", type=int, default=128, help="documents per batch")
    p.add_argument("--precision", default="highest")
    p.add_argument("--out", default=None, help="JSON of the results")
    cfg = p.parse_args(argv)

    model, variables, step, spec = load_run(Path(cfg.run), cfg.step)
    tr = spec["train"]
    cache = Path(cfg.cache or tr["cache"])
    arr = np.load(cache, mmap_mode="r")
    index = np.load(cache.with_name(cache.stem + ".index.npy"), mmap_mode="r")
    docstart = np.load(cache.with_name(cache.stem + ".docstart.npy"))
    sp = split(cache, tr["tail_rows"], tr["holdout"])
    starts, lengths = doc_spans(docstart, sp["n"], 0, sp["train_hi"], min_len=1)
    pick = np.sort(np.random.default_rng(0).choice(len(starts), min(cfg.fit_docs, len(starts)),
                                                   replace=False))
    test = doc_spans(docstart, sp["n"], sp["eval_lo"], sp["n"])
    with jax.default_matmul_precision(cfg.precision):
        res = measure(States(model, variables, cfg.blocks), arr, index,
                      (starts[pick], lengths[pick]), test, cfg.b, tr["t"], cfg.alpha)
    out = {"run": str(cfg.run), "step": int(step), "precision": cfg.precision,
           "blocks": {str(k): v for k, v in res.items()}}
    print(json.dumps(out, indent=2))
    if cfg.out:
        Path(cfg.out).write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    main()

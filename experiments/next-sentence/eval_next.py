"""Score next-sentence and reconstruction codes on held-out sentence pairs.

For each checkpoint, the whitened FVU of its output against the next
sentence (`next`) and against the input itself (`self`), on the pairs in the
cache's held-out tail (`pairs.pair_rows`). FVU is pareto.py's: the
`w`-weighted squared error over the `w`-weighted variance of the target.
Baselines for the next sentence:
- the training mean, about 1 by construction;
- copying the input sentence;
- ridge regression from x_t, fitted on `--fit` training pairs: the linear
  floor a code has to beat.

    uv run python experiments/next-sentence/eval_next.py \\
        --ckpt data/out/sonar/next-sentence/next_h380_l1_s42 \\
               data/out/sonar/next-sentence/self_h380_l1_s42

Seed agreement is `experiments/ste-arm/partition.py --cache` on the
`mc4_sent16` cache (README.md).
"""
import os

# before jax is imported: JAX otherwise reserves 75% of the card at its
# first touch, which ends whatever else is training there. setdefault, so
# an exported choice wins
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import functools  # noqa: E402
from typing import Tuple  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jaxtyping import Float  # noqa: E402

from pairs import pair_rows  # noqa: E402

CACHE = ROOT / "data" / "sonar_embeddings" / "mc4_sent16.npy"
WEIGHTS = ROOT / "data" / "out" / "sonar" / "sentcache" / "mse_weights_no837.npy"


def fvu(pred: Float[np.ndarray, "n d"], target: Float[np.ndarray, "n d"],
        w: Float[np.ndarray, "d"]) -> float:
    """Whitened FVU of `pred` against `target`, as pareto.py scores it."""
    err = ((pred - target) ** 2).mean(0)
    return float((err * w).mean() / (target.var(0) * w).mean())


def ridge(X: Float[np.ndarray, "n d"], Y: Float[np.ndarray, "n d"],
          alpha: float
          ) -> Tuple[Float[np.ndarray, "d d"], Float[np.ndarray, "d"]]:
    """Least squares Y ~ X @ W + b with an L2 penalty `alpha` on W,
    `alpha` scaled by the mean input variance so it is unitless."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    G = Xc.T @ Xc / len(X)
    lam = alpha * float(np.trace(G)) / G.shape[0]
    W = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ Yc / len(X))
    return W, my - mx @ W


def model_outputs(ckpt: str, step: int, X: Float[np.ndarray, "n d"],
                  temperature: float, b: int) -> Float[np.ndarray, "n d"]:
    """The checkpoint's plain forward pass over `X`, in batches."""
    from pareto import load_onto
    model, params, step = load_onto(ckpt, step)
    run = jax.jit(functools.partial(model.apply, {"params": params},
                                    temperature=temperature))
    out = [np.asarray(run(jnp.asarray(X[i:i + b])))
           for i in range(0, len(X), b)]
    return np.concatenate(out)[:len(X)]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="*", default=[],
                   help="checkpoint dirs to score")
    p.add_argument("--step", type=int, default=0,
                   help="checkpoint step (0 = latest)")
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--mse-weights", default=str(WEIGHTS),
                   help="the masked weights this experiment's trainers use "
                        "(coordinate 837 at 0); 'cache' for the cache's own "
                        "<name>.mse_weights.npy")
    p.add_argument("--holdout", type=int, default=32768,
                   help="held-out tail rows, snapped to whole documents")
    p.add_argument("--rows", type=int, default=4_000_000,
                   help="the training prefix, which the ridge fit samples")
    p.add_argument("--fit", type=int, default=200_000,
                   help="training pairs the ridge baseline is fitted on")
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="forward temperature; under `ste` the forward is "
                        "an argmax and this does not matter")
    p.add_argument("--b", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="JSON of the results (default: print only)")
    cfg = p.parse_args()

    cache = Path(cfg.cache)
    arr = np.load(cache, mmap_mode="r")
    w = np.load(cache.with_name(cache.stem + ".mse_weights.npy")
                if cfg.mse_weights == "cache"
                else cfg.mse_weights).astype(np.float64)
    train, held = pair_rows(cache, cfg.rows, cfg.holdout)
    X = np.asarray(arr[held], np.float64)
    Y = np.asarray(arr[held + 1], np.float64)
    print(f"{len(held)} held-out pairs; {len(train)} training pairs")

    rng = np.random.default_rng(cfg.seed)
    fit = np.sort(rng.choice(train, min(cfg.fit, len(train)), replace=False))
    Xf = np.asarray(arr[fit], np.float64)
    Yf = np.asarray(arr[fit + 1], np.float64)
    W, b0 = ridge(Xf, Yf, cfg.alpha)
    res = {"pairs": {"held": int(len(held)), "train": int(len(train))},
           "baselines": {
               "train mean": fvu(np.broadcast_to(Yf.mean(0), Y.shape), Y, w),
               "copy x_t": fvu(X, Y, w),
               f"ridge (alpha {cfg.alpha}, {len(fit)} pairs)":
                   fvu(X @ W + b0, Y, w)},
           "models": {}}
    for name, v in res["baselines"].items():
        print(f"  next-sentence FVU_w, {name}: {v:.4f}")

    for ckpt in cfg.ckpt:
        out = model_outputs(ckpt, cfg.step, X.astype(np.float32),
                            cfg.temperature, cfg.b).astype(np.float64)
        r = {"next": fvu(out, Y, w), "self": fvu(out, X, w)}
        res["models"][ckpt] = r
        print(f"  {Path(ckpt).name}: FVU_w against the next sentence "
              f"{r['next']:.4f}, against the input {r['self']:.4f}")

    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

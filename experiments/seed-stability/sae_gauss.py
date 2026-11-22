"""sae.py's trainer on a Gaussian copy of its cache (`gausscopy.py prepare`).

The k32 SAE seed pair on mc4_4M shares a subspace of its unmatched latents
at 96-97% of the optimum, about twice as well as the data's top principal
directions do (notes.md, 2026-11-20). The stack's subspace agreement turned
out to be the covariance's. This trains the same SAE, at the same settings,
on rows with the cache's mean and covariance and nothing else, so the same
statistic read on this pair says how much of the SAE's agreement the
covariance alone gives.

The training rows are the first n rows of `gausscopy.py`'s training stream
(seed 0, so both seeds of a pair see the same rows), generated once and
held in memory. n defaults to the cache's own training split (its rows less
--eval-rows), so the copy has as many distinct rows as the real run trained
on and the same number of epochs passes over them. The held-out rows are the
last --eval-rows of the copy's eval.npy, which `subspace.py --cache` reads
as it reads a cache's tail. Every other flag is sae.py's.

    uv run python experiments/seed-stability/sae_gauss.py \\
        --gauss data/out/sonar/gausscopy_4M --epochs 150 \\
        --out data/out/sonar/sae_gauss/m5120_k32
"""
import json
import sys
from pathlib import Path

import numpy as np
from jaxtyping import Float

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import sae  # noqa: E402
from gausscopy import BLOCK, block  # noqa: E402


def stream_rows(mean: Float[np.ndarray, "d"], F: Float[np.ndarray, "d d"],
                n: int, seed: int = 0) -> Float[np.ndarray, "n d"]:
    """The first n rows of the stream keyed by seed, as `GaussianSource`
    and `write_rows` give them."""
    X = np.empty((n, F.shape[0]), np.float32)
    for j in range(-(-n // BLOCK)):
        X[j * BLOCK:(j + 1) * BLOCK] = block(mean, F, seed, j)[:n - j * BLOCK]
    return X


def gauss_rows(gauss: Path, n: int, eval_rows: int
               ) -> tuple[Float[np.ndarray, "n d"], Float[np.ndarray, "m d"]]:
    """(n training rows of stream 0, the last eval_rows rows of eval.npy)."""
    z = np.load(gauss / "moments.npz")
    T = stream_rows(z["mean"].astype(np.float32),
                    z["factor"].astype(np.float32), n)
    E = np.load(gauss / "eval.npy", mmap_mode="r")
    assert len(E) >= eval_rows, f"{gauss / 'eval.npy'} has {len(E)} rows"
    return T, np.asarray(E[-eval_rows:], np.float32)


def main(argv=None) -> None:
    p = sae.parser()
    p.description = __doc__
    p.add_argument("--gauss", required=True,
                   help="gausscopy.py prepare's output directory")
    p.add_argument("--gauss-rows", type=int, default=0,
                   help="training rows (0 = the cache's rows less --eval-rows)")
    cfg = p.parse_args(argv)
    assert cfg.out, "--out: sae.py's default name would not say the rows are Gaussian"
    gauss = Path(cfg.gauss)
    n = cfg.gauss_rows or sae.splits(np.load(cfg.cache, mmap_mode="r").shape[0],
                                     cfg.eval_rows, 0)[0]
    print(f"{n:,} Gaussian training rows from {gauss}", flush=True)
    rows = gauss_rows(gauss, n, cfg.eval_rows)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gauss.json").write_text(json.dumps(
        {"gauss": str(gauss), "rows": n, "stream": 0,
         "eval": f"the last {cfg.eval_rows} rows of eval.npy"}, indent=1))
    sae.train(cfg, rows=rows)


if __name__ == "__main__":
    main()

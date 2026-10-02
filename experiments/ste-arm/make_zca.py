"""ZCA whitener for the SONAR cache, as encoder weights.

`inputgeom.py` measures what the layer-0 classifier sees: mean direction
0.56, mean pairwise cosine 0.31, effective dimension 107 of 1024, where
a residual layer sees 0.09/0.009/611. Centering alone fixes the first
two and not the third (107 -> 125); whitening takes it to 902. This
writes the transform that does it, so an `--encoded` arm can start from
a classifier input whose second moments are already conditioned.

`Linear` computes `W x + b` with `weights` shaped (d_out, d_in), so
`z = W_zca (x - mu)` is weights = W_zca, bias = -(W_zca @ mu). ZCA
rather than PCA because it is the whitening transform closest to the
identity, so the classifier's input keeps the embedding's orientation
and the decode side is unchanged.

Estimated on the cache head only. The eval tail every scoring script
holds out never enters the covariance.
"""
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from jaxtyping import Float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--rows", type=int, default=500000,
                   help="cache-head rows for the covariance estimate")
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="tail to exclude; must cover every scorer's split")
    p.add_argument("--eps", type=float, default=1e-3,
                   help="eigenvalue floor as a fraction of the LARGEST "
                        "eigenvalue, which bounds the amplification at "
                        "1/sqrt(eps). Against the mean instead, the floor "
                        "lands below the near-null tail and the transform "
                        "amplifies numerical noise: this cache reaches "
                        "condition 5.7e6 that way, against 31.6 here")
    p.add_argument("--out", default="data/out/sonar/zca_1024.npz")
    return p.parse_args()


def zca(X: Float[np.ndarray, "n d"], eps_frac: float
        ) -> Tuple[Float[np.ndarray, "d d"], Float[np.ndarray, "d"],
                   Float[np.ndarray, "d"]]:
    """ZCA whitener, rescaled so its output has unit mean norm.

    Returns (W, mu, eigenvalues): `W x + b` with `b = -(W @ mu)` is the
    transform, and the floored eigenvalues are returned for reporting.

    The scale matters and the direction does not. `gainshape_in`
    measures the gain on the classifier input and `gained` multiplies
    the layer's output contribution by it, so an encoder that changes
    the input's magnitude rescales the reconstruction. SONAR embeddings
    are exactly unit-norm, so without an encoder that gain is exactly 1
    and the mechanism is invisible; raw ZCA lands at 32 and inflates
    every contribution by that factor. Dividing the whole transform by
    the mean output norm restores the invariant, and since the
    classifier only ever sees the unit direction, it leaves the input
    geometry bit-identical.
    """
    mu = X.mean(0)
    C = np.cov((X - mu).T)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, eps_frac * w.max())
    W = (V * (1.0 / np.sqrt(w))) @ V.T
    scale = np.linalg.norm((X[:8192] - mu) @ W, axis=1).mean()
    return (W / scale).astype(np.float32), mu.astype(np.float32), w


def main() -> None:
    cfg = parse_args()
    mm = np.load(cfg.cache, mmap_mode="r")
    n = min(cfg.rows, mm.shape[0] - cfg.eval_rows)
    X = np.asarray(mm[:n], np.float32)
    W, mu, w = zca(X, cfg.eps)

    Z = (X[:8192] - mu) @ W
    nz = np.linalg.norm(Z, axis=1)
    Zn = Z / nz[:, None]
    ev = np.linalg.eigvalsh(np.cov(Zn.T)).clip(0)
    eff = ev.sum() ** 2 / (ev ** 2).sum()

    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.out, W=W, mu=mu, norm_mean=nz.mean(), norm_sd=nz.std(),
             eff_dim=eff)
    print(f"{n} rows, {X.shape[1]} dims -> {cfg.out}")
    print(f"  eigenvalue range {w.min():.3e} to {w.max():.3e}; the "
          f"transform amplifies by at most {np.sqrt(w.max() / w.min()):.1f}x")
    print(f"  whitened+normalized effective dim {eff:.1f} of {X.shape[1]}")
    print(f"  output norm mean {nz.mean():.4f} sd {nz.std():.4f}; the "
          f"gain must stay near the embedding's own 1.0")


if __name__ == "__main__":
    main()

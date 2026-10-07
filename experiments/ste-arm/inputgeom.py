"""Is a classifier's input direction distribution concentrated?

The premise behind the conditioning hypothesis: layer 0 classifies the
raw embedding and fills 10.6 of 32 entries, while layers 1-4 classify a
residual and fill all 32. If the cause is the input, layer 0's unit
directions should cluster around one dominant direction and the upper
layers' should not.

Reported per layer, on the exact vector `gainshape_in` hands the
classifier:
  |mean|      norm of the mean unit direction. 0 = centered, 1 = all
              samples point the same way.
  mean cos    average pairwise cosine between samples, which is |mean|^2
              for unit vectors, given here as the direct estimate.
  eff. dim    participation ratio of the covariance eigenvalues,
              (sum l)^2 / sum l^2: how many directions the cloud
              actually spans out of d.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from pareto import load_onto

CACHE = "data/sonar_embeddings/mc4_4M.npy"
ROWS, B = 8192, 512


def main(ckpt: str, T: float) -> None:
    """Layer 0 sees the raw embedding whenever `encoded` is off, so every
    arm sharing that setting reports an identical layer-0 row. That is
    the check, not a bug."""
    model, raw, step = load_onto(ckpt, 0)
    params = {"params": raw}
    mm = np.load(CACHE, mmap_mode="r")

    def inputs(module, X: Float[Array, "b d_in"]) -> Float[Array, "l b d"]:
        """The shaped classifier input U of every layer, (l, b, d)."""
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        Us = []
        for i, de in enumerate(module.dictencs):
            U, G = de.gainshape_in(Ein)
            Us.append(U)
            P = de.dict.cluster(de.classifier(U), T)
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
            if i < module.l - 1:
                Ein = module.nextinput(X, R, P.reshape(X.shape[0], -1))
        return jnp.stack(Us)

    f = jax.jit(lambda p, x: model.apply(p, x, method=inputs))
    chunks = [np.asarray(f(params, jnp.asarray(
        mm[-ROWS:][i:i + B], np.float32))) for i in range(0, ROWS, B)]
    U = np.concatenate(chunks, axis=1)            # (l, rows, d)

    print(f"{Path(ckpt).name} step {step}  l={model.l} h={model.h}")
    print(f"{'layer':>6}{'|mean|':>9}{'mean cos':>10}{'eff. dim':>10}{'of d':>7}")
    for i in range(U.shape[0]):
        A = U[i]
        d = A.shape[1]
        # drop the constant coordinate: it is the same for every sample
        # and would read as perfect concentration on its own
        keep = A.std(0) > 1e-8
        A = A[:, keep]
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        m = A.mean(0)
        idx = np.random.default_rng(0).choice(len(A), 2048, replace=False)
        S = A[idx] @ A[idx].T
        cos = (S.sum() - np.trace(S)) / (len(idx) * (len(idx) - 1))
        C = np.cov(A.T)
        ev = np.linalg.eigvalsh(C).clip(0)
        eff = ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30)
        print(f"{i:>6}{np.linalg.norm(m):9.4f}{cos:10.4f}{eff:10.1f}"
              f"{A.shape[1]:>7}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.5e-4)

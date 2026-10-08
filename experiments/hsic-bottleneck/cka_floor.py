"""The biased head-CKA floor on independent heads, and what lowers it.

Scores `hsic.pairwise_head_cka` with both estimators on heads that are
independent by construction: each head draws its entries from its own
Dirichlet(alpha) usage, with a few heads optionally frozen on one entry.
The biased estimator reads chance co-occurrence as dependence at roughly
(k_eff - 1)/(b - 1), k_eff = 1/sum p^2 a head's effective entry count,
and scores a frozen head 0 against every other, so concentrating or
freezing heads is the only way weights can lower it. The unbiased
estimator reads ~0 throughout.

  uv run python experiments/hsic-bottleneck/cka_floor.py
"""
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float

import hsic


def heads(rng: np.random.Generator, b: int, h: int, k: int, alpha: float,
          frozen: int = 0, smooth: float = 0.02,
          ) -> tuple[Float[Array, "b h k"], float]:
    """Independent near-one-hot heads; the first `frozen` always pick entry
    0. Returns the codes and the mean k_eff of the live heads."""
    P = np.zeros((b, h, k))
    P[:, :frozen, 0] = 1.0
    keff = []
    for i in range(frozen, h):
        p = rng.dirichlet(np.full(k, alpha))
        P[np.arange(b), i, rng.choice(k, size=b, p=p)] = 1.0
        keff.append(1.0 / np.sum(p ** 2))
    P = (1 - smooth) * P + smooth / k
    return jnp.asarray(P, jnp.float32), float(np.mean(keff))


def main(b: int = 256, h: int = 32, k: int = 32, draws: int = 5):
    rng = np.random.default_rng(0)
    print(f"b={b} h={h} k={k}, mean of {draws} draws; pred = live-pair "
          f"share * (k_eff - 1)/(b - 1)")
    print(f"{'alpha':>7s} {'frozen':>6s} {'k_eff':>6s} {'biased':>8s} "
          f"{'pred':>8s} {'unbiased':>9s}")
    for alpha in (100.0, 1.0, 0.1, 0.03):
        for frozen in (0, h // 2, h - 4):
            reps = [heads(rng, b, h, k, alpha, frozen) for _ in range(draws)]
            bi = np.mean([float(hsic.pairwise_head_cka(P, estimator="biased"))
                          for P, _ in reps])
            un = np.mean([float(hsic.pairwise_head_cka(P)) for P, _ in reps])
            ke = np.mean([e for _, e in reps])
            live = (h - frozen) * (h - frozen - 1) / (h * (h - 1))
            print(f"{alpha:7g} {frozen:6d} {ke:6.1f} {bi:8.4f} "
                  f"{live * (ke - 1) / (b - 1):8.4f} {un:9.5f}")


if __name__ == "__main__":
    main()

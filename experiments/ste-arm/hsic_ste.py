"""Head-independence penalty for the STE arm, on the current package.

`experiments/hsic-bottleneck/hsic_hyperparams.py` carries the same idea
against an older training path: its probe omits the constant coordinate
layer 0 now takes, and its `loss` takes four arguments where the jitted
step passes six. The HSIC math there is sound and is reused unchanged;
only the plumbing is rewritten here, and the parent's `loss` is called
rather than re-derived so the 16-column stats row and the two
controlled coefficients keep working.

Mechanism, as before: the ghost slot of the training step is free when
`ghost=False`, so `init` builds the step's `apply_fn` from a probe that
mirrors `Ontologizer.withStats` and returns the per-layer post-noise
classifications in that slot, and `loss` adds
`s_hsic_heads * sum_l pairwise_head_cka(P_l)` on top of the inherited
objective. The raw penalty is logged in column 2, which the stock
pipeline labels MSE_ghost and which is 0 for every run without a ghost
path.

Why this pressure for the STE arm: with several hundred heads the
model has no reason to put a variable in one head when it can spread
it across many at no cost, and the head-level language probe found
exactly that. Pairwise CKA between heads' classifications is the
penalty that makes sharing cost something.

`hsic_sigma2` defaults to 1 rather than the median heuristic because
the codes are one-hot: every pairwise squared distance is exactly 0 or
2, so the median heuristic returns 1 on any healthy head anyway, and on
a batch where a head is nearly constant it collapses to its floor and
turns the kernel into a hard indicator. A fixed unit bandwidth is the
same kernel on the healthy heads without that discontinuity.
"""
import functools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

import ontologize.fns.hsic as hsic

from ontologize.ontologizer import Ontologizer
from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import OntoState
from ontologize.fns.keys import get_dtype


def probe_with_codes(module, X: Float[Array, "... d_in"], sd_in: float = 0.0,
                     rng: Optional[PRNGKeyArray] = None, *args, **kwargs
                     ) -> Tuple[Float[Array, "..."],
                                Float[Array, "l b hk"],
                                Float[Array, "l n_stat"],
                                Optional[PRNGKeyArray]]:
    """`Ontologizer.withStats`, with the per-layer classifications kept
    in the ghost slot. Returns (Y, P_all (l, b, h*k), stats, rng).

    `Y` is `(..., d_out)`, or `(l, ..., d_out)` under `deepsup`, exactly
    as the parent's `withStats` returns it."""
    E, rng_K = module.encode(X, sd_in, rng)
    R = module.resid(E)
    E = module.constinput(E)
    stats, Ys, Ps = [], [], []
    for i, dictenc in enumerate(module.dictencs):
        R_0 = R
        R, K, stat, rng_K = dictenc.withStats(R, E, rng=rng_K,
                                              *args, **kwargs)
        stats.append(stat)
        Ps.append(K)
        if module.deepsup:
            Ys.append(module.decode(module.prefix(R, R_0)))
        if i < module.l - 1:
            E = module.nextinput(X, R, K)
    Y = jnp.stack(Ys) if module.deepsup else module.decode(R)
    return Y, jnp.stack(Ps), jnp.stack(stats), rng_K


@dataclass
class HSICSteHyperparams(Hyperparams):
    s_hsic_heads: float = 0.0
    hsic_h: int = 0
    hsic_sigma2: float = 1.0
    hsic_estimator: str = "unbiased"

    def init(self, model: Ontologizer, save_each: int = 1000) -> OntoState:
        x = jnp.zeros((self.b, model.d_in), get_dtype(model.dtype_str))
        params = model.init(self.rng(), x)
        state = OntoState.create(
            model=model, b=self.b, save_each=save_each,
            n_stats=self.n_stats, params=params, tx=self.opt(),
            apply_fn=functools.partial(model.apply, method=probe_with_codes))
        return state.replace(step=0, stats=state.newstats())

    def loss(self, X: Float[Array, "..."], Y: Float[Array, "... d_out"],
             P_all: Float[Array, "l b hk"], stats: Float[Array, "l n_stat"],
             s_L1F: Optional[Float[Array, ""]] = None,
             s_kcossim: Optional[Float[Array, ""]] = None
             ) -> Tuple[Float[Array, ""], Float[Array, "n_stat"]]:
        L, row = super().loss(X, Y, None, stats, s_L1F, s_kcossim)
        if not self.s_hsic_heads:
            return L, row
        assert self.hsic_h > 0, "hsic_h must be the model's h"
        l, b, hk = P_all.shape
        pen = jnp.zeros((), L.dtype)
        for i in range(l):
            pen = pen + hsic.pairwise_head_cka(
                P_all[i].reshape(b, self.hsic_h, hk // self.hsic_h),
                self.hsic_sigma2, self.hsic_estimator)
        L = L + self.s_hsic_heads * pen
        return L, row.at[0].set(L).at[2].set(pen)

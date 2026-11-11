"""The Fano pair term with the slow heads told apart.

`experiments/next-sentence`'s Fano term rewards each slow head for
carrying information between the two sentences of a pair. Nothing in it
compares two slow heads, and the block it trains is redundant: its heads
stack on the one or two slowest directions and agree with one another
nearly as well as with another seed's. The block reproduces; no head in
it is identified.

Slow feature analysis gets distinct features from the same objective by
two constraints the Fano term lacks: its outputs are decorrelated, and
they are ordered, each the slowest function uncorrelated with those
before it. `OrderedPairHyperparams` adds both as one term over the slow
heads:

    dec * mean over pairs j < i of CKA(head i, head j)

with CKA the normalized unbiased HSIC (Song et al.) between the two
heads' partitions of the batch, under the kernel "same entry". The
estimator is unbiased under independence at any batch size, so
independent heads pay nothing; the biased one carries a floor that only a
collapsed head escapes (`ontologize/fns/hsic.py`).

Ordered (the default), head j's code is stop-gradiented in every pair
j < i: head 0 is trained by the Fano term alone, head 1 must also be
independent of head 0, and so on, which gives the block a canonical
order as well as distinct members. `dec_ordered=False` lets the gradient
reach both heads of a pair: decorrelation without the order.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "next-sentence"))

from pairloss import PairHyperparams, slow_codes  # noqa: E402


def hsic_u(Kt: Float[Array, "... n n"], Lt: Float[Array, "... n n"]
           ) -> Float[Array, "..."]:
    """Song et al.'s unbiased HSIC for kernels whose diagonals are already
    zero. Bilinear in the two kernels."""
    n = Kt.shape[-1]
    t1 = jnp.sum(Kt * Lt, (-2, -1))
    t2 = Kt.sum((-2, -1)) * Lt.sum((-2, -1)) / ((n - 1) * (n - 2))
    t3 = 2.0 / (n - 2) * jnp.sum(Kt.sum(-1) * Lt.sum(-1), -1)
    return (t1 + t2 - t3) / (n * (n - 3))


def head_dependence(P_all: Float[Array, "l two_b hk"], h: int, heads: int,
                    ordered: bool) -> Float[Array, ""]:
    """Mean over layers and pairs j < i of the first `heads` heads of the
    CKA between their partitions of the batch's rows: 1 for two heads that
    always agree on which rows share an entry, about 0 for independent
    ones (and slightly negative by chance, the estimator being unbiased).

    Each head's kernel is scaled by its own HSIC to the power -1/2, held
    as a constant, so the term's gradient is that of the cross term alone
    and a head cannot lower it by changing how evenly it uses its entries.
    With `ordered`, the earlier head of each pair is a constant too."""
    P, _ = slow_codes(P_all, h, heads)                    # (l, n, s, k)
    n, s = P.shape[1], P.shape[2]
    K = jnp.einsum("lask,lbsk->lsab", P, P)               # same entry
    Kt = K * (1.0 - jnp.eye(n, dtype=K.dtype))
    own = hsic_u(Kt, Kt)                                  # (l, s)
    g = jax.lax.stop_gradient(
        jnp.where(own > 0, jax.lax.rsqrt(jnp.where(own > 0, own, 1.0)), 0.0))
    Kn = Kt * g[..., None, None]
    if ordered:
        fixed = jax.lax.stop_gradient(Kn)
        before = jnp.cumsum(fixed, axis=1) - fixed        # sum over j < i
        total = hsic_u(Kn, before).sum(-1)
    else:
        S = Kn.sum(1)
        total = 0.5 * (hsic_u(S, S) - hsic_u(Kn, Kn).sum(-1))
    return (total / (s * (s - 1) / 2)).mean()


@dataclass
class OrderedPairHyperparams(PairHyperparams):
    """`PairHyperparams` plus the dependence term between slow heads:

    dec          its weight, per unit of `s_pair` (0 = `PairHyperparams`)
    dec_ordered  stop the gradient at the earlier head of each pair
    """
    dec: float = 0.0
    dec_ordered: bool = True

    def pair_loss(self, P_all: Float[Array, "l two_b hk"]
                  ) -> Tuple[Float[Array, ""], Float[Array, ""]]:
        term, raw = super().pair_loss(P_all)
        if self.dec:
            term = term + self.dec * head_dependence(
                P_all, self.pair_h, self.pair_heads, self.dec_ordered)
        return term, raw

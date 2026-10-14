"""The pair objective of `pair_objective.md`, on the STE arm's training path.

A record is two sentences of one document, concatenated, (2d,)
(`pairs.StackedPairSource`). Every sentence is its own target, so the batch
reaches the training step as an ordinary self-target batch (b, 2d).
`pair_probe` splits it into (2b, d), every pair's first sentence and then
every pair's second, and runs `hsic_ste.probe_with_codes`. That probe mirrors
`Ontologizer.withStats` and returns each layer's codes in the ghost slot,
which is free with `ghost=False`.

`PairHyperparams.loss` splits the target the same way and takes the
inherited loss over all 2b rows. It then adds `s_pair` times a term over the
designated slow heads, chosen by `pair_term`:
- `agree`: the share of slow heads whose entry differs between the two
  sentences of a pair (`switch_share`). A head that always picks one entry
  satisfies it, and does: see notes.md.
- `fano`: minus a lower bound on each slow head's mutual information between
  the two sentences' entries (`fano_bound`). A head scores on it only by
  using its entries evenly and keeping them across the pair, so a collapsed
  head scores 0, as does one that ignores the pairing.
Under `select="ste"` the codes are one-hot going forward, so switch shares
and usage frequencies are exact; their gradients are the surrogate's.

The raw value is logged in column 2, as the HSIC losses do: the switch share
under `agree`, the bound (as a fraction of log k) under `fano`. The stock
pipeline labels that column MSE_ghost, and it is 0 for every run without a
ghost path. Both are taken on the codes each layer decoded with, after logit
noise and winner dropout.
"""
import functools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "ste-arm"))

from dictwd import DictWDHyperparams  # noqa: E402
from hsic_ste import probe_with_codes  # noqa: E402

from ontologize.ontologizer import Ontologizer  # noqa: E402
from ontologize.training.ontostate import OntoState  # noqa: E402


def unstack(X: Float[Array, "b two_d"]) -> Float[Array, "two_b d"]:
    """(b, 2d) pairs -> (2b, d): every pair's first sentence, then every
    pair's second, so row r pairs with row r + b."""
    b, two_d = X.shape
    d = two_d // 2
    return X.reshape(b, 2, d).swapaxes(0, 1).reshape(2 * b, d)


def pair_probe(module: Ontologizer, X: Float[Array, "b two_d"],
               *args, **kwargs):
    """`probe_with_codes` on the unstacked pairs: (Y, codes (l, 2b, h*k),
    stats, rng)."""
    return probe_with_codes(module, unstack(X), *args, **kwargs)


def switch_share(P_all: Float[Array, "l two_b hk"], h: int, heads: int
                 ) -> Float[Array, ""]:
    """Mean over pairs, layers and the first `heads` heads of each layer
    (all of them when 0) of 1 - <p(x), p(x')>, where row r pairs with row
    r + b. For one-hot codes, the share of those heads whose entry differs
    between the two sentences of a pair."""
    P, b = slow_codes(P_all, h, heads)
    return 1.0 - (P[:, :b] * P[:, b:]).sum(-1).mean()


def slow_codes(P_all: Float[Array, "l two_b hk"], h: int, heads: int
               ) -> Tuple[Float[Array, "l two_b s k"], int]:
    """The codes unflattened per head and cut to the first `heads` heads of
    each layer (all of them when 0), with the number of pairs b."""
    l, two_b, hk = P_all.shape
    P = P_all.reshape(l, two_b, h, hk // h)
    if heads:
        P = P[:, :, :heads]
    return P, two_b // 2


def fano_bound(P_all: Float[Array, "l two_b hk"], h: int, heads: int
               ) -> Float[Array, ""]:
    """Mean over layers and the first `heads` heads of each layer (all of
    them when 0) of a lower bound on the mutual information between the
    entries a head gives the two sentences of a pair, as a fraction of
    log k:

        I(a; a') >= H(a) - H_b(s) - s log(k - 1)

    where H(a) is the entropy of the head's entry usage over the batch's 2b
    rows and s its switch share. Guessing a as a' errs with probability s,
    so Fano's inequality bounds H(a | a') by the last two terms. The bound
    is 0 for a head that uses one entry and for one whose two entries are
    independent, and 1 for one that uses every entry equally and never
    switches."""
    P, b = slow_codes(P_all, h, heads)
    k = P.shape[-1]
    s = 1.0 - (P[:, :b] * P[:, b:]).sum(-1).mean(1)              # (l, slow)
    s = jnp.clip(s, 1e-6, 1.0 - 1e-6)
    use = P.mean(1)                                               # (l, slow, k)
    H = -(use * jnp.log(use + 1e-8)).sum(-1)
    H_s = -(s * jnp.log(s) + (1.0 - s) * jnp.log(1.0 - s))
    return ((H - H_s - s * jnp.log(k - 1.0)) / jnp.log(float(k))).mean()


@dataclass
class PairHyperparams(DictWDHyperparams):
    """`DictWDHyperparams`, and so `--dict-init-scale`, plus the pair term:

    s_pair      weight of the pair term
    pair_term   `agree` (the slow heads' switch share, added) or `fano`
                (their mutual-information bound, subtracted)
    pair_h      heads per layer, to unflatten the codes
    pair_heads  slow heads per layer, the first that many; 0 means all
    """
    s_pair: float = 0.0
    pair_term: str = "agree"
    pair_h: int = 0
    pair_heads: int = 0

    def pair_loss(self, P_all: Float[Array, "l two_b hk"]
                  ) -> Tuple[Float[Array, ""], Float[Array, ""]]:
        """The pair term as it enters the loss per unit of `s_pair`, and
        the raw value logged in column 2."""
        if self.pair_term == "agree":
            raw = switch_share(P_all, self.pair_h, self.pair_heads)
            return raw, raw
        raw = fano_bound(P_all, self.pair_h, self.pair_heads)
        return -raw, raw

    def init(self, model: Ontologizer, save_each: int = 1000) -> OntoState:
        """As the parent's, with the step's `apply_fn` swapped for
        `pair_probe` so `loss` sees the codes of both sentences."""
        assert not self.ghost, "the codes ride the ghost slot"
        assert self.pair_term in ("agree", "fano"), self.pair_term
        assert self.pair_h == model.h, \
            f"pair_h {self.pair_h} is not the model's h {model.h}"
        assert 0 <= self.pair_heads <= model.h
        state = super().init(model, save_each)
        return state.replace(apply_fn=functools.partial(
            model.apply, method=pair_probe))

    def loss(self, X, Y, P_all, stats, s_L1F=None, s_kcossim=None):
        """X: the decodes of all 2b rows, (2b, d) or (l, 2b, d) under
        deepsup. Y: the stacked batch (b, 2d), split here as the probe split
        the input. P_all: the codes, from the ghost slot. Returns (L, stats)
        in the stock layout, with the pair term's raw value in column 2."""
        L, row = super().loss(X, unstack(Y), None, stats, s_L1F, s_kcossim)
        term, raw = self.pair_loss(P_all)
        L = L + self.s_pair * term
        return L, row.at[0].set(L).at[2].set(raw)

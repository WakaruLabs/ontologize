"""The pair objective of `pair_objective.md`, on the STE arm's training path.

A record is two sentences of one document, concatenated, (2d,)
(`pairs.StackedPairSource`). Every sentence is its own target, so the batch
reaches the training step as an ordinary self-target batch (b, 2d).
`pair_probe` splits it into (2b, d), every pair's first sentence and then
every pair's second, and runs `hsic_ste.probe_with_codes`. That probe mirrors
`Ontologizer.withStats` and returns each layer's codes in the ghost slot,
which is free with `ghost=False`.

`PairHyperparams.loss` splits the target the same way and takes the
inherited loss over all 2b rows. It then adds `s_pair` times the share of
the designated slow heads whose entry differs between the two sentences of a
pair. Under `select="ste"` the codes are one-hot going forward, so that share
is exact; its gradient is the surrogate's.

The raw share is logged in column 2, as the HSIC losses do; the stock
pipeline labels that column MSE_ghost, and it is 0 for every run without a
ghost path. The share is the training-time counterpart of `transitions.py`'s
"share of heads that switch", restricted to the slow heads and taken on the
codes each layer decoded with, after logit noise and winner dropout.
"""
import functools
import sys
from dataclasses import dataclass
from pathlib import Path

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
    l, two_b, hk = P_all.shape
    b = two_b // 2
    P = P_all.reshape(l, two_b, h, hk // h)
    if heads:
        P = P[:, :, :heads]
    return 1.0 - (P[:, :b] * P[:, b:]).sum(-1).mean()


@dataclass
class PairHyperparams(DictWDHyperparams):
    """`DictWDHyperparams`, and so `--dict-init-scale`, plus the pair term:

    s_pair      weight of the slow heads' switch share
    pair_h      heads per layer, to unflatten the codes
    pair_heads  slow heads per layer, the first that many; 0 means all
    """
    s_pair: float = 0.0
    pair_h: int = 0
    pair_heads: int = 0

    def init(self, model: Ontologizer, save_each: int = 1000) -> OntoState:
        """As the parent's, with the step's `apply_fn` swapped for
        `pair_probe` so `loss` sees the codes of both sentences."""
        assert not self.ghost, "the codes ride the ghost slot"
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
        in the stock layout, with the raw switch share in column 2."""
        L, row = super().loss(X, unstack(Y), None, stats, s_L1F, s_kcossim)
        raw = switch_share(P_all, self.pair_h, self.pair_heads)
        L = L + self.s_pair * raw
        return L, row.at[0].set(L).at[2].set(raw)

"""Train one layer and freeze the rest.

The premise this exists to test: the gradient reaching layer 4 is 17x
smaller than layer 0's, so later layers may be starved. Raw magnitude
turns out to be the wrong measure -- Adam divides by its own running
RMS, so a uniformly smaller gradient is largely absorbed, and what it
cannot absorb is a gradient small relative to its batch-to-batch noise.
Measured on `ste_h76`, that ratio is 1.35, 0.66, 0.71, 0.77, 0.86 by
layer: it falls once and then RISES with depth, so layer 4 is better
conditioned for Adam than layers 1 to 3.

This is the direct test of the same question. Freeze everything but one
layer, hand it a stationary problem, and see whether held-out error
moves. If it does not, the small gradient was reporting a small
remaining opportunity rather than causing one.

Two overrides, in the manner of `whiten_enc.py`. `opt` labels every
parameter outside the chosen `dictencs_<n>` frozen and routes it
through `optax.set_to_zero` under `optax.multi_transform`. `init` takes
the starting parameters from another checkpoint directly.

`init_from` rather than the ordinary resume, because `multi_transform`
changes `opt_state`'s tree structure and the stock restore matches the
saved optimizer state against the new one leaf for leaf. Resuming a
plain-Adam checkpoint into this optimizer raises a structure mismatch
before the first step, so the parameters are loaded on their own and
the optimizer starts fresh. For a short probe that is the intended
behaviour; it does mean the first steps carry an Adam warm-up
transient, which matters if the question is a small improvement.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import optax
from jax.tree_util import KeyPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ontologize.ontologizer import Ontologizer
from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import OntoState


def _in_layer(path: KeyPath, n: int) -> bool:
    """Exact match on the `dictencs_<n>` path component. A substring
    test would make layer 1 also catch layers 10 and up."""
    tag = f"dictencs_{n}"
    return any(getattr(k, "key", None) == tag for k in path)


@dataclass
class LayerFTHyperparams(Hyperparams):
    train_layer: int = -1
    init_from: str = ""

    def init(self, model: Ontologizer, save_each: int = 1000) -> OntoState:
        state = super().init(model, save_each)
        if not self.init_from:
            return state
        from pareto import load_onto
        _, raw, step = load_onto(self.init_from, 0)
        print(f"  starting parameters from {self.init_from} step {step}")
        return state.replace(params={"params": raw})

    def opt(self, *args, **kwargs) -> optax.GradientTransformation:
        base = super().opt(*args, **kwargs)
        if self.train_layer < 0:
            return base
        n = self.train_layer
        labels = lambda p: jax.tree_util.tree_map_with_path(
            lambda path, _: "train" if _in_layer(path, n) else "frozen", p)
        return optax.multi_transform(
            {"train": base, "frozen": optax.set_to_zero()}, labels)

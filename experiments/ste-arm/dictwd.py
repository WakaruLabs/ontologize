"""Weight decay on the dictionary alone, against null-space drift.

The decoder is wider than the output it feeds, so it has a null space,
and a dictionary component lying in it changes no output and receives no
gradient from the reconstruction. Nothing else opposes such a component:
`wd` is 0, and Adam divides by a second moment that is itself near zero
there, so numerical noise is rescaled into full-size steps and the
component random-walks outward for the whole run.

Measured on `ste_h76` between step 10000 and the end, the dictionary's
displacement reaches 11.7x its own norm with atoms rotating to cosine
0.51 -- and 97.7% of that movement is in the null space. Pooled by
energy, 2.4% of the final dictionary is visible downstream at all. The
atoms are not merely partly invisible; the mass of the dictionary is
somewhere the model cannot see.

`L1_F` is the existing force that could oppose this, and it does reach
the right quantity: the dictionary and the classifications are both
non-negative, so `|F|_1 = sum_hk c_hk |W_hk|_1` exactly -- a usage-
weighted L1 of whole atoms, invisible components included. It is aimed
badly rather than aimed elsewhere. Weighting by `c_hk` discounts the
atoms that drifted, because those are selected less often, and the L1
norm understates them again because they are sparser than average. The
top tenth of atoms by norm hold 81.7% of the dictionary's energy at 0.4%
visibility and absorb 7.5% of the penalty; the bottom nine tenths, which
are doing the work, absorb 92.5%. Raising `s_L1F` therefore damages the
code well before it clears the drift.

Decoupled weight decay avoids both faults: it is unconditional, so no
usage discount, and it acts on squared norm, so its pressure lands in
proportion to the energy it is meant to remove. It is still not the
targeted fix -- a projection onto the decoder's row space would be
exactly function-neutral where this also shrinks the visible part -- but
it is one scalar and it composes with everything already here.

Applied through `optax.adamw`'s `mask`, which takes a callable, so the
parameter tree does not have to exist when the optimizer is built. Only
`dictencs_*/dict/weights` is decayed: the classifier, decoder and biases
keep plain Adam, so an arm differs from its control in one place.

Subclasses `InitScaleHyperparams` rather than `Hyperparams` so the arm
keeps the initialization scaling its control used and differs in the
decay alone.

Choosing `dict_wd`: decay moves a component by `lr * wd * W` per step
against an Adam step of order `lr`, so a drifting component settles at
RMS `sqrt(lr / (2 * dict_wd))` per coordinate. At `lr=5e-5` that is
5e-3 for `dict_wd=1`, 1.6e-3 for 10 and 5e-4 for 100, against a median
atom norm of 0.2 over 1536 coordinates. The same decay biases a
component the gradient is actively holding by roughly `dict_wd * |W|`
relative to that gradient, so the cost rises with `dict_wd` where the
benefit saturates. That argues for the middle of the range, but the
estimate assumes Adam's normalized step is order `lr` in both regimes,
which is exactly what is in question -- so pilot before committing a
full arm.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import optax
from jaxtyping import PyTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from initscale import InitScaleHyperparams, _is_dict_weight


def dict_mask(params: PyTree) -> PyTree:
    """Bool tree selecting the dictionary weights and nothing else."""
    return jax.tree_util.tree_map_with_path(
        lambda path, _: _is_dict_weight(path), params)


@dataclass
class DictWDHyperparams(InitScaleHyperparams):
    dict_wd: float = 0.0

    def opt(self, *args, **kwargs):
        if not self.dict_wd:
            return super().opt(*args, **kwargs)
        assert not self.wd, \
            "dict_wd and the global wd would both decay the dictionary; " \
            "set one or the other"
        return optax.adamw(self.lr, weight_decay=self.dict_wd,
                           mask=dict_mask, *args, **kwargs)

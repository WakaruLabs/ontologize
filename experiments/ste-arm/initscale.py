"""Scale the dictionary initialization so depth and width stop fighting it.

The dictionary is `abs()`'d, so a layer's `h` rows sum coherently and its
output magnitude grows with `h`, not `sqrt(h)`. `gained` then multiplies
the next layer's output by that norm, so the cascade compounds it and
the initial residual grows roughly as (c*h)^l. Measured at init, with
380 heads spread over l layers:

    l   h    residual norm
    1   380            220
    2   190       1.22e+04
    4    95       9.77e+06
    5    76       1.84e+08
   10    38       4.17e+13
   19    20       overflow

The live arm starts at 1.84e+08 and trains out of it. At 380 heads and
5 layers it starts at 5.25e+11 and does not: MSE sits at 3.58e+19 from
step 0 and has not moved after 3000 steps, at any temperature. Arity is
irrelevant to this -- k=2 and k=32 agree to three digits.

None of it is a capacity problem. The TRAINED model sits in the
contracting regime, with per-layer gains 1.00, 0.64, 0.53, 0.46, 0.40.
Training finds that regime; it just cannot reach it from far enough
away at lr 1e-5 with clipped gradients.

One scalar removes the head dependence entirely. Scaling every
dictionary by `scale / c0`, where c0 is the layer-0 residual norm this
initialization actually produces, gives 20.1 for the live arm and 20.1
for 380 heads -- identical to three digits, so `h` drops out. `scale`
below 1 starts the cascade further into the contracting regime: 0.5
gives 4.34.

Off by default, because changing every existing arm's initialization
would break comparability with everything already measured.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import KeyPath
from typing import Tuple

from jaxtyping import Array, Float, PyTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ontologize.ontologizer import Ontologizer
from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import OntoState
from ontologize.fns.keys import get_dtype


def _is_dict_weight(path: KeyPath) -> bool:
    keys = [getattr(k, "key", None) for k in path]
    return "dict" in keys and "weights" in keys


def layer0_scales(model: Ontologizer, params: PyTree,
                  X: Float[Array, "b d_in"], temperature: float
                  ) -> Tuple[float, float, float]:
    """Mean ||decode(R_0)||, ||decode(R_0) - X|| and ||X|| at init.

    The second is what one layer leaves behind, and the quantity
    `dict_init_scale` targets. The first is what scaling the dictionary
    actually controls, and the two only agree while the decode dominates
    the input: the residual cannot fall below ||X||, so a target under
    that floor is unreachable and reporting the residual alone would not
    show it."""
    def probe(module, Xb: Float[Array, "b d_in"]
              ) -> Tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        E, _ = module.encode(Xb, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        de = module.dictencs[0]
        U, G = de.gainshape_in(Ein)
        P = de.dict.cluster(de.classifier(U), temperature)
        R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
        Y = module.decode(R)
        return (jnp.mean(jnp.linalg.norm(Y, axis=-1)),
                jnp.mean(jnp.linalg.norm(Y - Xb, axis=-1)),
                jnp.mean(jnp.linalg.norm(Xb, axis=-1)))
    return tuple(float(v) for v in model.apply(params, X, method=probe))


def layer0_residual(model: Ontologizer, params: PyTree,
                    X: Float[Array, "b d_in"], temperature: float) -> float:
    """Mean ||decode(R_0) - X||: what one layer leaves behind at init."""
    return layer0_scales(model, params, X, temperature)[1]


@dataclass
class InitScaleHyperparams(Hyperparams):
    dict_init_scale: float = 0.0
    init_probe_rows: int = 64

    # cache to draw the probe batch from. `gained` multiplies a layer's
    # output by its input norm, so the residual this measures scales with
    # the data: a unit-norm probe against a cache that is not unit-norm
    # reports a c0 too small by that factor, once per layer, and the
    # correction comes out inverted. The offset and anisotropy matter as
    # well, through `constinput` and `gainshape_in`, so the probe is real
    # rows rather than random ones at a matched norm. Unset falls back to
    # unit-norm Gaussians, which is what SONAR's embeddings are.
    init_probe_cache: str = None

    def probe_batch(self, model: Ontologizer) -> Float[Array, "b d_in"]:
        d = get_dtype(model.dtype_str)
        if self.init_probe_cache:
            X = np.load(self.init_probe_cache, mmap_mode="r")
            return jnp.asarray(np.asarray(X[:self.init_probe_rows]), d)
        X = jax.random.normal(jax.random.PRNGKey(0),
                              (self.init_probe_rows, model.d_in), d)
        return X / jnp.linalg.norm(X, axis=-1, keepdims=True)

    def init(self, model: Ontologizer, save_each: int = 1000) -> OntoState:
        state = super().init(model, save_each)
        if not self.dict_init_scale:
            return state
        X = self.probe_batch(model)
        y0, c0, xn = layer0_scales(model, state.params, X, self.temperature)
        # the target is a MULTIPLE of the mean input norm, not an absolute
        # one. It sets how large layer 0's decode is against the input it
        # has to explain, which is the scale-free statement; an absolute
        # target under ||X|| is below the residual's floor and unreachable.
        # SONAR embeddings are unit-norm, so the two readings coincide
        # there and every measurement taken against them is unchanged.
        target = self.dict_init_scale * xn
        s = target / max(c0, 1e-9)
        print(f"  init probe: {X.shape[0]} rows from "
              f"{self.init_probe_cache or 'unit-norm Gaussians'}, "
              f"mean |x| {xn:.4g}")
        print(f"  dictionary init: layer-0 decode {y0:.4g}, residual {c0:.4g}"
              f" -> scaling by {s:.4g} for a target of "
              f"{self.dict_init_scale:g} x |x| = {target:.4g}")
        params = jax.tree_util.tree_map_with_path(
            lambda path, v: v * s if _is_dict_weight(path) else v,
            state.params)
        y1, c1, _ = layer0_scales(model, params, X, self.temperature)
        print(f"  after scaling: layer-0 decode {y1:.4g}, residual {c1:.4g}")
        return state.replace(params=params)

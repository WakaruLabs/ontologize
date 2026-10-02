"""Install a fixed ZCA whitener as the encoder.

The conditioning hypothesis says the flat arm underfills its code
because its classifier sees the raw embedding cloud -- effective
dimension 107 of 1024, mean pairwise cosine 0.31 -- where every
residual layer sees 507 to 611 and under 0.06 and fills its code
completely. With `encoded`, the encoder is the only thing standing
between the cache and the classifier, and the reconstruction target
stays the raw embedding either way, so replacing it with a known-good
transform tests the hypothesis without asking training to find one.

Two overrides on `Hyperparams`, in the manner of `hsic_ste.py`:

  init  after the parent builds the state, overwrite the encoder's
        weights and bias with `make_zca.py`'s transform. `Linear`
        computes `W x + b`, so `z = W_zca (x - mu)` is weights = W_zca,
        bias = -(W_zca @ mu).
  opt   label the encoder subtree `frozen` and everything else `train`,
        then `optax.multi_transform` with `set_to_zero` on the frozen
        label.

Frozen rather than merely initialized on purpose: the question is
whether a conditioned input fixes utilization, and a trainable encoder
can drift back toward the raw geometry and leave the answer
unattributable. If this arm works, the trainable version is the
follow-up that asks whether training would have found it.

`TrainingEnv.init` restores over this on resume, which is correct --
the whitener is in the checkpoint by then.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax.tree_util import KeyPath
from jaxtyping import Float, PyTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ontologize.ontologizer import Ontologizer
from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import OntoState

ENCODER_KEY = "encoder"


def _is_encoder(path: KeyPath) -> bool:
    """Exact match on a path component, not a substring of the rendered
    path. `"encoder" in "decoder"` is False by one character, and a
    substring test that silently froze the decoder would look exactly
    like the whitener working."""
    return any(getattr(k, "key", None) == ENCODER_KEY for k in path)


def load_zca(path: str, tol: float = 0.1
             ) -> Tuple[Float[np.ndarray, "d d"], Float[np.ndarray, "d"]]:
    """(W, mu) as float32, from make_zca.py's npz.

    Refuses a transform whose output norm is far from 1. `gainshape_in`
    measures the gain on the classifier input and `gained` multiplies
    the layer's output contribution by it, so an encoder that changes
    the input's magnitude silently rescales the reconstruction by the
    same factor. Unscaled ZCA lands at 32 and costs a run to notice:
    training error sits three orders of magnitude above the baseline
    while everything else looks ordinary.
    """
    d = np.load(path)
    m = float(d["norm_mean"]) if "norm_mean" in d else None
    if m is None or abs(m - 1.0) > tol:
        raise SystemExit(
            f"{path} has output norm {m}; it must be within {tol} of 1, "
            "since SONAR embeddings are unit-norm and the layer gain is "
            "measured on the classifier input. Rebuild with make_zca.py")
    return np.asarray(d["W"], np.float32), np.asarray(d["mu"], np.float32)


def install(params: PyTree, W: Float[np.ndarray, "d d"],
            mu: Float[np.ndarray, "d"]) -> PyTree:
    """Set the encoder's weights to W and its bias to -(W @ mu)."""
    flat = jax.tree_util.tree_leaves_with_path(params)
    found = {jax.tree_util.keystr(p) for p, _ in flat if _is_encoder(p)}
    if not found:
        raise SystemExit(
            "no encoder in the parameter tree; --encoded builds it, and "
            "without it the model has a Sparse passthrough instead")

    b = -(W @ mu)

    def setleaf(path: KeyPath, v: Any) -> Any:
        if not _is_encoder(path):
            return v
        name = jax.tree_util.keystr(path).rstrip("']").split("'")[-1]
        if name == "weights":
            if v.shape != W.shape:
                raise SystemExit(
                    f"encoder weights are {v.shape}, whitener is {W.shape}; "
                    "--e-enc must equal the embedding width")
            return jnp.asarray(W, v.dtype)
        if name == "bias":
            return jnp.asarray(b, v.dtype)
        return v

    return jax.tree_util.tree_map_with_path(setleaf, params)


@dataclass
class WhitenEncHyperparams(Hyperparams):
    zca_path: str = ""
    freeze_enc: bool = True

    def init(self, model: Ontologizer,
             save_each: int = 1000) -> OntoState:
        state = super().init(model, save_each)
        if not self.zca_path:
            return state
        W, mu = load_zca(self.zca_path)
        return state.replace(params=install(state.params, W, mu))

    def opt(self, *args, **kwargs) -> optax.GradientTransformation:
        base = super().opt(*args, **kwargs)
        if not self.freeze_enc:
            return base
        labels = lambda p: jax.tree_util.tree_map_with_path(
            lambda path, _: "frozen" if _is_encoder(path) else "train", p)
        return optax.multi_transform(
            {"train": base, "frozen": optax.set_to_zero()}, labels)

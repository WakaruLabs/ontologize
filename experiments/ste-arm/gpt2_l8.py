"""Base configuration for the GPT-2 model organism, as `train_ste.py --base
gpt2_l8` reads it.

`sonar.py` is both the live SONAR run and the config every arm inherits, so
this takes all of it with `import *` and overrides only what the change of
model organism actually changes: the widths, the cache and its sidecars,
the output directory, and the MSE weighting. The loss weights, schedules,
selection rule, noise and architecture flags are deliberately NOT touched,
so a difference between the two arms is a difference in the data rather
than in what was asked of the model.

Two of the overrides are not free choices:

`holdout` is snapped up to a whole-document tail. A token-level cache has
many rows per document, and `NpyDataSource(holdout=)` takes a contiguous
tail, so a round number would leave the eval split holding other positions
from documents the model trained on. `doc_holdout` reads the sidecar and
rounds; see its docstring.

`mse_weights` is rescaled, not just recomputed. Every loss weight in
`sonar.py` is a ratio against the whitened MSE, and the two caches sit
orders of magnitude apart in that quantity: SONAR embeddings are unit-norm
by construction while a residual stream is not, and neither per-dimension
whitening nor mean-1 normalization removes the overall scale. Left alone,
every penalty would be negligible against reconstruction and the run would
be a plain autoencoder wearing the arm's flags. Scaling the weight vector
so the whitened constant-predictor MSE matches SONAR's puts the MSE term in
the same units the weights were calibrated in, which is the one change that
carries all of them at once. `sae.py` divides by this same quantity to form
FVU, so the SAE comparison is unaffected either way.
"""
from sonar import *                              # noqa: F401,F403

from pathlib import Path

import numpy as np

import sonar as _sonar
from ontologize.data.loaders import doc_holdout

_root = Path(__file__).resolve().parents[2]
_acts = _root / "data" / "activations"
_name = _sonar._env("ACTS", "gpt2_l8")           # which harvest to train on

# GPT-2 small's residual width, and sonar.py's 2x dictionary-to-input ratio
d = 768
e_dec = 2 * d
e_enc = 2 * d                                    # only read under `encoded`

encoder = "gpt2"
cache = _acts / f"{_name}.npy"
out = _root / "data" / "out" / _name / _sonar._env("RUN", "resid_nc_hm_c0")

_ROWS = 1 << 16                                  # rows sampled for the scale


def _whitened_base(cache_path, weights) -> float:
    """Whitened constant-predictor MSE, `sae.py`'s `base_w`: the error a
    model that only ever predicts the mean would incur under `weights`."""
    X = np.asarray(np.load(cache_path, mmap_mode="r")[:_ROWS], np.float64)
    return float((X.var(0) * np.asarray(weights, np.float64)).mean())


def matched_weights() -> Path:
    """This cache's inverse-variance weights, rescaled so its whitened MSE
    reads on SONAR's scale. Written next to the cache on first use."""
    dst = cache.with_name(cache.name.replace(".npy", ".mse_weights_matched.npy"))
    if dst.exists():
        return dst
    src = cache.with_name(cache.name.replace(".npy", ".mse_weights.npy"))
    if not src.exists():
        raise FileNotFoundError(
            f"{src} missing; encode_acts.py writes it beside the cache")
    if not (Path(_sonar.cache).exists() and Path(_sonar.mse_weights).exists()):
        raise FileNotFoundError(
            f"matching the loss scale needs SONAR's cache and weights "
            f"({_sonar.cache}, {_sonar.mse_weights}). Point --mse-weights at "
            f"{src} to train unmatched, knowing every penalty in sonar.py is "
            f"then priced against a different MSE scale")
    w = np.load(src).astype(np.float64)
    ratio = (_whitened_base(_sonar.cache, np.load(_sonar.mse_weights))
             / _whitened_base(cache, w))
    np.save(dst, (w * ratio).astype(np.float32))
    print(f"{dst.name}: scaled x{ratio:.4g} to put the whitened MSE on "
          f"SONAR's scale, so sonar.py's loss weights transfer")
    return dst


if cache.exists():
    holdout = doc_holdout(cache, _sonar.holdout)
    mse_weights = str(matched_weights())
else:                                            # -h and imports still work
    holdout = _sonar.holdout
    mse_weights = None

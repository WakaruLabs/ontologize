# Force CPU before jax loads: the tiny models here don't need the GPU, and
# tests must be runnable alongside a live training run without touching its
# VRAM. (Checkpoints created inside tests are CPU-sharded, so this also
# avoids the GPU-sharding restore problems of real run checkpoints.)
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import jax
import jax.numpy as jnp
import pytest

from ontologize.ontologizer import Ontologizer
from ontologize.layers.dictblock import DictBlock

# Small version of the SONAR run architecture: bilinear classifiers,
# residual forwarding, deep supervision. Individual tests override flags.
KW = dict(d_in=64, d_out=64, e_dec=128, k=8, h=4, l=3, n=2, gate="none",
          forward="resid", deepsup=True,
          dtype_str="float32", dtype_p_str="float32")
B = 32


@pytest.fixture(scope="session")
def X():
    """Unit-norm inputs, like SONAR embeddings."""
    X = jax.random.normal(jax.random.PRNGKey(1), (B, KW["d_in"]), jnp.float32)
    return X / jnp.linalg.norm(X, axis=-1, keepdims=True)


DB_KW = dict(k=8, d=16, h=4, dtype_str="float32", dtype_p_str="float32")


# Retired statistics keep their slots and read NaN (see
# `Hyperparams.RETIRED_STATS`), so "the stats are finite" means the live
# columns are. Positions in the per-layer `DictEnc.withStats` row, and in a
# loss.csv row, derived from the authoritative layouts.
from ontologize.training.config import Hyperparams
from ontologize.visualize.loss import COLUMNS

RETIRED_LAYER = tuple(i - 1 for i in Hyperparams.RETIRED_STATS)
RETIRED_LOSS = tuple(COLUMNS.index(c) for c in ("cossim_h", "cossim_flat"))
LAYER_WIDTH = len(Hyperparams(1, 1, 1).s_loss()[0]) - 1
N_STATS = len(COLUMNS)


def live(x, retired):
    """`x` without its retired columns (last axis)."""
    keep = [i for i in range(x.shape[-1]) if i not in retired]
    return jnp.asarray(x)[..., jnp.array(keep)]


def finite_live(x, retired) -> bool:
    """Every live column finite, and every retired one NaN."""
    x = jnp.asarray(x)
    return (bool(jnp.all(jnp.isfinite(live(x, retired))))
            and bool(jnp.all(jnp.isnan(x[..., jnp.array(retired)]))))


@pytest.fixture(scope="session")
def dictblock():
    """Standalone DictBlock with params and a batch of soft classifications."""
    db = DictBlock(**DB_KW)
    K = jax.random.normal(jax.random.PRNGKey(3), (16, DB_KW["h"], DB_KW["k"]))
    params = db.init(jax.random.PRNGKey(4), K)
    P = db.apply(params, K, 0.5, method=DictBlock.cluster)
    return db, params, P


@pytest.fixture(scope="session")
def build(X):
    """Factory: Ontologizer with KW plus overrides, and freshly initialized
    params. Params are seed-deterministic, so two models differing only in
    gradient-path flags (deepsup_sg, *_loss gates) get identical params."""
    def _build(**overrides):
        model = Ontologizer(**{**KW, **overrides})
        params = model.init(jax.random.PRNGKey(0), X)
        return model, params
    return _build

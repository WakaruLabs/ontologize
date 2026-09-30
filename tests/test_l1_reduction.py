# L1_K and L1_F are batch means: each sample's L1 norm over its feature
# axes, averaged over the batch. The reduction used to be a bare jnp.sum,
# which made both stats -- and any s_L1K/s_L1F calibrated against them --
# scale with batch size, so the same weight meant different strengths at
# different `b`. The batch mean also matches sae.py's convention for the
# same quantity, so the Ontologizer and SAE L1 columns are comparable.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.fns.loss import batchmean, l1
from ontologize.layers.dictblock import DictBlock
from ontologize.ontologizer import Ontologizer

from conftest import KW, B, DB_KW


def test_batchmean_matches_its_definition():
    x = jax.random.normal(jax.random.PRNGKey(0), (16, 4, 8))
    got = float(l1(x, batchmean))
    want = float(jnp.abs(x).reshape(16, -1).sum(-1).mean())
    assert got == pytest.approx(want, rel=1e-6)
    # and is the old unreduced sum divided by the batch size
    assert got == pytest.approx(float(jnp.abs(x).sum()) / 16, rel=1e-6)


def test_batch_size_invariant():
    # the point of the change: tiling the batch must not move the stat,
    # which a bare jnp.sum would double
    x = jax.random.normal(jax.random.PRNGKey(1), (16, 4, 8))
    x2 = jnp.concatenate([x, x])
    assert float(l1(x2, batchmean)) == pytest.approx(float(l1(x, batchmean)),
                                                     rel=1e-5)
    assert float(l1(x2)) == pytest.approx(2 * float(l1(x)), rel=1e-5)


def test_matches_sae_convention():
    # sae.py logs l1_eval as jnp.abs(z).sum(-1).mean() on a (b, m) code;
    # batchmean must agree with it on that shape
    z = jnp.abs(jax.random.normal(jax.random.PRNGKey(2), (32, 64)))
    assert float(l1(z, batchmean)) == pytest.approx(
        float(jnp.abs(z).sum(-1).mean()), rel=1e-6)


def test_dictblock_stat_is_the_batch_mean():
    # withStats takes LOGITS and clusters them internally, so the expected
    # value has to come from the probabilities it returns, not from its input
    db = DictBlock(**DB_KW)
    K = jax.random.normal(jax.random.PRNGKey(5), (16, DB_KW["h"], DB_KW["k"]))
    params = db.init(jax.random.PRNGKey(6), K)
    F, P, stats, _ = db.apply(params, K, method=DictBlock.withStats)
    Fs = db.apply(params, P, method=DictBlock.hfwd)
    # DictBlock stats row is [L1_F, entropy, cossim_b, cossim_h, KL_m]
    assert float(stats[0]) == pytest.approx(
        float(jnp.abs(Fs).reshape(Fs.shape[0], -1).sum(-1).mean()), rel=1e-4)


def test_model_stats_scale_with_features_not_batch(build):
    # end to end: L1_K and L1_F must be unchanged by a duplicated batch
    model, params = build()
    X = jax.random.normal(jax.random.PRNGKey(3), (B, KW["d_in"]), jnp.float32)
    X = X / jnp.linalg.norm(X, axis=-1, keepdims=True)

    def stats(Xb):
        _, s, _ = model.apply(params, Xb, temperature=0.5,
                              method=Ontologizer.withStats)
        return np.asarray(s)

    one, two = stats(X), stats(jnp.concatenate([X, X]))
    for col, name in ((0, "L1_K"), (1, "L1_F")):
        assert one[:, col] == pytest.approx(two[:, col], rel=1e-4), name

# DictBlock.fast_stats: withStats computed without materializing the
# per-head outputs must match the reference path exactly when there is no
# noise (values and gradients), and match its noise distribution otherwise.
import jax
import jax.numpy as jnp
import pytest

from conftest import DB_KW, KW
from ontologize.layers.dictblock import DictBlock
from ontologize.ontologizer import Ontologizer

GATES = dict(sparse=True, entropy_loss=True, cossim_loss=True,
             bcossim_loss=True, hmean_loss=True)


def pair(**kw):
    ref = DictBlock(**DB_KW, **kw)
    fast = DictBlock(**DB_KW, fast_stats=True, **kw)
    K = jax.random.normal(jax.random.PRNGKey(3), (64, DB_KW["h"], DB_KW["k"]))
    params = ref.init(jax.random.PRNGKey(4), K)
    return ref, fast, params, K


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("gates", [{}, GATES])
def test_noiseless_values_and_grads_match(scaled, gates):
    ref, fast, params, K = pair(**gates)
    S = jnp.abs(jax.random.normal(jax.random.PRNGKey(5), K.shape[:-1])) if scaled else None

    def run(model, p):
        F, P, st, _ = model.apply(p, K, S, 0.0, None, temperature=0.7,
                                  method=DictBlock.withStats)
        return F, P, st

    for a, b in zip(run(ref, params), run(fast, params)):
        assert jnp.allclose(a, b, rtol=1e-5, atol=1e-6)

    def loss(model, p):
        F, _, st = run(model, p)
        return (F ** 2).mean() + (st * jnp.array([1e-3, 1.0, 1.0, 1.0, 1.0])).sum()

    ga = jax.grad(lambda p: loss(ref, p))(params)
    gb = jax.grad(lambda p: loss(fast, p))(params)
    for a, b in zip(jax.tree_util.tree_leaves(ga), jax.tree_util.tree_leaves(gb)):
        assert jnp.allclose(a, b, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("noise", ["normal", "featvar", "batchnorm"])
def test_noise_distribution_matches(noise):
    ref, fast, params, K = pair(noise=noise)
    sd = 0.3

    def noise_of(model, key):
        Fn, _, _, _ = model.apply(params, K, None, sd, key, temperature=0.7,
                                  method=DictBlock.withStats)
        F, _, _, _ = model.apply(params, K, None, 0.0, None, temperature=0.7,
                                 method=DictBlock.withStats)
        return Fn - F

    keys = jax.random.split(jax.random.PRNGKey(9), 400)
    vr = jax.vmap(lambda k: noise_of(ref, k))(keys).var(0)
    vf = jax.vmap(lambda k: noise_of(fast, k))(keys).var(0)
    # 400 draws: per-entry variance estimates agree to ~15% (chi-square)
    assert jnp.median(jnp.abs(vf / vr - 1)) < 0.1
    assert jnp.allclose(vf.mean(), vr.mean(), rtol=0.05)


def test_interventions_use_reference_path():
    ref, fast, params, K = pair()
    kw = dict(temperature=0.7, h_set=jnp.array([0]), k_set=jnp.array([2]))
    a = ref.apply(params, K, None, 0.0, None, method=DictBlock.withStats, **kw)
    b = fast.apply(params, K, None, 0.0, None, method=DictBlock.withStats, **kw)
    for x, y in zip(a[:3], b[:3]):
        assert jnp.array_equal(x, y)


def test_ontologizer_training_forward_matches(X):
    flags = dict(resid_norm=True, resid_const=True)
    ref = Ontologizer(**{**KW, **flags})
    fast = Ontologizer(**{**KW, **flags, "fast_stats": True})
    params = ref.init(jax.random.PRNGKey(0), X)
    a = ref.apply(params, X, 0.0, None, temperature=0.5, method=Ontologizer.withStats)
    b = fast.apply(params, X, 0.0, None, temperature=0.5, method=Ontologizer.withStats)
    for x, y in zip(a[:2], b[:2]):
        assert jnp.allclose(x, y, rtol=1e-5, atol=1e-6)

# The experiment-branch options: logit_norm (standardized logits, so the
# temperature alone sets their spread), resid_first (layer 0 consumes the
# conditioned residual too) and resid_gain (gain-shape: each layer's
# contribution scales with the norm of the residual it consumed).
import jax
import jax.numpy as jnp
import pytest

from conftest import KW
from ontologize.ontologizer import Ontologizer
from ontologize.layers.dictenc import DictEnc

COND = dict(resid_norm=True, resid_const=True)


def test_logit_norm_standardizes_per_head(build, X):
    model, params = build(logit_norm=True)

    def probe(module, X):
        return module.dictencs[0].logits(X)

    K = model.apply(params, X, method=probe)
    assert jnp.allclose(K.mean(-1), 0.0, atol=1e-5)
    assert jnp.allclose(K.std(-1), 1.0, atol=1e-3)


def test_logit_norm_off_is_raw_classifier(build, X):
    model, params = build()

    def probe(module, X):
        de = module.dictencs[0]
        return de.logits(X), de.classifier(X)

    K, raw = model.apply(params, X, method=probe)
    assert jnp.array_equal(K, raw)


def test_resid_first_layer0_input(build, X):
    model, params = build(resid_first=True, **COND)
    w = params["params"]["dictencs_0"]["classifier"]["weight"]
    assert w.shape[-1] == KW["d_out"] + 1

    def probe(module, X):
        R = module.resid(X)
        return module.first_input(X, X, R)

    E = model.apply(params, X, method=probe)
    assert jnp.allclose(jnp.linalg.norm(E[..., :-1], axis=-1), 1.0, atol=1e-5)
    assert jnp.allclose(E[..., -1], 1.0)


def test_resid_first_breaks_layer0_sign_blindness(build, X):
    blind, pb = build(**COND)
    seeing, ps = build(resid_first=True, **COND)

    def p0(module, X):
        R = module.resid(X)
        return module.dictencs[0].classify(module.first_input(X, X, R))

    Pb, Pb_neg = blind.apply(pb, X, method=p0), blind.apply(pb, -X, method=p0)
    Ps, Ps_neg = seeing.apply(ps, X, method=p0), seeing.apply(ps, -X, method=p0)
    assert jnp.allclose(Pb, Pb_neg, atol=1e-6)          # even logits
    assert not jnp.allclose(Ps, Ps_neg, atol=1e-4)


def test_gain_shape_is_positively_homogeneous(build, X):
    """With every layer seeing only residual directions and scaling its
    output by the residual norm (and no decoder bias), f(cX) = c f(X)."""
    model, params = build(resid_first=True, resid_gain=True, **COND)
    Y = model.apply(params, X)
    Y3 = model.apply(params, 3.0 * X)
    assert jnp.allclose(Y3, 3.0 * Y, rtol=1e-4, atol=1e-5)


def test_without_gain_not_homogeneous(build, X):
    model, params = build(resid_first=True, **COND)
    Y = model.apply(params, X)
    Y3 = model.apply(params, 3.0 * X)
    assert not jnp.allclose(Y3, 3.0 * Y, rtol=1e-2)


@pytest.mark.parametrize("flags", [
    dict(logit_norm=True),
    dict(resid_first=True, **COND),
    dict(resid_first=True, resid_gain=True, **COND),
    dict(resid_first=True, resid_gain=True, logit_norm=True, **COND),
])
def test_withstats_matches_call_and_is_finite(build, X, flags):
    """Training forward (no noise, no rng) agrees with the plain forward on
    the final prefix, and gradients are finite."""
    model, params = build(**flags)
    Y = model.apply(params, X, temperature=0.5)
    Ys, stats, _ = model.apply(params, X, 0.0, None, temperature=0.5,
                               method=Ontologizer.withStats)
    assert jnp.allclose(Ys[-1], Y, atol=1e-5)

    def loss(p):
        Ys, _, _ = model.apply(p, X, 0.0, None, temperature=0.5,
                               method=Ontologizer.withStats)
        return ((Ys - X) ** 2).mean()

    g = jax.grad(loss)(params)
    assert all(bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves(g))

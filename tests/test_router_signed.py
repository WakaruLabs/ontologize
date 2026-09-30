"""`DictEnc.router_signed`, and that every path agrees on `S`'s sign.

`S` is one scalar gain per head, so the `abs` in `scale` keeps each head's
contribution additive. `router_signed` drops it, which lets the input
decide a head's polarity.

The agreement tests are the load-bearing ones. `scale` is meant to be the
only site that decides the convention, and `__call__` once applied the
router raw while `withClusts` and `withStats` went through `scale`, so a
`scaled` model evaluated one way disagreed with the same model trained the
other. `DictEnc.fwd` is excluded deliberately: it is the unbiased-linear
path used to compose ghost gradients, and takes `S` raw by contract.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.layers.dictenc import DictEnc
from ontologize.ontologizer import Ontologizer


D, K, H, B = 32, 8, 4, 16


def enc(router_signed, gate_router="none", **kw):
    m = DictEnc(D, D, K, H, n=2, scaled=True, gate_router=gate_router,
                router_signed=router_signed, dtype_str="float32", **kw)
    p = m.init(jax.random.key(0), jnp.zeros((B, D)))
    return m, p


X = jax.random.normal(jax.random.key(1), (B, D))


def S_of(m, p):
    return m.apply(p, X, method=m.scale)


def test_abs_only_when_unsigned():
    m_off, p_off = enc(False)
    m_on, p_on = enc(True)
    S_off, S_on = S_of(m_off, p_off), S_of(m_on, p_on)
    assert S_off.shape == (B, H)
    assert float(S_off.min()) >= 0.0
    # not vacuous: the raw router straddles zero, which is why abs folds
    assert float(S_on.min()) < 0.0
    np.testing.assert_allclose(S_off, jnp.abs(S_on), rtol=1e-6)


@pytest.mark.parametrize("router_signed", [False, True])
def test_call_and_withclusts_agree(router_signed):
    """The regression: both must read `S` through `scale`."""
    m, p = enc(router_signed)
    y_call = m.apply(p, X)
    y_clust, _ = m.apply(p, jnp.zeros((B, D)), X, method=m.withClusts)
    np.testing.assert_allclose(y_call, y_clust, atol=1e-5)


@pytest.mark.parametrize("router_signed", [False, True])
def test_withstats_agrees_with_scale(router_signed):
    """`withStats` repeats the rule rather than calling `scale`, so it is
    the one place the two can drift apart."""
    m, p = enc(router_signed)
    y_clust, _ = m.apply(p, jnp.zeros((B, D)), X, method=m.withClusts)
    y_stats, _, _, _ = m.apply(p, jnp.zeros((B, D)), X, rng=jax.random.key(2),
                               method=m.withStats)
    np.testing.assert_allclose(y_clust, y_stats, atol=1e-5)


def test_signed_routing_changes_the_forward():
    m_off, p_off = enc(False)
    m_on, p_on = enc(True)
    y_off = m_off.apply(p_off, X)
    y_on = m_on.apply(p_on, X)
    assert not bool(jnp.allclose(y_off, y_on, atol=1e-6))


@pytest.mark.parametrize("gate", ["none", "sigmoid", "relu"])
def test_no_gate_makes_the_flag_inert(gate):
    """A non-negative gate does NOT make the abs a no-op. `NLinearBlock`
    computes `fn(gate(Ys[0]) * prod(Ys[1:]))`, so the gate bounds only the
    gate factor while the magnitude factor stays a signed linear term, and
    `S` straddles zero whatever the gate is."""
    m_on, p_on = enc(True, gate_router=gate)
    m_off, p_off = enc(False, gate_router=gate)
    assert float(S_of(m_on, p_on).min()) < 0.0
    assert float(S_of(m_off, p_off).min()) >= 0.0


def test_l1s_does_not_depend_on_the_sign_rule():
    """`L1_S` reads the gate factor, upstream of the abs, so the penalty is
    the same either way and the flag cannot be used to dodge it. The rest
    of the row may differ: the sign reaches the feature geometry."""
    stats = {}
    for sgn in (False, True):
        m, p = enc(sgn, gate_router="sigmoid")
        _, _, st, _ = m.apply(p, jnp.zeros((B, D)), X, rng=jax.random.key(2),
                              method=m.withStats)
        stats[sgn] = st
    # `withPWAK` documents L1_S as the last entry
    np.testing.assert_allclose(stats[False][-1], stats[True][-1], rtol=1e-6)
    assert not bool(jnp.allclose(stats[False], stats[True]))


@pytest.mark.parametrize("router_signed", [False, True])
def test_flag_reaches_dictenc_through_ontologizer(router_signed):
    model = Ontologizer(D, D, 16, K, H, 2, scaled=True, forward="resid",
                        router_signed=router_signed, dtype_str="float32")
    p = model.init(jax.random.key(0), jnp.zeros((B, D)))
    for i in range(model.l):
        S = model.apply(p, method=lambda m, i=i: m.dictencs[i].scale(X))
        assert (float(S.min()) < 0.0) is router_signed
    assert jnp.isfinite(model.apply(p, X)).all()


def test_unscaled_ignores_the_flag():
    """With `scaled=False` there is no router at all, so `S` is all ones."""
    for sgn in (False, True):
        m = DictEnc(D, D, K, H, n=2, scaled=False, router_signed=sgn,
                    dtype_str="float32")
        p = m.init(jax.random.key(0), jnp.zeros((B, D)))
        np.testing.assert_allclose(m.apply(p, X, method=m.scale),
                                   jnp.ones((B, H)))

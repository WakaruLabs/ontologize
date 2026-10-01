"""`DictBlock.signed`: dropping the `abs()` in `dicts()`.

Non-negativity binds through the Gram matrix: `<a, a'> >= 0` holds exactly
for non-negative vectors, so an unsigned dictionary is non-negatively
correlated by construction, and a head that selects one of `k` rows spends
part of its capacity on a direction all of them share. These tests pin the
parts of that which are properties of the constraint set rather than of a
trained model.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.layers.dictblock import ConcatDictBlock, DictBlock
from ontologize.ontologizer import Ontologizer


H, K, D, B = 4, 8, 64, 16


def block(signed, cls=DictBlock, **kw):
    m = cls(k=K, d=D, h=H, signed=signed, dtype_str="float32", **kw)
    p = m.init(jax.random.key(0), jnp.zeros((B, H, K)))
    return m, p


@pytest.mark.parametrize("cls", [DictBlock, ConcatDictBlock])
def test_dicts_is_raw_weights_under_signed(cls):
    m_off, p_off = block(False, cls)
    m_on, p_on = block(True, cls)
    W = p_on["params"]["weights"]
    np.testing.assert_allclose(m_on.apply(p_on, method=m_on.dicts), W)
    np.testing.assert_allclose(m_off.apply(p_off, method=m_off.dicts),
                               jnp.abs(p_off["params"]["weights"]))
    # not a vacuous distinction: the initializer is mixed-sign
    assert float(m_on.apply(p_on, method=m_on.dicts).min()) < 0.0
    assert float(m_off.apply(p_off, method=m_off.dicts).min()) >= 0.0


def test_norm_rows_composes_and_keeps_signs():
    m, p = block(True, norm_rows=True)
    A = m.apply(p, method=m.dicts)
    np.testing.assert_allclose(jnp.linalg.norm(A, axis=-1), jnp.ones((H, K)),
                               atol=1e-5)
    assert float(A.min()) < 0.0
    np.testing.assert_array_equal(jnp.sign(A), jnp.sign(p["params"]["weights"]))


def test_signed_removes_the_l1_floor():
    """`norm_rows`'s `|F|_1 >= h` bound rests on non-cancellation: rows are
    non-negative and each head's classification sums to 1. Signed rows can
    cancel across `k`, so the floor is no longer a bound at all."""
    m = DictBlock(k=K, d=D, h=H, norm_rows=True, signed=True,
                  dtype_str="float32")
    p = m.init(jax.random.key(0), jnp.zeros((B, H, K)))
    W = jnp.zeros((H, K, D)).at[:, 0, 0].set(1.0).at[:, 1, 0].set(-1.0)
    p = {"params": {**p["params"], "weights": W}}
    P = jnp.zeros((B, H, K)).at[:, :, 0].set(0.5).at[:, :, 1].set(0.5)
    F = m.apply(p, P, method=m.fwd)
    np.testing.assert_allclose(jnp.abs(F).sum(-1), jnp.zeros((B,)), atol=1e-6)


def test_non_negative_rows_cannot_be_decorrelated():
    """The mechanism. `<a, a'> >= 0` holds exactly for non-negative vectors,
    so an unsigned dictionary is non-negatively correlated by construction
    and a head's `k` rows must share a direction -- which says nothing
    about which row was selected, the only thing the head's choice
    expresses. Signed rows carry no such floor."""
    m_off, p_off = block(False)
    m_on, p_on = block(True)
    for m, p, signed in ((m_off, p_off, False), (m_on, p_on, True)):
        A = np.asarray(m.apply(p, method=m.dicts))
        A = A / np.linalg.norm(A, axis=-1, keepdims=True)
        G = np.einsum("hkd,hjd->hkj", A, A)
        off = G[:, ~np.eye(K, dtype=bool)]
        assert (float(off.min()) < 0.0) is signed
        # 2/pi is the chance cosine for independent abs-normal entries
        assert (abs(float(off.mean())) < 0.1) is signed


def test_abs_makes_each_weight_a_sign_gauge():
    """`abs` costs a per-entry sign degree of freedom: the loss is an even
    function of every weight, so `W` and any sign-flip of it are the same
    model with mirrored gradients. `signed` spends that freedom on the
    dictionary instead, which is the point of the flag."""
    P = jax.nn.softmax(jax.random.normal(jax.random.key(3), (B, H, K)), -1)
    flip = jnp.where(jax.random.bernoulli(jax.random.key(4), 0.5, (H, K, D)),
                     -1.0, 1.0)

    def loss_and_grad(signed, W):
        m, p = block(signed)
        p = {"params": {**p["params"], "weights": W}}

        def f(params):
            F = m.apply({"params": params}, P, method=m.fwd)
            return jnp.sum((F - jnp.ones((B, D))) ** 2)

        return f(p["params"]), jax.grad(f)(p["params"])["weights"]

    W = jax.random.normal(jax.random.key(5), (H, K, D))
    for signed, same in ((False, True), (True, False)):
        L, g = loss_and_grad(signed, W)
        L_f, g_f = loss_and_grad(signed, W * flip)
        assert bool(jnp.allclose(L, L_f, rtol=1e-5)) is same
        assert bool(jnp.allclose(g * flip, g_f, atol=1e-4)) is same


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("concat", [False, True])
def test_flag_reaches_dictblock_through_ontologizer(signed, concat):
    model = Ontologizer(8, 8, 16, K, H, 2, signed=signed, concat=concat,
                        forward="resid", dtype_str="float32")
    p = model.init(jax.random.key(0), jnp.zeros((B, 8)))
    for i in range(model.l):
        A = model.apply(p, method=lambda m, i=i: m.dictencs[i].dict.dicts())
        assert (float(A.min()) < 0.0) is signed
    Y = model.apply(p, jnp.ones((B, 8)))
    assert jnp.isfinite(Y).all()


def test_signed_changes_the_forward():
    def out(signed):
        model = Ontologizer(8, 8, 16, K, H, 2, signed=signed, select="ste",
                            forward="resid", dtype_str="float32")
        p = model.init(jax.random.key(0), jnp.zeros((B, 8)))
        return model.apply(p, jnp.ones((B, 8)))

    assert not bool(jnp.allclose(out(False), out(True)))

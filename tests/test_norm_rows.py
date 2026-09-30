"""`DictBlock.norm_rows`: unit-L2 dictionary rows and the L1 floor they imply.

The floor is the point of the flag. Rows are non-negative and each head's
classification sums to 1, so nothing cancels and
`|F|_1 = sum_hk P_hk |W_hk|_1 >= sum_h min_k |W_hk|_1 = h` once every row
has `|W_hk|_2 = 1`. An L1 penalty can therefore sparsify a tag's support
but cannot drive the code to zero.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.layers.dictblock import DictBlock
from ontologize.ontologizer import Ontologizer


H, K, D, B = 4, 8, 64, 16


def block(norm_rows):
    m = DictBlock(k=K, d=D, h=H, norm_rows=norm_rows, dtype_str="float32")
    p = m.init(jax.random.key(0), jnp.zeros((B, H, K)))
    return m, p


def test_rows_are_unit_norm_only_when_enabled():
    m_off, p_off = block(False)
    m_on, p_on = block(True)
    n_off = jnp.linalg.norm(m_off.apply(p_off, method=m_off.dicts), axis=-1)
    n_on = jnp.linalg.norm(m_on.apply(p_on, method=m_on.dicts), axis=-1)
    assert n_on.shape == (H, K)
    np.testing.assert_allclose(n_on, jnp.ones((H, K)), atol=1e-5)
    # the off case is only unit-norm by accident of the initializer
    assert float(n_off.max() - n_off.min()) > 1e-3


def test_dicts_stay_non_negative():
    m, p = block(True)
    assert float(m.apply(p, method=m.dicts).min()) >= 0.0


def test_l1_floor_holds_for_any_classification():
    """h is a lower bound on the layer's per-sample L1, whatever P is."""
    m, p = block(True)
    key = jax.random.key(1)
    for scale in (0.0, 1.0, 20.0):
        K_logits = jax.random.normal(key, (B, H, K)) * scale
        P = jax.nn.softmax(K_logits, -1)
        F = m.apply(p, P, method=m.fwd)
        assert float(jnp.abs(F).sum(-1).min()) >= H - 1e-3


def test_l1_floor_is_attained_by_one_hot_rows():
    """The bound is tight: a row concentrated on one coordinate has
    |W|_1 = |W|_2 = 1, so a dictionary of such rows sits exactly at h."""
    m = DictBlock(k=K, d=D, h=H, norm_rows=True, dtype_str="float32")
    p = m.init(jax.random.key(0), jnp.zeros((B, H, K)))
    onehot = jnp.zeros((H, K, D)).at[..., 0].set(1.0)
    p = {"params": {**p["params"], "weights": onehot}}
    P = jax.nn.softmax(jax.random.normal(jax.random.key(2), (B, H, K)), -1)
    F = m.apply(p, P, method=m.fwd)
    np.testing.assert_allclose(jnp.abs(F).sum(-1), jnp.full((B,), float(H)),
                               rtol=1e-5)


def test_radial_direction_carries_no_gradient():
    """Normalizing inside the forward pass makes each row's own direction an
    exact null direction, so nothing has to be projected out after the step."""
    m, p = block(True)
    P = jax.nn.softmax(jax.random.normal(jax.random.key(3), (B, H, K)), -1)

    def f(params):
        return jnp.sum(m.apply({"params": params}, P, method=m.fwd) ** 2)

    g = jax.grad(f)(p["params"])["weights"]
    W = p["params"]["weights"]
    # `dicts()` normalizes `A = abs(W)`, so the gradient is orthogonal to `A`
    # in `dL/dA` space; pulled back through `sign(W)` that is `<g, W>`.
    radial = jnp.sum(g * W, axis=-1)
    np.testing.assert_allclose(radial, jnp.zeros((H, K)), atol=1e-4)


@pytest.mark.parametrize("norm_rows", [False, True])
def test_flag_reaches_dictblock_through_ontologizer(norm_rows):
    model = Ontologizer(8, 8, 16, K, H, 2, norm_rows=norm_rows,
                        forward="resid", dtype_str="float32")
    p = model.init(jax.random.key(0), jnp.zeros((B, 8)))
    for i in range(model.l):
        n = jnp.linalg.norm(
            model.apply(p, method=lambda m, i=i: m.dictencs[i].dict.dicts()),
            axis=-1)
        assert n.shape == (H, K)
        assert bool(jnp.allclose(n, 1.0, atol=1e-5)) is norm_rows
    Y = model.apply(p, jnp.ones((B, 8)))
    assert jnp.isfinite(Y).all()

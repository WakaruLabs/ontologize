# bcossim_tags claims to be identical in value and gradient to the naive
# Sparse.bcossim(hfwd(P, S)) while computing in k x k tag space instead of
# materializing the flattened (h d) Gram. The bcossim regularizer weight
# rides on this equivalence.
import jax
import jax.numpy as jnp
import pytest

from ontologize.layers.dictblock import DictBlock

from conftest import DB_KW


@pytest.fixture(scope="module")
def db_loss(dictblock):
    # bcossim_loss=True so gradients flow through both formulations;
    # params carry over (the flag doesn't change the param tree)
    _, params, P = dictblock
    return DictBlock(**DB_KW, bcossim_loss=True), params, P


def ref(module, P, S=None):
    return module.bcossim(module.hfwd(P, S))


def fast(module, P, S=None):
    return module.bcossim_tags(P, S)


@pytest.mark.parametrize("with_scale", [False, True], ids=["plain", "scaled"])
def test_value_and_gradient_equivalence(db_loss, with_scale):
    db, params, P = db_loss
    S = None
    if with_scale:
        S = jax.random.uniform(jax.random.PRNGKey(5),
                               P.shape[:-1], minval=0.5, maxval=2.0)

    def f(method):
        return lambda p, P: db.apply(p, P, S, method=method)

    v_ref = f(ref)(params, P)
    v_fast = f(fast)(params, P)
    assert jnp.allclose(v_ref, v_fast, atol=1e-5)

    g_ref = jax.grad(f(ref), argnums=(0, 1))(params, P)
    g_fast = jax.grad(f(fast), argnums=(0, 1))(params, P)
    for a, b in zip(jax.tree_util.tree_leaves(g_ref),
                    jax.tree_util.tree_leaves(g_fast)):
        assert jnp.allclose(a, b, atol=1e-5)


def test_stat_gated_without_flag(dictblock):
    # with bcossim_loss=False the stat must carry no gradient
    db, params, P = dictblock
    g = jax.grad(lambda p, P: db.apply(p, P, None, method=fast),
                 argnums=(0, 1))(params, P)
    total = sum(float(jnp.abs(v).sum()) for v in jax.tree_util.tree_leaves(g))
    assert total == 0.0

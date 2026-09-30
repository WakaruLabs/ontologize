# Stagewise deep supervision (deepsup_sg): a gradient-only change. Prefix
# values must be identical to joint mode; under stagewise, each layer's
# gradient must equal the gradient of its own prefix term alone (pure
# RVQ/boosting credit assignment).
import jax
import jax.numpy as jnp
import pytest

from ontologize.ontologizer import Ontologizer

from conftest import KW


def fwd(model, params, X):
    Y, _, _ = model.apply(params, X, temperature=1.0,
                          method=Ontologizer.withStats)
    return Y


@pytest.fixture(scope="module")
def modes(build, X):
    m_joint, params = build(deepsup_sg=False)
    m_sg, _ = build(deepsup_sg=True)

    def loss_all(m):
        return lambda p: jnp.mean((fwd(m, p, X) - X[None]) ** 2)

    g_joint = jax.grad(loss_all(m_joint))(params)
    g_sg = jax.grad(loss_all(m_sg))(params)
    return m_joint, m_sg, params, g_joint, g_sg


def dictw(g, i):
    return g['params'][f'dictencs_{i}']['dict']['weights']


def test_forward_values_identical(modes, X):
    m_joint, m_sg, params, _, _ = modes
    Y_j = fwd(m_joint, params, X)
    Y_s = fwd(m_sg, params, X)
    assert Y_j.shape == (KW["l"], X.shape[0], KW["d_out"])
    assert jnp.allclose(Y_j, Y_s, atol=1e-6)


def test_last_layer_grads_identical(modes):
    # the last layer only appears in the last prefix in either mode
    _, _, _, g_joint, g_sg = modes
    diff = jnp.abs(dictw(g_joint, KW["l"] - 1) - dictw(g_sg, KW["l"] - 1))
    assert diff.max() < 1e-7


def test_earlier_layer_grads_differ(modes):
    # joint mode couples earlier layers to later prefix losses
    _, _, _, g_joint, g_sg = modes
    assert jnp.abs(dictw(g_joint, 0) - dictw(g_sg, 0)).max() > 1e-6


@pytest.mark.parametrize("layer", range(KW["l"]))
def test_stagewise_credit_is_layer_local(modes, X, layer):
    # under stagewise, layer i's gradient == gradient of prefix term i alone
    _, m_sg, params, _, g_sg = modes

    def loss_one(p):
        return jnp.mean((fwd(m_sg, p, X)[layer] - X) ** 2) / KW["l"]

    g_one = jax.grad(loss_one)(params)
    assert jnp.abs(dictw(g_sg, layer) - dictw(g_one, layer)).max() < 1e-7

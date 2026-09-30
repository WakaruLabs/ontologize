# Gain-shape separation at the DictEnc input (resid_gain): each layer
# classifies the unit-normalized direction of its input (the shape) and
# scales its residual contribution by the measured input norm (the gain).
# resid_norm alone discards that norm -- the classifier sees a unit vector,
# so the dictionary must bake in a corpus-average residual scale; resid_gain
# carries the magnitude through exactly (stagewise gain-shape VQ). The gain
# is stop-gradiented (measured, not trained), adds no parameters, and skips
# nextinput's normalization so the layer sees the raw magnitude.
import jax
import jax.numpy as jnp
import pytest

from conftest import KW, B


@pytest.fixture(scope="module")
def gained(build):
    return build(resid_gain=True, resid_const=True)


def test_requires_resid_forwarding(build):
    with pytest.raises(ValueError):
        build(forward="labels", resid_gain=True)


def test_no_new_params(build, gained):
    # no new parameters and no input-dim change relative to the resid_norm
    # conditioning it supersedes: old checkpoints load
    _, p_norm = build(resid_norm=True, resid_const=True, resid_gain=False)
    _, p_gain = gained
    norm_shapes = jax.tree_util.tree_map(jnp.shape, p_norm)
    gain_shapes = jax.tree_util.tree_map(jnp.shape, p_gain)
    assert norm_shapes == gain_shapes


def test_layer0_identity_on_unit_input(build, X):
    # layer 0's gain is ||X|| = 1 for unit-norm (SONAR-like) input, so a
    # single-layer gained model reproduces the baseline forward
    m0, p0 = build(l=1, resid_gain=False)
    m1, p1 = build(l=1, resid_gain=True)
    Y0 = m0.apply(p0, X)
    Y1 = m1.apply(p1, X)
    assert jnp.allclose(Y0, Y1, atol=1e-5)


def layer_rows(model, params, X):
    """(raw input, shaped input, gain) per layer along the withClusts path."""
    def probe(module, X):
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        rows = []
        E_in = module.constinput(E)
        for i, dictenc in enumerate(module.dictencs):
            U, G = dictenc.gainshape_in(E_in)
            rows.append((E_in, U, G))
            R, K = dictenc.withClusts(R, E_in, temperature=1.0)
            if i < module.l - 1:
                E_in = module.nextinput(X, R, K)
        return rows
    return model.apply(params, X, method=probe)


def test_shape_unit_gain_measured(gained, X):
    # upper layers: raw residual in, unit shape + measured norm out of the
    # split; the const coordinate is excluded from the norm and untouched
    model, params = gained
    rows = layer_rows(model, params, X)
    for Ein, U, G in rows[1:]:
        n = jnp.linalg.norm(Ein[..., :-1], axis=-1, keepdims=True)
        # nextinput skipped its normalization: the raw residual is not unit
        assert not jnp.allclose(n, 1.0, atol=1e-3)
        assert jnp.allclose(jnp.linalg.norm(U[..., :-1], axis=-1), 1.0,
                            atol=1e-5)
        assert jnp.all(U[..., -1] == 1.0)
        assert jnp.allclose(G, n, atol=1e-6)
        assert jnp.all(G > 0)


def test_contribution_homogeneous_in_input(build, gained, X):
    # the point of gain-shape: the layer's contribution is degree-1
    # homogeneous in its input (shape invariant, gain linear). The bilinear
    # baseline is not -- logits scale with ||input||^2, so the softmax
    # itself moves.
    def probe(module, X):
        d0 = module.dictencs[0]
        R = jnp.zeros(X.shape[:-1] + (module.e_dec,), X.dtype)
        Y1, _ = d0.withClusts(R, module.constinput(X), temperature=1.0)
        Y2, _ = d0.withClusts(R, module.constinput(2.0 * X), temperature=1.0)
        return Y1, Y2

    model, params = gained
    Y1, Y2 = model.apply(params, X, method=probe)
    assert jnp.allclose(Y2, 2.0 * Y1, atol=1e-4)

    m0, p0 = build(resid_gain=False)
    Y1, Y2 = m0.apply(p0, X, method=probe)
    assert not jnp.allclose(Y2, 2.0 * Y1, atol=1e-2)


def test_gain_carries_no_gradient(gained, X):
    # the gain is measured, not trained: zero gradient through it
    model, params = gained

    def f(X):
        def probe(module, X):
            _, G = module.dictencs[0].gainshape_in(module.constinput(X))
            return G.sum()
        return model.apply(params, X, method=probe)

    g = jax.grad(f)(X)
    assert jnp.all(g == 0.0)


def test_zero_input_no_nan(gained, X):
    # a perfectly reconstructed residual gives a zero layer input: the
    # eps-normalized zero direction must be killed by the zero gain, not
    # reach the residual accumulator (or produce NaNs)
    model, params = gained

    def probe(module, X):
        d0 = module.dictencs[0]
        R = jnp.ones(X.shape[:-1] + (module.e_dec,), X.dtype)
        Y, _ = d0.withClusts(R, module.constinput(jnp.zeros_like(X)),
                             temperature=1.0)
        return Y

    Y = model.apply(params, X, method=probe)
    assert jnp.all(jnp.isfinite(Y))
    assert jnp.allclose(Y, 1.0)  # contribution is exactly zero


def test_training_gradients_finite(gained, X):
    # full deepsup withStats forward: finite loss, finite nonzero grads
    model, params = gained

    def loss(p):
        Y, stats, _ = model.apply(p, X, method="withStats")
        return jnp.mean((Y - X) ** 2)

    L, g = jax.value_and_grad(loss)(params)
    assert jnp.isfinite(L)
    leaves = jax.tree_util.tree_leaves(g)
    assert all(jnp.all(jnp.isfinite(v)) for v in leaves)
    assert sum(float(jnp.abs(v).sum()) for v in leaves) > 0.0


def test_ghost_shapes_and_finite(X):
    # ghost path under gain scaling: G is a stop-gradiented constant, so
    # both accumulators take the same rescale and shapes still pair up
    from ontologize.ontologizer import Ontologizer
    model = Ontologizer(**dict(KW, gate="relu", activation_dec="relu",
                               resid_gain=True, resid_const=True))
    params = model.init(jax.random.PRNGKey(0), X)
    Y, Y_g, stats, _ = model.apply(params, X, method="withGhost")
    assert Y.shape == (KW["l"], B, KW["d_out"])
    assert Y_g.shape == Y.shape
    assert jnp.all(jnp.isfinite(Y))
    assert jnp.all(jnp.isfinite(Y_g))


def test_withargs_interventions_run(gained, X):
    # intervention path picks the gain up through the same loop
    from ontologize.ontologizer import DictIntervention
    model, params = gained
    arglist = [DictIntervention() for _ in range(KW["l"])]
    arglist[1] = DictIntervention(h_set=jnp.array([0]), k_set=jnp.array([3]))
    Y, K, stats, _ = model.apply(params, X, arglist, method="withArgs")
    assert Y.shape == (B, KW["d_out"])
    assert jnp.all(jnp.isfinite(Y))

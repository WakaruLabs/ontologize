# Residual input conditioning (resid_norm + resid_const): fixes for the
# two bilinear-classifier failure modes on raw residuals -- logits scale
# with ||input||^2 (vanishing gradients on small residuals) and unbiased
# bilinear logits are even functions (blind to residual sign).
import jax
import jax.numpy as jnp
import pytest

from conftest import KW


@pytest.fixture(scope="module")
def conditioned(build):
    return build(resid_norm=True, resid_const=True)


def layer_rows(model, params, X):
    """(input, logit std) per layer along the withClusts path."""
    def probe(module, X):
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        rows = []
        Ein = E
        for i, dictenc in enumerate(module.dictencs):
            K = dictenc.classifier(Ein)
            rows.append((Ein, jnp.std(K, axis=-1).mean()))
            R, Kc = dictenc.withClusts(R, Ein, temperature=1.0)
            if i < module.l - 1:
                Ein = module.nextinput(X, R, Kc)
        return rows
    return model.apply(params, X, method=probe)


def test_upper_classifier_input_dim(conditioned):
    # resid_const widens upper-layer classifier input by the constant coord
    _, params = conditioned
    w1 = params['params']['dictencs_1']['classifier']['weight']
    assert w1.shape == (KW["n"], KW["h"], KW["k"], KW["d_out"] + 1)


def test_upper_input_normalized_with_const_coord(conditioned, X):
    model, params = conditioned
    for Ein, _ in layer_rows(model, params, X)[1:]:
        assert jnp.allclose(jnp.linalg.norm(Ein[..., :-1], axis=-1), 1.0,
                            atol=1e-5)
        assert jnp.all(Ein[..., -1] == 1.0)


def test_init_logit_spread_not_starved(conditioned, X):
    # the unconditioned resid run had upper/lower logit-spread ratio ~0.002
    # at init (residual norm ~0.25, squared by the bilinear form): flat
    # softmax, vanishing gradients, weights never left init
    model, params = conditioned
    rows = layer_rows(model, params, X)
    s0 = float(rows[0][1])
    for i, (_, s) in enumerate(rows[1:], 1):
        assert float(s) / s0 > 0.2, f"layer {i} logit spread starved"


def test_residual_input_carries_no_gradient(conditioned, X):
    # the residual handed to the next layer is stop-gradiented: lower
    # layers must not be able to write a communication code into it
    # instead of reducing it. A function of layer 1's input must have zero
    # gradient w.r.t. every parameter.
    model, params = conditioned

    def f(p):
        def probe(module, X):
            E, _ = module.encode(X, 0.0, None)
            R = module.resid(E)
            R, K = module.dictencs[0].withClusts(R, E, temperature=1.0)
            return (module.nextinput(X, R, K) ** 2).sum()
        return model.apply(p, X, method=probe)

    g = jax.grad(f)(params)
    total = sum(float(jnp.abs(v).sum()) for v in jax.tree_util.tree_leaves(g))
    assert total == 0.0


def test_sign_sensitivity(conditioned, X):
    # without the constant coordinate the pure bilinear form is provably
    # even: K(+d) == K(-d) exactly. The constant coordinate adds linear
    # terms that break it.
    model, params = conditioned
    E1 = layer_rows(model, params, X)[1][0]
    D = E1[..., :-1]
    Dpm = jnp.concatenate([D, -D])
    Dpm = jnp.concatenate([Dpm, jnp.ones(Dpm.shape[:-1] + (1,))], -1)

    def upper_logits(module, D):
        return module.dictencs[1].classifier(D)

    K = model.apply(params, Dpm, method=upper_logits)
    b = X.shape[0]
    assert jnp.abs(K[:b] - K[b:]).max() > 1e-3

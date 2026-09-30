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
    # resid_gain pinned off: this file tests the nextinput-side
    # conditioning it supersedes (gain-shape moves the normalization
    # inside the DictEnc; that path is covered by test_resid_gain.py)
    return build(resid_norm=True, resid_const=True, resid_gain=False)


def layer_rows(model, params, X):
    """(input, logit std) per layer along the withClusts path."""
    def probe(module, X):
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        rows = []
        E_in = module.constinput(E)
        for i, dictenc in enumerate(module.dictencs):
            K = dictenc.classifier(E_in)
            rows.append((E_in, jnp.std(K, axis=-1).mean()))
            R, Kc = dictenc.withClusts(R, E_in, temperature=1.0)
            if i < module.l - 1:
                E_in = module.nextinput(X, R, Kc)
        return rows
    return model.apply(params, X, method=probe)


def test_classifier_input_dims(conditioned):
    # resid_const widens every classifier input by the constant coord:
    # layer 0's encoder output and the upper layers' residual alike
    _, params = conditioned
    w0 = params['params']['dictencs_0']['classifier']['weight']
    assert w0.shape == (KW["n"], KW["h"], KW["k"], KW["d_in"] + 1)
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
            R, K = module.dictencs[0].withClusts(
                    R, module.constinput(E), temperature=1.0)
            return (module.nextinput(X, R, K) ** 2).sum()
        return model.apply(p, X, method=probe)

    g = jax.grad(f)(params)
    total = sum(float(jnp.abs(v).sum()) for v in jax.tree_util.tree_leaves(g))
    assert total == 0.0


def test_layer0_sign_sensitivity(conditioned, X):
    # layer 0's classifier gets the constant coordinate too: without it
    # the pure bilinear form is even in the encoder output, and the
    # anisotropic input cone can only emulate linear terms through
    # cross-terms with the corpus mean
    model, params = conditioned

    def logits(module, X):
        E, _ = module.encode(X, 0.0, None)
        return module.dictencs[0].classifier(module.constinput(E))

    Xpm = jnp.concatenate([X, -X])
    K = model.apply(params, Xpm, method=logits)
    b = X.shape[0]
    assert jnp.abs(K[:b] - K[b:]).max() > 1e-3


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

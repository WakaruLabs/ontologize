# forward="resid_labels": upper layers read [residual | const | code].
# The code rides ALONGSIDE the residual direction rather than replacing
# it, so the gain-shape split must still measure the residual alone and
# the pass-through tail must widen to cover the code.
import jax
import jax.numpy as jnp
import pytest

from ontologize.ontologizer import Ontologizer

from conftest import KW, B

HK = KW["h"] * KW["k"]


@pytest.fixture(scope="module")
def both(build):
    return build(forward="resid_labels", resid_const=True)


def cl_width(params, i):
    """Realized classifier input width of layer i, (n, h, k, d_in)."""
    return params["params"][f"dictencs_{i}"]["classifier"]["weight"].shape[-1]


def test_layer_widths(both):
    _, params = both
    # layer 0 has no previous code: encoder output plus the constant
    assert cl_width(params, 0) == KW["d_in"] + 1
    for i in range(1, KW["l"]):
        assert cl_width(params, i) == KW["d_out"] + 1 + HK


def test_nextinput_layout_and_gain(both, X):
    """The block order is [residual | const | code]; the split normalizes
    the residual only and the gain is its norm."""
    model, params = both

    def probe(module, X):
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        de = module.dictencs[0]
        U, G = de.gainshape_in(module.constinput(E))
        K = de.dict.cluster(de.classifier(U))
        R = R + de.gained(de.dict.combine(de.dict.hfwd(K)), G)
        Kf = K.reshape(X.shape[0], -1)
        E_in = module.nextinput(X, R, Kf)
        U1, G1 = module.dictencs[1].gainshape_in(E_in)
        return E_in, Kf, X - module.decode(R), U1, G1

    E_in, Kf, D, U1, G1 = model.apply(params, X, method=probe)
    assert E_in.shape == (B, KW["d_out"] + 1 + HK)
    assert jnp.allclose(E_in[:, :KW["d_out"]], D, atol=1e-5)   # residual
    assert jnp.allclose(E_in[:, KW["d_out"]], 1.0)             # const
    assert jnp.allclose(E_in[:, KW["d_out"] + 1:], Kf, atol=1e-6)   # code
    # the gain is the residual's norm, not the concatenation's
    assert jnp.allclose(G1[:, 0], jnp.linalg.norm(D, axis=-1), rtol=1e-4)
    # the code and constant blocks pass through the split untouched
    assert jnp.allclose(U1[:, KW["d_out"]:], E_in[:, KW["d_out"]:], atol=1e-6)
    assert jnp.allclose(jnp.linalg.norm(U1[:, :KW["d_out"]], axis=-1), 1.0,
                        atol=1e-4)


def test_code_block_is_stop_gradiented(both, X):
    """Both forwarded blocks are cut: the residual already was, and the
    code must be too, or a lower layer can write a communication channel
    into its classification instead of reducing the residual."""
    model, params = both

    def probe(module, X, K):
        R = module.resid(module.encode(X, 0.0, None)[0])
        return (module.nextinput(X, R, K) ** 2).sum()

    K = jnp.zeros((B, HK)) + 0.1
    g = jax.grad(lambda K: model.apply(params, X, K, method=probe))(K)
    assert jnp.all(g == 0.0)


def test_missing_code_is_a_loud_error(both, X):
    model, params = both

    def probe(module, X):
        R = module.resid(module.encode(X, 0.0, None)[0])
        return module.nextinput(X, R, None)

    with pytest.raises(ValueError, match="resid_labels"):
        model.apply(params, X, method=probe)


def test_forward_and_stats_finite(both, X):
    model, params = both
    Y, stats, _ = model.apply(params, X, temperature=0.5,
                              method=Ontologizer.withStats)
    assert Y.shape == (KW["l"], B, KW["d_out"])   # deepsup prefixes
    assert jnp.all(jnp.isfinite(Y)) and jnp.all(jnp.isfinite(stats))


def test_resid_mode_unchanged(build, X):
    """The added branch must not move the plain residual mode."""
    m0, p0 = build(resid_const=True)
    Y0, _, _ = m0.apply(p0, X, temperature=0.5, method=Ontologizer.withStats)
    assert cl_width(p0, 1) == KW["d_out"] + 1
    assert jnp.all(jnp.isfinite(Y0))

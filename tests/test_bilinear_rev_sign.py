# `Bilinear.rev` / `BilinearBlock.rev` read the top eigenvector of each
# output's bilinear form. `eigh` fixes eigenvectors only up to sign, so
# without a convention the returned direction's sign is a property of
# LAPACK rather than of the layer. `orient` settles it, and the tests
# below also pin what that fix does NOT buy: the sign a steering caller
# wants is sample-dependent and no weight-only reverse can supply it.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.layers.nlinear import Bilinear, BilinearBlock, orient

D_IN, D_OUT, H = 6, 4, 3
KW = dict(dtype_str="float32", dtype_p_str="float32")


def test_orient_is_sign_invariant():
    """The whole point: negating the eigenvector must not change the
    answer, since eigh's choice between v and -v is arbitrary."""
    V = jax.random.normal(jax.random.PRNGKey(0), (5, D_IN))
    lam = jax.random.normal(jax.random.PRNGKey(1), (5, 1))
    assert jnp.allclose(orient(V, lam), orient(-V, lam))
    # and the eigenvalue's sign still flips it, since that says whether
    # the form grows or shrinks along the vector
    assert jnp.allclose(orient(V, lam), -orient(V, -lam))
    assert jnp.allclose(jnp.linalg.norm(orient(V, lam), axis=-1),
                        jnp.linalg.norm(V, axis=-1))


def planted_e():
    e = np.zeros((D_OUT, D_IN), np.float32)
    e[np.arange(D_OUT), np.arange(D_OUT)] = 1.0
    e[:, -1] = 0.3
    return e / np.linalg.norm(e, axis=-1, keepdims=True)


def rev_of_planted(e):
    """`rev` of a layer whose form for output k is the rank-1 e_k e_k^T
    (W = V = e), so the top eigenvector is +-e_k with eigenvalue 1."""
    lay = Bilinear(d_in=D_IN, d_out=D_OUT, biased=False, **KW)
    p = lay.init(jax.random.PRNGKey(0), jnp.zeros((1, D_IN)))
    p = dict(params=dict(p["params"],
                         weight=jnp.stack([jnp.asarray(e)] * 2)))
    return np.asarray(lay.apply(p, jnp.eye(D_OUT), method=Bilinear.rev))


def test_rev_recovers_the_planted_directions():
    e = planted_e()
    out = rev_of_planted(e)
    out = out / np.linalg.norm(out, axis=-1, keepdims=True)
    assert np.allclose(np.abs(out), np.abs(e), atol=1e-4)


def test_rev_ignores_the_eigenvector_sign_convention():
    """e and -e give the SAME bilinear form, so `rev` must not
    distinguish them. Without a convention it can, since eigh's choice
    between v and -v is free."""
    e = planted_e()
    assert np.allclose(rev_of_planted(e), rev_of_planted(-e), atol=1e-4)


def test_block_rev_shape_and_finiteness():
    lay = BilinearBlock(d_in=D_IN, d_out=D_OUT, h=H, biased=False, **KW)
    X = jax.random.normal(jax.random.PRNGKey(2), (5, D_IN))
    p = lay.init(jax.random.PRNGKey(3), X)
    out = lay.apply(p, jnp.ones((5, H, D_OUT)), method=BilinearBlock.rev)
    assert out.shape == (5, D_IN) and jnp.all(jnp.isfinite(out))


def test_steering_sign_is_sample_dependent():
    """Why the convention cannot fix steering: moving from x along v
    changes x'Bx by 2e*lam*(x.v), so which way to move depends on the
    sample. Over random samples both signs occur about equally."""
    lay = Bilinear(d_in=D_IN, d_out=D_OUT, biased=False, **KW)
    X = jax.random.normal(jax.random.PRNGKey(4), (256, D_IN))
    p = lay.init(jax.random.PRNGKey(5), X)
    v = lay.apply(p, jnp.eye(D_OUT)[0][None], method=Bilinear.rev)[0]
    v = v / jnp.linalg.norm(v)
    base = lay.apply(p, X, method=Bilinear.fwd)[:, 0]
    up = lay.apply(p, X + 0.01 * v, method=Bilinear.fwd)[:, 0]
    rose = np.asarray(up > base)
    assert 0.2 < rose.mean() < 0.8, (
        "a single weight-only direction should raise the logit on only "
        f"some samples; got {rose.mean():.2f}")

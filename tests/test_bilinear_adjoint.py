# DictEnc now instantiates BilinearBlock (not NLinearBlock) when n=2, which
# provides the eigendecomposition-based classifier adjoint `rev` that
# Ontologizer.intervene / DictEnc.rev need for input-space steering on the
# bilinear runs. Checkpoint compatibility rides on BilinearBlock being
# param-tree- and forward-identical to the NLinearBlock(n=2) the trained
# runs were built with.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.layers.nlinear import (NLinear, Bilinear,
                                       NLinearBlock, BilinearBlock)
from ontologize.ontologizer import Ontologizer
from ontologize.layers.dictenc import DictEnc

H, K, D, B = 4, 8, 16, 12
BLK_KW = dict(d_in=D, d_out=K, h=H, n=2, biased=False, gate="none",
              dtype_str="float32", dtype_p_str="float32")


@pytest.fixture(scope="module")
def block():
    blk = BilinearBlock(**BLK_KW)
    X = jax.random.normal(jax.random.PRNGKey(0), (B, D))
    params = blk.init(jax.random.PRNGKey(1), X)
    return blk, params, X


def test_param_tree_and_forward_match_nlinearblock(block):
    # the trained checkpoints were built through NLinearBlock(n=2); the
    # swap must not change the param tree or the forward values
    blk, params, X = block
    old = NLinearBlock(**BLK_KW)
    params_old = old.init(jax.random.PRNGKey(1), X)
    assert (jax.tree_util.tree_structure(params)
            == jax.tree_util.tree_structure(params_old))
    for a, b in zip(jax.tree_util.tree_leaves(params),
                    jax.tree_util.tree_leaves(params_old)):
        assert jnp.array_equal(a, b)
    assert jnp.array_equal(blk.apply(params, X), old.apply(params, X))


def test_dictenc_uses_bilinear_block():
    X = jnp.zeros((2, D))
    de = DictEnc(D, D, K, H, n=2, gate="none",
                 dtype_str="float32", dtype_p_str="float32")
    params = de.init(jax.random.PRNGKey(0), X)
    assert isinstance(de.bind(params).classifier, BilinearBlock)
    de1 = DictEnc(D, D, K, H, n=1, gate="relu",
                  dtype_str="float32", dtype_p_str="float32")
    params1 = de1.init(jax.random.PRNGKey(0), X)
    cl1 = de1.bind(params1).classifier
    assert isinstance(cl1, NLinearBlock) and not isinstance(cl1, BilinearBlock)


def test_rev_uses_genuine_top_eigenvectors(block):
    # for each (head, entry): the vector rev maps that logit to must be a
    # unit eigenvector of its bilinear tensor with the largest-|.| eigenvalue
    blk, params, _ = block
    Bt = blk.apply(params, method=BilinearBlock.bilinearTensor)
    vals, vecs = blk.apply(params, method=BilinearBlock.decompose)
    for h in range(H):
        for k in range(K):
            onehot = jnp.zeros((1, H, K)).at[0, h, k].set(1.0)
            v = blk.apply(params, onehot, method=BilinearBlock.rev)[0]
            lam = vals[h, k, jnp.argmax(jnp.abs(vals[h, k]))]
            assert jnp.allclose(jnp.linalg.norm(v), 1.0, atol=1e-5)
            assert jnp.allclose(Bt[h, k] @ v, lam * v, atol=1e-5)
            assert jnp.allclose(jnp.abs(v @ Bt[h, k] @ v),
                                jnp.abs(vals[h, k]).max(), atol=1e-5)


def test_rev_is_linear(block):
    blk, params, _ = block
    Y1 = jax.random.normal(jax.random.PRNGKey(2), (B, H, K))
    Y2 = jax.random.normal(jax.random.PRNGKey(3), (B, H, K))

    def rev(Y):
        return blk.apply(params, Y, method=BilinearBlock.rev)

    assert jnp.array_equal(rev(jnp.zeros((B, H, K))), jnp.zeros((B, D)))
    assert jnp.allclose(rev(2.0 * Y1 - 3.0 * Y2),
                        2.0 * rev(Y1) - 3.0 * rev(Y2), atol=1e-5)


def test_block_rev_matches_per_head_bilinear(block):
    # the multihead rev must be the sum over heads of the single-head
    # Bilinear.rev on that head's weight slice
    blk, params, _ = block
    W = params['params']['weight']  # (2, h, k, d)
    Y = jax.random.normal(jax.random.PRNGKey(4), (B, H, K))
    out = blk.apply(params, Y, method=BilinearBlock.rev)

    single = Bilinear(d_in=D, d_out=K, n=2, biased=False, gate="none",
                      dtype_str="float32", dtype_p_str="float32")
    manual = np.zeros((B, D), np.float32)
    for h in range(H):
        p_h = {'params': {'weight': W[:, h]}}
        manual += np.asarray(
            single.apply(p_h, Y[:, h], method=Bilinear.rev))
    assert jnp.allclose(out, manual, atol=1e-5)


def test_scaled_bilinear_forward_smoke(X, build):
    # n=2 with scaled=True now routes the scaling layer through Bilinear;
    # the plain forward (scale enters through hfwd's einsum) must run finite
    model, params = build(scaled=True, resid_norm=True, resid_const=True)
    Y = model.apply(params, X, temperature=0.5)
    assert jnp.all(jnp.isfinite(Y))


def test_scaled_stats_path(X, build):
    # regression: this used to crash on cossim(Fs, S) -- the (b, h, h)
    # head-similarity matrix cannot broadcast against the (b, h) scale, and
    # per-head cosine is invariant to the positive scale anyway, so S was
    # dropped from the cossim_h call
    model, params = build(scaled=True, resid_norm=True, resid_const=True)
    Y, stats, _ = model.apply(params, X, temperature=0.5,
                              method=Ontologizer.withStats)
    assert jnp.all(jnp.isfinite(Y))
    assert stats.shape == (model.l, 12)
    assert jnp.all(jnp.isfinite(stats))

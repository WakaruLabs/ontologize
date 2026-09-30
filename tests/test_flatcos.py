# `DictBlock.flatcos`: mean off-diagonal cosine over the dictionary
# flattened, every row pair rather than only pairs inside a head. The
# complement of `rowcos`, which is within-head by construction and so
# cannot see heads whose rows duplicate another head's.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.layers.dictblock import DictBlock
from ontologize.ontologizer import Ontologizer
from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import update

from conftest import KW, B, DB_KW


def explicit(U):
    """Mean off-diagonal cosine via the full Gram, which is what the
    O(N d) identity in `flatcos` has to reproduce."""
    U = U / np.linalg.norm(U, axis=-1, keepdims=True)
    C = U @ U.T
    return C[~np.eye(len(U), dtype=bool)].mean()


def test_matches_the_explicit_gram(dictblock):
    db, params, _ = dictblock
    got = float(db.apply(params, method=DictBlock.flatcos))
    W = np.asarray(db.apply(params, method=DictBlock.dicts))
    assert abs(got - explicit(W.reshape(-1, W.shape[-1]))) < 1e-5


def test_counts_between_head_pairs_that_rowcos_cannot_see():
    """Two dictionaries with IDENTICAL per-head statistics and opposite
    between-head structure. Every head holds distinct one-hot rows, so
    every within-head cosine is 0 in both; the heads either occupy
    disjoint coordinates or all occupy the same ones."""
    h, k = 4, 4
    db = DictBlock(k=k, d=h * k, h=h, dtype_str="float32",
                   dtype_p_str="float32")
    eye = np.eye(h * k, dtype=np.float32)
    disjoint = eye.reshape(h, k, h * k)                  # head i owns its own
    shared = np.broadcast_to(eye[:k], (h, k, h * k)).copy()   # all the same

    def stats(W):
        p = {"params": {"weights": jnp.asarray(W)}}
        return (np.asarray(db.apply(p, method=DictBlock.rowcos)),
                float(db.apply(p, method=DictBlock.flatcos)))

    r_dis, f_dis = stats(disjoint)
    r_sha, f_sha = stats(shared)
    assert np.allclose(r_dis, 0.0, atol=1e-6)
    assert np.allclose(r_sha, 0.0, atol=1e-6)       # rowcos cannot tell them apart
    assert f_dis == pytest.approx(0.0, abs=1e-6)
    # each row matches its h-1 namesakes out of h*k-1 partners
    assert f_sha == pytest.approx((h - 1) / (h * k - 1), abs=1e-6)


def test_collapse_saturates_rowcos_but_barely_moves_flatcos(dictblock):
    """The converse, and why neither statistic replaces the other: only
    k-1 of a row's h*k-1 partners share its head."""
    db, params, _ = dictblock
    W = np.asarray(db.apply(params, method=DictBlock.dicts))
    dead = W.copy()
    dead[0] = W[0, :1]                    # every entry of head 0 identical
    p0 = {"params": {"weights": jnp.asarray(W)}}
    p1 = {"params": {"weights": jnp.asarray(dead)}}
    assert float(np.max(db.apply(p1, method=DictBlock.rowcos))) > 0.99
    f0 = float(db.apply(p0, method=DictBlock.flatcos))
    f1 = float(db.apply(p1, method=DictBlock.flatcos))
    assert abs(f1 - f0) < 0.2 * max(f0, 1e-9)


def test_gradient_gate(build, X):
    """Logged either way; differentiable only when weighted."""
    def summed(model, params):
        return lambda p: model.apply(
            p, X, temperature=0.5, method=Ontologizer.withStats)[1][:, 10].sum()

    off, p_off = build(flatcos_loss=False)
    g = jax.grad(summed(off, p_off))(p_off)
    assert sum(float(jnp.abs(v).sum())
               for v in jax.tree_util.tree_leaves(g)) == 0.0
    on, p_on = build(flatcos_loss=True)
    g = jax.grad(summed(on, p_on))(p_on)
    assert sum(float(jnp.abs(v).sum())
               for v in jax.tree_util.tree_leaves(g)) > 0.0


def test_row_position_is_last(X):
    """The weight vector pairs against the row positionally, so the new
    stat must sit at the end of both the per-layer and the logged row."""
    d = KW["d_in"]
    s_flat = 1e-3
    hyper = Hyperparams(d, d, B, s_flatcos=s_flat, ghost=False,
                        noise_K="normal", noise_F="featvar")
    model = hyper.ontologizer(
        d, d, KW["e_dec"], KW["k"], KW["h"], KW["l"], n=2, gate="none",
        select="ste", forward="resid", deepsup=True, resid_const=True,
        dtype_str="float32", dtype_p_str="float32")
    assert model.flatcos_loss is True
    state = hyper.init(model, save_each=10)
    assert state.stats.shape == (10, 18)
    params = state.params
    state, L, _ = update(state, hyper.loss, jax.random.PRNGKey(2), X, X,
                         temperature=0.5, sd_in=0.0, sd_K=0.01, sd_F=0.01,
                         p_drop=0.1, grad_clip=1.0)
    row = np.asarray(state.stats[0])
    # column 16 is the layer sum of the per-layer entry, both last
    direct = 0.0
    for li in range(KW["l"]):
        W = np.asarray(model.apply(
            params, method=lambda m: m.dictencs[li].dict.dicts()))
        direct += explicit(W.reshape(-1, W.shape[-1]))
    assert abs(row[16] - direct) < 1e-3
    # the pwak columns keep their places, so older rows still align
    assert row[11] == 0.0 and row[12] == 0.0
    assert abs(float(L) - row[0]) < 1e-6

# anatomy.py's blocks pinned against the code paths they claim to draw:
# the kernel against pwak.diffuse_kl, the cosine matrices against
# DictBlock.rowcos / support_overlap, plus the shared ordering and a
# figure drawn from synthetic inputs.
import jax
import jax.numpy as jnp
import numpy as np

import anatomy
from ontologize.fns.pwak import diffuse_kl
from ontologize.layers.dictblock import DictBlock


def _rand(n=24, h=3, k=5, d=8, seed=0):
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(n, h, k)) * 2
    P = np.exp(L) / np.exp(L).sum(-1, keepdims=True)
    E = rng.normal(size=(n, d))
    return P.astype(np.float32), E.astype(np.float32)


def test_kernel_is_diffuse_kl_graph():
    P, E = _rand()
    for hd in range(P.shape[1]):
        G = anatomy.pwak_kernel(P, E, 0.2, hd)
        assert np.allclose(np.diag(G), 0)
        assert np.allclose(G.sum(1), 1, atol=1e-5)
        # one diffusion step with this graph is diffuse_kl at s=1
        Q = G @ P[:, hd]
        Q = Q / Q.sum(-1, keepdims=True)
        ref = np.asarray(diffuse_kl(jnp.asarray(P), jnp.asarray(E), 1, 0.2))
        assert np.allclose(Q, ref[:, hd], atol=1e-5)


def test_kernel_hard_code_is_block_diagonal():
    P = np.zeros((6, 1, 2), np.float32)
    P[:3, 0, 0] = 1
    P[3:, 0, 1] = 1
    E = np.random.default_rng(1).normal(size=(6, 4)).astype(np.float32)
    G = anatomy.pwak_kernel(P, E, 0.2, 0)
    assert np.allclose(G[:3, 3:], 0) and np.allclose(G[3:, :3], 0)


def _dictblock(h=3, k=5, d=8, **kw):
    db = DictBlock(k=k, d=d, h=h, dtype_str="float32", **kw)
    params = db.init(jax.random.PRNGKey(0), jnp.zeros((2, h, k)))
    W = np.asarray(db.apply(params, method=DictBlock.dicts))
    return db, params, W


def test_rowcos_matrix_reduces_to_cossim_k():
    db, params, W = _dictblock()
    ref = np.asarray(db.apply(params, method=DictBlock.rowcos))
    C = anatomy.rowcos_matrix(W)
    assert np.allclose(np.diagonal(C, axis1=1, axis2=2), 1)
    assert np.allclose(anatomy.offdiag_mean(C), ref, atol=1e-5)


def test_rowcos_matrix_signed_can_be_negative():
    db, params, W = _dictblock(signed=True)
    ref = np.asarray(db.apply(params, method=DictBlock.rowcos))
    C = anatomy.rowcos_matrix(W)
    assert C.min() < 0
    assert np.allclose(anatomy.offdiag_mean(C), ref, atol=1e-5)


def test_support_matrix_reduces_to_support():
    P, _ = _rand()
    for signed in (False, True):
        db, params, W = _dictblock(signed=signed)
        ref = float(db.apply(params, jnp.asarray(P),
                             method=DictBlock.support_overlap))
        S = anatomy.support_matrix(P, W)
        assert np.allclose(np.diag(S), 1)
        assert np.isclose(anatomy.offdiag_mean(S), ref, atol=1e-5)


def test_anatomy_order():
    Ph = np.array([[.1, .8, .1],   # entry 1
                   [.7, .2, .1],   # entry 0
                   [.2, .6, .2],   # entry 1, weaker
                   [.1, .9, .0],   # entry 1, strongest
                   [.1, .1, .8]])  # entry 2
    rorder, eorder = anatomy.anatomy_order(Ph)
    assert eorder.tolist() == [1, 0, 2]  # by mean probability
    assert rorder.tolist() == [3, 0, 2, 1, 4]


def test_top_coords_and_clip():
    assert anatomy.top_coords(np.array([.1, .5, .3, .5]), 3).tolist() \
        == [1, 3, 2]
    assert anatomy.clip_top(np.zeros((3, 3))) == 1.0
    A = np.array([0, 1, 2, 3, 100.0])
    assert anatomy.clip_top(A, 50) == 2.5


def test_draw_synthetic(tmp_path):
    P, E = _rand(n=40, h=4, k=6, d=10)
    _, _, W = _dictblock(h=4, k=6, d=10)
    z = dict(P=P, E=E, W=W, step=1, temperature=1.0, tau=0.2,
             concat=False, signed=False, run="synthetic", layer=0, head=2)
    anatomy.draw(z, 5, tmp_path / "a.pdf", tmp_path / "a.png")
    assert (tmp_path / "a.pdf").stat().st_size > 0
    z["concat"] = True
    anatomy.draw(z, 5, tmp_path / "b.pdf", tmp_path / "b.png")
    assert (tmp_path / "b.png").stat().st_size > 0

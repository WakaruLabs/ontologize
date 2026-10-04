# Fibers (a per-entry local basis added to the selected entry) and head
# dropout (heads replaced by their batch-mean classification in training).
import jax
import jax.numpy as jnp
import pytest

from conftest import DB_KW, KW
from ontologize.layers.dictblock import DictBlock, ConcatDictBlock
from ontologize.layers.dictenc import DictEnc
from ontologize.ontologizer import Ontologizer

COND = dict(resid_norm=True, resid_const=True)


def block(cls=DictBlock, **kw):
    db = cls(**DB_KW, **kw)
    K = jax.random.normal(jax.random.PRNGKey(3), (64, DB_KW["h"], DB_KW["k"]))
    params = db.init(jax.random.PRNGKey(4), K)
    return db, params, K


def onehot(K):
    return jax.nn.one_hot(jnp.argmax(K, -1), DB_KW["k"])


def test_head_drop_all_heads_to_origin():
    db, params, K = block(p_head_drop=1.0)
    F, P, _, _ = db.apply(params, K, None, 0.0, jax.random.PRNGKey(0),
                          temperature=0.7, method=DictBlock.withStats)
    W = db.apply(params, method=DictBlock.dicts)
    F_origin = jnp.einsum("hk,hkd->d", P.mean(0), W)
    assert jnp.allclose(F, jnp.broadcast_to(F_origin, F.shape), rtol=1e-5, atol=1e-6)


def test_head_drop_inactive_at_inference():
    db, params, K = block(p_head_drop=1.0)
    plain, _, _ = block()
    a = db.apply(params, K, None, 0.0, None, temperature=0.7, method=DictBlock.withStats)
    b = plain.apply(params, K, None, 0.0, None, temperature=0.7, method=DictBlock.withStats)
    assert jnp.allclose(a[0], b[0])


def test_head_drop_is_inverted():
    db, params, K = block(p_head_drop=0.5)
    P = db.apply(params, K, 0.7, method=DictBlock.cluster)
    origin = P.mean(0)
    # kept heads' deviations from the origin are scaled by 1/(1-p), so that
    # E[output] matches inference (where nothing is dropped)
    P_used, _ = db.apply(params, P, jax.random.PRNGKey(1), method=DictBlock.head_drop)
    dev = P_used - origin
    kept = jnp.abs(dev).sum(-1) > 1e-6
    assert jnp.allclose(dev[kept], 2.0 * (P - origin)[kept], atol=1e-5)


def test_head_drop_stats_describe_the_classification():
    # the stats read the classification, not the dropped-out lookup
    db, params, K = block(p_head_drop=1.0)
    plain, _, _ = block()
    a = db.apply(params, K, None, 0.0, jax.random.PRNGKey(0), temperature=0.7,
                 method=DictBlock.withStats)
    b = plain.apply(params, K, None, 0.0, None, temperature=0.7,
                    method=DictBlock.withStats)
    assert jnp.allclose(a[1], b[1])                    # P
    assert jnp.allclose(a[2][1], b[2][1])              # entropy


def test_features_off_draw_no_keys():
    # with both features off the random stream is untouched, so existing
    # models' noise and dropout draws are unchanged
    db, params, K = block(p_head_drop=0.0, fiber_rank=0)
    P = db.apply(params, K, 0.7, method=DictBlock.cluster)
    rng = jax.random.PRNGKey(5)
    _, r1 = db.apply(params, P, rng, method=DictBlock.head_drop)
    _, r2 = db.apply(params, None, rng, method=DictBlock.fiber_dropout)
    assert jnp.array_equal(r1, rng) and jnp.array_equal(r2, rng)


@pytest.mark.parametrize("cls", [DictBlock, ConcatDictBlock])
def test_fiber_adds_selected_entry_local_basis(cls):
    r = 3
    db, params, K = block(cls, fiber_rank=r)
    h, d = DB_KW["h"], DB_KW["d"]
    e = d // h if cls is ConcatDictBlock else d
    assert params["params"]["fiber"].shape == (h, DB_KW["k"], e, r)
    C = jax.random.normal(jax.random.PRNGKey(7), (K.shape[0], h, r))
    P = onehot(K)
    F = db.apply(params, P, None, C=C, method=cls.fwd)
    F0 = db.apply(params, P, None, C=None, method=cls.fwd)
    U = params["params"]["fiber"]                             # (h, k, e, r)
    Ub = U[jnp.arange(h)[None, :], jnp.argmax(K, -1)]         # (b, h, e, r)
    fib = jnp.einsum("bher,bhr->bhe", Ub, C)                  # (b, h, e)
    fib = fib.reshape(K.shape[0], d) if cls is ConcatDictBlock else fib.sum(1)
    assert jnp.allclose(F - F0, fib, rtol=1e-4, atol=1e-5)


def test_fiber_stats_describe_the_base():
    db, params, K = block(fiber_rank=2)
    plain, _, _ = block()
    C = jax.random.normal(jax.random.PRNGKey(8), (K.shape[0], DB_KW["h"], 2))
    pp = {"params": {"weights": params["params"]["weights"]}}
    a = db.apply(params, K, None, 0.0, None, temperature=0.7, C=C,
                 return_base=True, method=DictBlock.withStats)
    b = plain.apply(pp, K, None, 0.0, None, temperature=0.7,
                    method=DictBlock.withStats)
    assert jnp.allclose(a[2], b[2], rtol=1e-5, atol=1e-6)    # stats: base only
    assert jnp.allclose(a[4], b[0], rtol=1e-5, atol=1e-6)    # base output
    assert not jnp.allclose(a[0], b[0])                       # F has the fiber


def test_fiber_eps_caps_per_head_norm():
    db, params, K = block(fiber_rank=3, fiber_eps=0.1)
    C = 50.0 * jax.random.normal(jax.random.PRNGKey(7), (K.shape[0], DB_KW["h"], 3))
    Fb = db.apply(params, onehot(K), C, True, method=DictBlock.fiber_out)
    assert jnp.all(jnp.linalg.norm(Fb, axis=-1) <= 0.1 + 1e-5)


def test_fiber_eps_layer_caps_sum_over_heads():
    db, params, K = block(fiber_rank=3, fiber_eps_layer=0.3)
    C = 50.0 * jax.random.normal(jax.random.PRNGKey(7), (K.shape[0], DB_KW["h"], 3))
    Fb = db.apply(params, onehot(K), C, True, method=DictBlock.fiber_out)
    assert jnp.all(jnp.linalg.norm(Fb, axis=-1).sum(-1) <= 0.3 + 1e-5)


def test_fiber_radial_is_per_head_gain():
    db, params, K = block(fiber_rank=1, fiber_radial=True, fiber_bound=0.5)
    assert "fiber" not in params["params"]
    P = onehot(K)
    C = jax.random.normal(jax.random.PRNGKey(7), (K.shape[0], DB_KW["h"], 1))
    F = db.apply(params, P, None, C=C, method=DictBlock.fwd)
    W = db.apply(params, method=DictBlock.dicts)
    Fs = jnp.einsum("bhk,hkd->bhd", P, W)
    delta = 0.5 * jnp.tanh(C / 0.5)
    assert jnp.allclose(F, (Fs * (1 + delta)).sum(1), rtol=1e-4, atol=1e-5)


def test_fiber_radial_needs_rank_one():
    with pytest.raises(ValueError, match="fiber_rank == 1"):
        block(fiber_rank=2, fiber_radial=True)


def test_fiber_scales_with_the_layer_gain():
    # fibers are read from the shaped input and gained with the entry, so a
    # layer's whole output is linear in the input norm
    de = DictEnc(d_in=17, d_out=32, k=8, h=4, gainshape=True, n_const=1,
                 fiber_rank=3, select="ste",
                 dtype_str="float32", dtype_p_str="float32")
    E = jax.random.normal(jax.random.PRNGKey(2), (16, 16))
    E1 = jnp.concatenate([E, jnp.ones((16, 1))], -1)
    E2 = jnp.concatenate([2.0 * E, jnp.ones((16, 1))], -1)
    params = de.init(jax.random.PRNGKey(0), E1)
    Y1, Y2 = de.apply(params, E1), de.apply(params, E2)
    assert jnp.allclose(Y2, 2.0 * Y1, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("concat", [False, True])
def test_fiber_ontologizer_forward_and_grads(X, concat):
    model = Ontologizer(**{**KW, "fiber_rank": 4, "concat": concat,
                           "select": "ste", **COND})
    params = model.init(jax.random.PRNGKey(0), X)
    assert params["params"]["dictencs_1"]["fiber_read"].shape == (KW["h"], 4, KW["d_out"] + 1)
    Y = model.apply(params, X, temperature=0.5)
    Ys, _, _ = model.apply(params, X, 0.0, None, temperature=0.5, method=Ontologizer.withStats)
    assert jnp.allclose(Ys[-1], Y, atol=1e-5)

    def loss(p):
        Ys, _, _ = model.apply(p, X, 0.0, None, temperature=0.5, method=Ontologizer.withStats)
        return ((Ys - X) ** 2).mean()

    g = jax.grad(loss)(params)
    assert all(bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves(g))
    assert bool(jnp.any(g["params"]["dictencs_0"]["dict"]["fiber"] != 0))
    assert bool(jnp.any(g["params"]["dictencs_0"]["fiber_read"] != 0))


def test_fiber_shared_reads_one_vector(X):
    model = Ontologizer(**{**KW, "fiber_rank": 3, "fiber_shared": True, **COND})
    params = model.init(jax.random.PRNGKey(0), X)
    assert params["params"]["dictencs_1"]["fiber_read"].shape == (1, 3, KW["d_out"] + 1)

    def probe(m, X):
        U, _ = m.dictencs[0].gainshape_in(m.constinput(X))
        return m.dictencs[0].fiber_coords(U)

    C = model.apply(params, X, method=probe)
    assert C.shape == (X.shape[0], KW["h"], 3)
    assert jnp.allclose(C[:, 0], C[:, 1])
    assert jnp.isfinite(model.apply(params, X)).all()


def test_base_aux_appends_fiber_free_decode(X):
    kw = {**KW, "fiber_rank": 3, "select": "ste", **COND}
    model = Ontologizer(**{**kw, "base_aux": 2})
    params = model.init(jax.random.PRNGKey(0), X)
    Ys, _, _ = model.apply(params, X, 0.0, None, temperature=0.5, method=Ontologizer.withStats)
    assert Ys.shape[0] == KW["l"] + 2
    plain = Ontologizer(**kw)
    Yp, _, _ = plain.apply(params, X, 0.0, None, temperature=0.5, method=Ontologizer.withStats)
    assert jnp.allclose(Ys[:KW["l"]], Yp, atol=1e-5)    # the usual prefixes are unchanged
    assert jnp.allclose(Ys[-1], Ys[-2])                  # copies of one decode

    # the base decode is the same selections with the fiber removed: a model
    # whose fiber bases are zero reproduces it exactly, since zero fibers
    # change neither the residuals nor the selections
    p0 = jax.tree_util.tree_map(lambda x: x, params)
    for i in range(KW["l"]):
        f = p0["params"][f"dictencs_{i}"]["dict"]["fiber"]
        p0["params"][f"dictencs_{i}"]["dict"]["fiber"] = jnp.zeros_like(f)
    Y0, _, _ = model.apply(p0, X, 0.0, None, temperature=0.5, method=Ontologizer.withStats)
    assert jnp.allclose(Y0[-1], Y0[KW["l"] - 1], atol=1e-5)


def test_base_aux_needs_fibers_and_deepsup(X):
    with pytest.raises(ValueError, match="base_aux"):
        Ontologizer(**{**KW, "base_aux": 1, **COND}).init(jax.random.PRNGKey(0), X)
    with pytest.raises(ValueError, match="base_aux"):
        Ontologizer(**{**KW, "base_aux": 1, "fiber_rank": 2, "deepsup": False,
                       **COND}).init(jax.random.PRNGKey(0), X)


def test_base_aux_refuses_the_ghost_path(X):
    model = Ontologizer(**{**KW, "base_aux": 1, "fiber_rank": 2, **COND})
    params = model.init(jax.random.PRNGKey(0), X)
    with pytest.raises(NotImplementedError, match="base_aux"):
        model.apply(params, X, 0.0, None, temperature=0.5, method=Ontologizer.withGhost)

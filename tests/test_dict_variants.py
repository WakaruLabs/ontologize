# Dictionary-space variants: head-private blocks, signed entries, direct
# (decoder-free) output-space entries, and head dropout to the origin.
import jax
import jax.numpy as jnp
import pytest

from conftest import DB_KW, KW
from ontologize.layers.dictblock import DictBlock
from ontologize.ontologizer import Ontologizer

COND = dict(resid_norm=True, resid_const=True)


def block(**kw):
    db = DictBlock(**DB_KW, **kw)
    K = jax.random.normal(jax.random.PRNGKey(3), (64, DB_KW["h"], DB_KW["k"]))
    params = db.init(jax.random.PRNGKey(4), K)
    return db, params, K


def test_private_dicts_are_block_diagonal():
    db, params, _ = block(private=True)
    h, k, d = DB_KW["h"], DB_KW["k"], DB_KW["d"]
    assert params["params"]["weights"].shape == (h, k, d // h)
    W = db.apply(params, method=DictBlock.dicts)
    assert W.shape == (h, k, d)
    e = d // h
    for i in range(h):
        outside = jnp.concatenate([W[i, :, :i * e], W[i, :, (i + 1) * e:]], -1)
        assert jnp.all(outside == 0) and jnp.all(W[i, :, i * e:(i + 1) * e] >= 0)


def test_private_fast_stats_match_reference():
    ref, params, K = block(private=True)
    fast = DictBlock(**DB_KW, private=True, fast_stats=True)
    a = ref.apply(params, K, None, 0.0, None, temperature=0.7, method=DictBlock.withStats)
    b = fast.apply(params, K, None, 0.0, None, temperature=0.7, method=DictBlock.withStats)
    for x, y in zip(a[:3], b[:3]):
        assert jnp.allclose(x, y, rtol=1e-5, atol=1e-6)


def test_signed_dicts_keep_sign_fast_path_output():
    db, params, K = block(signed=True, fast_stats=True)
    W = db.apply(params, method=DictBlock.dicts)
    assert jnp.any(W < 0)
    F, P, st, _ = db.apply(params, K, None, 0.0, None, temperature=0.7,
                           method=DictBlock.withStats)
    F_ref = jnp.einsum("bhk,hkd->bd", P, W)
    assert jnp.allclose(F, F_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("fast", [False, True])
def test_head_drop_all_heads_to_origin(fast):
    db, params, K = block(p_head_drop=1.0, fast_stats=fast)
    F, P, _, _ = db.apply(params, K, None, 0.0, jax.random.PRNGKey(0),
                          temperature=0.7, method=DictBlock.withStats)
    W = db.apply(params, method=DictBlock.dicts)
    origin = P.mean(0)
    F_origin = jnp.einsum("hk,hkd->d", origin, W)
    assert jnp.allclose(F, jnp.broadcast_to(F_origin, F.shape), rtol=1e-5, atol=1e-6)


def test_head_drop_inactive_at_inference():
    db, params, K = block(p_head_drop=1.0)
    plain, _, _ = block()
    a = db.apply(params, K, None, 0.0, None, temperature=0.7, method=DictBlock.withStats)
    b = plain.apply(params, K, None, 0.0, None, temperature=0.7, method=DictBlock.withStats)
    assert jnp.allclose(a[0], b[0])


def test_direct_has_no_decoder_and_decodes_identity(X):
    model = Ontologizer(**{**KW, "e_dec": KW["d_out"], "direct": True,
                           "signed_dict": True, **COND})
    params = model.init(jax.random.PRNGKey(0), X)
    assert "decoder" not in params["params"]
    Y = model.apply(params, X)
    Ys, _, _ = model.apply(params, X, 0.0, None, method=Ontologizer.withStats)
    assert Y.shape == X.shape and jnp.allclose(Ys[-1], Y, atol=1e-5)

    def dec(m, v):
        return m.decode(v)

    V = jax.random.normal(jax.random.PRNGKey(2), X.shape)
    assert jnp.array_equal(model.apply(params, V, method=dec), V)


def test_private_heads_ontologizer_trains(X):
    model = Ontologizer(**{**KW, "private_heads": True, "fast_stats": True, **COND})
    params = model.init(jax.random.PRNGKey(0), X)

    def loss(p):
        Ys, _, _ = model.apply(p, X, 0.0, None, temperature=0.5,
                               method=Ontologizer.withStats)
        return ((Ys - X) ** 2).mean()

    g = jax.grad(loss)(params)
    assert all(bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves(g))


@pytest.mark.parametrize("mode", ["jumprelu", "topk"])
def test_head_sparse_encoder_in_ontologizer(X, mode):
    model = Ontologizer(**{**KW, "head_sparse": mode, "m_h": 16, "k_z": 3,
                           "fast_stats": True, **COND})
    params = model.init(jax.random.PRNGKey(0), X)
    assert "hs_encoder" in params["params"]["dictencs_0"]
    assert params["params"]["dictencs_0"]["classifier"]["W"].shape == (KW["h"], KW["k"], 16)
    Ys, stats, _ = model.apply(params, X, 0.0, None, temperature=0.5,
                               method=Ontologizer.withStats)
    assert Ys.shape == (KW["l"],) + X.shape
    L0 = stats[:, 0]                       # the L1_K slot holds the encoders' L0
    assert jnp.all(L0 >= 0) and jnp.all(L0 <= KW["h"] * 16)
    if mode == "topk":
        assert jnp.allclose(L0, KW["h"] * 3, atol=KW["h"] * 3 * 0.5)  # <= k_z per head

    def loss(p):
        Ys, st, _ = model.apply(p, X, 0.0, None, temperature=0.5,
                                method=Ontologizer.withStats)
        return ((Ys - X) ** 2).mean() + 1e-3 * st[:, 0].sum()

    g = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(g)
    assert all(bool(jnp.isfinite(x).all()) for x in leaves)
    if mode == "jumprelu":  # the L0 penalty reaches the thresholds via the STE
        gt = g["params"]["dictencs_0"]["hs_encoder"]["log_threshold"]
        assert bool(jnp.any(gt != 0))


@pytest.mark.parametrize("private", [False, True])
def test_fiber_adds_selected_entry_local_basis(private):
    r = 3
    db, params, K = block(fiber_rank=r, private=private)
    h, k, d = DB_KW["h"], DB_KW["k"], DB_KW["d"]
    C = jax.random.normal(jax.random.PRNGKey(7), (K.shape[0], h, r))
    P = jax.nn.one_hot(jnp.argmax(K, -1), k)                 # hard selection
    F = db.apply(params, P, None, C=C, method=DictBlock.fwd)
    F0 = db.apply(params, P, None, C=None, method=DictBlock.fwd)
    W = db.apply(params, method=DictBlock.dicts)
    U = params["params"]["fiber"]                             # (h, k, e, r)
    sel = jnp.argmax(K, -1)                                   # (b, h)
    Ub = U[jnp.arange(h)[None, :], sel]                       # (b, h, e, r)
    fib = jnp.einsum("bher,bhr->bhe", Ub, C)                  # (b, h, e)
    if private:
        fib = fib.reshape(K.shape[0], d)                      # concatenated blocks
    else:
        fib = fib.sum(1)
    assert jnp.allclose(F0, jnp.einsum("bhk,hkd->bd", P, W), atol=1e-5)
    assert jnp.allclose(F - F0, fib, rtol=1e-4, atol=1e-5)


def test_fiber_fast_stats_match_reference():
    r = 2
    ref, params, K = block(fiber_rank=r, private=True)
    fast = DictBlock(**DB_KW, fiber_rank=r, private=True, fast_stats=True)
    C = jax.random.normal(jax.random.PRNGKey(8), (K.shape[0], DB_KW["h"], r))
    a = ref.apply(params, K, None, 0.0, None, temperature=0.7, C=C, method=DictBlock.withStats)
    b = fast.apply(params, K, None, 0.0, None, temperature=0.7, C=C, method=DictBlock.withStats)
    assert jnp.allclose(a[0], b[0], rtol=1e-5, atol=1e-6)      # F includes the fiber
    assert jnp.allclose(a[2], b[2], rtol=1e-5, atol=1e-6)      # stats: base only


def test_fiber_ontologizer_forward_and_grads(X):
    model = Ontologizer(**{**KW, "fiber_rank": 4, "private_heads": True,
                           "fast_stats": True, "select": "ste", **COND})
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


def test_gain_clip_bounds_gain(X):
    model = Ontologizer(**{**KW, "resid_first": True, "resid_gain": True,
                           "gain_clip": True, **COND})
    params = model.init(jax.random.PRNGKey(0), X)

    def probe(m, X):
        R = m.resid(X)
        E = m.first_input(X, X, R)
        R1, _ = m.dictencs[0].withClusts(R, E)
        return m.gain(X, R1, 1), jnp.linalg.norm(X, axis=-1, keepdims=True)

    g, nx = model.apply(params, X, method=probe)
    assert jnp.all(g <= nx + 1e-6)


def test_anneal_select_interpolates_to_ste():
    db, params, K = block(select="anneal")
    P0 = db.apply(params, K, 1.0, 0.0, method=DictBlock.cluster)
    P1 = db.apply(params, K, 1.0, 1.0, method=DictBlock.cluster)
    Pd = db.apply(params, K, 1.0, method=DictBlock.cluster)          # default: hard
    soft = jax.nn.softmax(K, -1)
    hard = jax.nn.one_hot(jnp.argmax(K, -1), DB_KW["k"])
    assert jnp.allclose(P0, soft, atol=1e-6)
    assert jnp.allclose(P1, hard, atol=1e-6) and jnp.allclose(Pd, hard, atol=1e-6)
    Ph = db.apply(params, K, 1.0, 0.5, method=DictBlock.cluster)
    assert jnp.allclose(Ph, 0.5 * soft + 0.5 * hard, atol=1e-6)
    # gradient flows through the soft part at every hardness
    g = jax.grad(lambda k: db.apply(params, k, 1.0, 1.0, method=DictBlock.cluster).sum())(K)
    assert bool(jnp.isfinite(g).all())


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


def test_topm_select_schedule():
    db, params, K = block(select="topm")
    k = DB_KW["k"]
    soft = jax.nn.softmax(K, -1)
    P0 = db.apply(params, K, 1.0, 0.0, method=DictBlock.cluster)       # m = k: soft
    assert jnp.allclose(P0, soft, atol=1e-6)
    P1 = db.apply(params, K, 1.0, 1.0, method=DictBlock.cluster)       # m = 1: argmax
    assert jnp.allclose(P1, jax.nn.one_hot(jnp.argmax(K, -1), k), atol=1e-6)
    Ph = db.apply(params, K, 1.0, 0.5, method=DictBlock.cluster)       # m = round(k^0.5)
    m = int(round(k ** 0.5))
    assert jnp.allclose((Ph > 0).sum(-1), m) and jnp.allclose(Ph.sum(-1), 1.0, atol=1e-5)
    top = jnp.sort(soft, -1)[..., ::-1][..., :m]
    assert jnp.allclose(jnp.sort(Ph, -1)[..., ::-1][..., :m], top / top.sum(-1, keepdims=True), atol=1e-5)
    g = jax.grad(lambda x: db.apply(params, x, 1.0, 1.0, method=DictBlock.cluster).sum())(K)
    assert bool(jnp.isfinite(g).all())


def test_fiber_eps_caps_per_head_norm():
    db, params, K = block(fiber_rank=3, fiber_eps=0.1)
    C = 50.0 * jax.random.normal(jax.random.PRNGKey(7), (K.shape[0], DB_KW["h"], 3))
    P = jax.nn.one_hot(jnp.argmax(K, -1), DB_KW["k"])
    Fb = db.apply(params, P, C, True, method=DictBlock.fiber_out)
    assert jnp.all(jnp.linalg.norm(Fb, axis=-1) <= 0.1 + 1e-5)


def test_fiber_radial_is_per_head_gain():
    db, params, K = block(fiber_rank=1, fiber_radial=True, fiber_bound=0.5)
    assert "fiber" not in params["params"]
    P = jax.nn.one_hot(jnp.argmax(K, -1), DB_KW["k"])
    C = jax.random.normal(jax.random.PRNGKey(7), (K.shape[0], DB_KW["h"], 1))
    F = db.apply(params, P, None, C=C, method=DictBlock.fwd)
    W = db.apply(params, method=DictBlock.dicts)
    Fs = jnp.einsum("bhk,hkd->bhd", P, W)
    delta = 0.5 * jnp.tanh(C / 0.5)
    assert jnp.allclose(F, (Fs * (1 + delta)).sum(1), rtol=1e-4, atol=1e-5)


def test_fiber_shared_reads_one_vector(X):
    model = Ontologizer(**{**KW, "fiber_rank": 3, "fiber_shared": True, "direct": True,
                           "e_dec": KW["d_out"], "signed_dict": True, **COND})
    params = model.init(jax.random.PRNGKey(0), X)
    assert params["params"]["dictencs_1"]["fiber_read"].shape == (1, 3, KW["d_out"] + 1)

    def probe(m, X):
        R = m.resid(X)
        E = m.first_input(X, X, R)
        return m.dictencs[0].fiber_coords(E)

    C = model.apply(params, X, method=probe)
    assert C.shape == (X.shape[0], KW["h"], 3)
    assert jnp.allclose(C[:, 0], C[:, 1])
    assert jnp.isfinite(model.apply(params, X)).all()


def test_fiber_eps_layer_caps_sum_over_heads():
    db, params, K = block(fiber_rank=3, fiber_eps_layer=0.3)
    C = 50.0 * jax.random.normal(jax.random.PRNGKey(7), (K.shape[0], DB_KW["h"], 3))
    P = jax.nn.one_hot(jnp.argmax(K, -1), DB_KW["k"])
    Fb = db.apply(params, P, C, True, method=DictBlock.fiber_out)
    assert jnp.all(jnp.linalg.norm(Fb, axis=-1).sum(-1) <= 0.3 + 1e-5)


def test_base_aux_appends_fiber_free_decode(X):
    kw = {**KW, "fiber_rank": 3, "direct": True, "e_dec": KW["d_out"], "signed_dict": True,
          "resid_first": True, "fast_stats": True, **COND}
    model = Ontologizer(**{**kw, "base_aux": 1})
    params = model.init(jax.random.PRNGKey(0), X)
    Ys, _, _ = model.apply(params, X, 0.0, None, temperature=0.5, method=Ontologizer.withStats)
    assert Ys.shape[0] == KW["l"] + 1
    plain = Ontologizer(**kw)
    Yp, _, _ = plain.apply(params, X, 0.0, None, temperature=0.5, method=Ontologizer.withStats)
    assert jnp.allclose(Ys[:-1], Yp, atol=1e-5)          # the usual prefixes are unchanged

    # a model with the fiber bases zeroed reproduces the base-only decode
    p0 = jax.tree_util.tree_map(lambda x: x, params)
    for i in range(KW["l"]):
        p0["params"][f"dictencs_{i}"]["dict"]["fiber"] = jnp.zeros_like(p0["params"][f"dictencs_{i}"]["dict"]["fiber"])
    Y0, _, _ = plain.apply(p0, X, 0.0, None, temperature=0.5, method=Ontologizer.withStats)
    # (selections drift once residuals differ, so this is only a coarse check)
    assert float(jnp.abs(Ys[-1] - Y0[-1]).mean()) < float(jnp.abs(Ys[-1] - Yp[-1]).mean()) + 1e-3
    assert bool(jnp.isfinite(Ys).all())


def test_shared_head_sparse_encoder(X):
    model = Ontologizer(**{**KW, "head_sparse": "topk", "hs_shared": True, "m_h": 24, "k_z": 3,
                           "fast_stats": True, **COND})
    params = model.init(jax.random.PRNGKey(0), X)
    assert params["params"]["dictencs_0"]["hs_encoder"]["W_enc"].shape == (1, 24, KW["d_in"])
    Ys, stats, _ = model.apply(params, X, 0.0, None, temperature=0.5, method=Ontologizer.withStats)
    assert bool(jnp.isfinite(Ys).all())
    assert float(stats[0, 0]) <= 3 * 1 + 1e-6      # L0 of one shared encoder: <= k_z

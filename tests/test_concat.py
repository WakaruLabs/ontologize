# `ConcatDictBlock`: heads write disjoint slices of the layer output
# instead of summing into all of it. The overrides exist because the
# parent's `cossim`, `flatcos` and `tags` would compare or hand out
# vectors from different slices as though they shared a space, so the
# tests here pin each against an explicit reference computed on the
# slices EMBEDDED in the full width -- which is what concatenation means
# geometrically -- rather than against the closed forms they use.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.layers.dictblock import DictBlock, ConcatDictBlock
from ontologize.ontologizer import Ontologizer

from conftest import KW, B, DB_KW, RETIRED_LAYER, finite_live


CAT_KW = dict(DB_KW)                      # d=16, h=4 -> 4 dims per head


def embed(Fs: np.ndarray, h: int, d: int) -> np.ndarray:
    """Per-head slices placed in the full width, zero outside the slice."""
    out = np.zeros(Fs.shape[:-2] + (h, d), Fs.dtype)
    w = d // h
    for i in range(h):
        out[..., i, i * w:(i + 1) * w] = Fs[..., i, :]
    return out


def gram_mean(U: np.ndarray) -> float:
    """`Sparse.cossim`'s reduction: full cosine Gram over the -1 axis with
    -2 as the batch, absolute value, meaned over every entry including the
    diagonal. A zero row's cosines are 0 (`recip_norm`)."""
    n = np.linalg.norm(U, axis=-1, keepdims=True)
    U = np.divide(U, n, out=np.zeros_like(U), where=n > 0)
    return float(np.abs(np.einsum("...id,...jd->...ij", U, U)).mean())


def offdiag_mean(U: np.ndarray) -> float:
    """Mean off-diagonal cosine over every pair of rows, `flatcos`'s
    definition, via the full Gram it replaces with an O(N d) identity."""
    n = np.linalg.norm(U, axis=-1, keepdims=True)
    U = np.divide(U, n, out=np.zeros_like(U), where=n > 0)
    C = U @ U.T
    return float(C[~np.eye(len(U), dtype=bool)].mean())


@pytest.fixture(scope="module")
def block():
    """A `ConcatDictBlock` with params, soft classifications, and the
    per-head and combined outputs."""
    db = ConcatDictBlock(**CAT_KW)
    K = jax.random.normal(jax.random.PRNGKey(3), (16, CAT_KW["h"], CAT_KW["k"]))
    params = db.init(jax.random.PRNGKey(4), K)
    P = db.apply(params, K, 0.5, method=ConcatDictBlock.cluster)
    Fs = db.apply(params, P, method=ConcatDictBlock.hfwd)
    return db, params, K, P, Fs


def test_weights_narrow_to_a_slice_per_head():
    """The point of the construction: h times fewer dictionary
    parameters at the same layer output width."""
    args = dict(k=DB_KW["k"], d=DB_KW["d"], h=DB_KW["h"],
                dtype_str="float32", dtype_p_str="float32")
    K = jnp.zeros((2, args["h"], args["k"]))
    shapes = {}
    for name, cls in (("sum", DictBlock), ("cat", ConcatDictBlock)):
        p = cls(**args).init(jax.random.PRNGKey(0), K)
        shapes[name] = p["params"]["weights"].shape
    h, k, d = args["h"], args["k"], args["d"]
    assert shapes["sum"] == (h, k, d)
    assert shapes["cat"] == (h, k, d // h)
    assert np.prod(shapes["sum"]) == h * np.prod(shapes["cat"])


def test_indivisible_width_raises():
    """A truncating `d // h` would silently drop output dimensions, so
    this has to fail at construction rather than produce a model."""
    db = ConcatDictBlock(k=4, d=18, h=4, dtype_str="float32",
                         dtype_p_str="float32")
    with pytest.raises(ValueError, match="divisible"):
        db.init(jax.random.PRNGKey(0), jnp.zeros((2, 4, 4)))


def test_combine_splits_back_and_fwd_agrees(block):
    db, params, K, P, Fs = block
    F = db.apply(params, Fs, method=ConcatDictBlock.combine)
    back = db.apply(params, F, method=ConcatDictBlock.split)
    assert F.shape == (16, CAT_KW["d"])
    assert np.array_equal(np.asarray(back), np.asarray(Fs))
    fwd = db.apply(params, P, method=ConcatDictBlock.fwd)
    assert np.array_equal(np.asarray(fwd), np.asarray(F))


def test_a_head_writes_only_its_own_slice(block):
    """Equivalently: the layer output is the slices laid end to end, so
    reading slice i back gives head i's contribution untouched by any
    other head. This is what makes the heads orthogonal."""
    db, params, K, P, Fs = block
    F = np.asarray(db.apply(params, P, method=ConcatDictBlock.fwd))
    h, d = CAT_KW["h"], CAT_KW["d"]
    assert np.array_equal(F.reshape(-1, h, d // h), np.asarray(Fs))


def test_rev_reads_each_head_against_its_own_slice(block):
    db, params, K, P, Fs = block
    F = db.apply(params, Fs, method=ConcatDictBlock.combine)
    W = np.asarray(db.apply(params, method=ConcatDictBlock.dicts))
    got = np.asarray(db.apply(params, F, method=ConcatDictBlock.rev))
    want = np.einsum("hkd,bhd->bhk", W, np.asarray(Fs))
    assert np.abs(got - want).max() < 1e-5


def test_scaling_vector_scales_the_slice_not_the_whole(block):
    db, params, K, P, Fs = block
    h, d = CAT_KW["h"], CAT_KW["d"]
    S = jax.random.uniform(jax.random.PRNGKey(5), (16, h)) + 0.5
    F = np.asarray(db.apply(params, P, S, method=ConcatDictBlock.fwd))
    want = np.asarray(S)[:, :, None] * np.asarray(Fs)
    assert np.abs(F.reshape(-1, h, d // h) - want).max() < 1e-5


def test_cossim_equals_the_parent_formula_on_embedded_slices(block):
    """The override is a closed form of the parent's reduction, not a
    redefinition of it. In particular it is NOT 0: the parent means over
    the full Gram including its unit diagonal, so a layer whose heads are
    all live scores 1/h, and the column stays comparable with a summing
    layer's."""
    db, params, K, P, Fs = block
    h = CAT_KW["h"]
    got = float(db.apply(params, Fs, method=ConcatDictBlock.cossim))
    want = gram_mean(embed(np.asarray(Fs, np.float64), h, CAT_KW["d"]))
    assert got == pytest.approx(want, abs=1e-6)
    assert got == pytest.approx(1.0 / h, abs=1e-6)


def test_cossim_drops_a_dead_head(block):
    """`recip_norm` gives a zero vector cosine 0, so the parent's Gram
    loses that head's diagonal entry too. The closed form has to track
    that or it would report structure a collapsed head does not have."""
    db, params, K, P, Fs = block
    h = CAT_KW["h"]
    Fs_d = np.asarray(Fs, np.float64).copy()
    Fs_d[:, 0, :] = 0.0
    got = float(db.apply(params, jnp.asarray(Fs_d, jnp.float32),
                         method=ConcatDictBlock.cossim))
    assert got == pytest.approx(gram_mean(embed(Fs_d, h, CAT_KW["d"])), abs=1e-6)
    assert got == pytest.approx((h - 1) / h ** 2, abs=1e-6)


def test_tags_embed_atoms_in_the_layer_output_space(block):
    """Callers read `tags()` rows as vectors in the layer's output space
    (`decodeEntries` pushes them through the decoder), so an atom has to
    carry its head's offset rather than arrive as a bare slice."""
    db, params, K, P, Fs = block
    h, k, d = CAT_KW["h"], CAT_KW["k"], CAT_KW["d"]
    W = np.asarray(db.apply(params, method=ConcatDictBlock.dicts))
    T = np.asarray(db.apply(params, method=ConcatDictBlock.tags))
    assert T.shape == (h * k, d)
    blocks = T.reshape(h, k, h, d // h)
    for i in range(h):
        assert np.array_equal(blocks[i, :, i, :], W[i])
        off = np.delete(blocks[i], i, axis=1)
        assert not off.any()


def test_flatcos_matches_the_explicit_gram_over_those_tags(block):
    """Ties the two overrides together: `flatcos`'s pair-count rescaling
    of `rowcos` has to equal the honest Gram mean over the embedded
    dictionary, and every cross-head pair in it is exactly 0."""
    db, params, K, P, Fs = block
    got = float(db.apply(params, method=ConcatDictBlock.flatcos))
    T = np.asarray(db.apply(params, method=ConcatDictBlock.tags), np.float64)
    assert got == pytest.approx(offdiag_mean(T), abs=1e-6)


def test_ghost_reads_each_head_against_its_own_slice(block):
    """The parent hands every head the whole output, which here is
    neither the right width nor the right dimensions."""
    db, params, K, P, Fs = block
    F = db.apply(params, Fs, method=ConcatDictBlock.combine)
    G = db.apply(params, K, F, method=ConcatDictBlock.ghost)
    assert G.shape == F.shape
    assert np.isfinite(np.asarray(G)).all()


def test_zero_ablation_is_confined_to_the_head(block):
    """Intervention locality is the property the construction is for: in
    the summing form an ablated head's atoms were entangled with every
    other head's in the same coordinates."""
    db, params, K, P, Fs = block
    h, d = CAT_KW["h"], CAT_KW["d"]
    F0 = np.asarray(db.apply(params, P, method=ConcatDictBlock.fwd))
    F1 = np.asarray(db.apply(params, P, None, h_zero=jnp.array([0, 1]),
                             method=ConcatDictBlock.fwd))
    B0 = F0.reshape(-1, h, d // h)
    B1 = F1.reshape(-1, h, d // h)
    assert not B1[:, :2].any()
    assert np.array_equal(B1[:, 2:], B0[:, 2:])


def _decoder(params):
    W = np.asarray(params["params"]["decoder"]["weights"])
    return W if W.shape[0] == KW["d_out"] else W.T


@pytest.mark.parametrize("d_head,null", [(KW["d_out"] // 2, 0),
                                         (KW["d_out"], 0),
                                         (KW["d_out"] * 2, KW["d_out"])])
def test_decoder_column_blocks_lose_their_null_space_below_d_out(d_head, null):
    """The construction's actual content. A head reaches the output only
    through its own `(d_out, d_head)` column block of the decoder, which
    has full column rank while `d_head <= d_out` -- so no dictionary
    direction is invisible downstream, unlike the summing form, where
    every head goes through the whole `(d_out, e_dec)` decoder and an
    atom can sit in its null space taking no gradient. Widening a slice
    past `d_out` gives that back, which is why `e_dec = h * d_head` with
    a small `d_head` is the useful setting and not `d_head = e_dec`."""
    h = KW["h"]
    X = jnp.zeros((2, KW["d_in"]))
    model = Ontologizer(**{**KW, "e_dec": h * d_head, "concat": True})
    W = _decoder(model.init(jax.random.PRNGKey(0), X))
    ranks = [np.linalg.matrix_rank(W[:, i * d_head:(i + 1) * d_head])
             for i in range(h)]
    assert all(d_head - r == null for r in ranks)


def test_summing_decoder_has_the_null_space_concat_removes():
    """The contrast the test above is against, so a change in the
    decoder's construction cannot quietly make both pass."""
    X = jnp.zeros((2, KW["d_in"]))
    model = Ontologizer(**{**KW, "concat": False})
    W = _decoder(model.init(jax.random.PRNGKey(0), X))
    assert KW["e_dec"] > KW["d_out"]
    assert KW["e_dec"] - np.linalg.matrix_rank(W) == KW["e_dec"] - KW["d_out"]


def test_ontologizer_forward_differs_and_the_dictionary_shrinks(X):
    """`concat` has to reach the model, not merely construct: the run
    that silently trains a bit-identical baseline is the expensive
    failure mode here."""
    built = {}
    for concat in (False, True):
        model = Ontologizer(**{**KW, "concat": concat})
        params = model.init(jax.random.PRNGKey(0), X)
        built[concat] = (np.asarray(model.apply(params, X)),
                         sum(int(np.prod(v.shape)) for v in
                             jax.tree_util.tree_leaves(params)),
                         params["params"]["dictencs_0"]["dict"]
                               ["weights"].shape)
    Y_sum, n_sum, w_sum = built[False]
    Y_cat, n_cat, w_cat = built[True]
    assert np.isfinite(Y_cat).all() and Y_cat.shape == Y_sum.shape
    assert np.abs(Y_cat - Y_sum).max() > 1e-6
    assert w_cat == (KW["h"], KW["k"], KW["e_dec"] // KW["h"])
    assert n_cat < n_sum


SUPPORT = 12  # per-layer stats column of `support_overlap`


def test_stats_row_is_finite_and_support_overlap_is_zero(X):
    """`withStats` is the training path, and it is where the override's
    value has to land: heads own disjoint slices, so their support
    overlap is exactly 0 in every layer."""
    model = Ontologizer(**{**KW, "concat": True})
    params = model.init(jax.random.PRNGKey(0), X)
    _, stats, _ = model.apply(params, X, 1.0, rng=jax.random.PRNGKey(2),
                              method=Ontologizer.withStats)
    stats = np.asarray(stats)
    assert stats.shape[0] == KW["l"]
    assert finite_live(stats, RETIRED_LAYER)
    assert np.all(stats[:, SUPPORT] == 0.0)


def test_summing_support_overlap_is_not_zero(X):
    """Guards the column above from being right by accident: summing heads
    share the whole dictionary space, so their supports overlap."""
    model = Ontologizer(**{**KW, "concat": False})
    params = model.init(jax.random.PRNGKey(0), X)
    _, stats, _ = model.apply(params, X, 1.0, rng=jax.random.PRNGKey(2),
                              method=Ontologizer.withStats)
    assert np.all(np.asarray(stats)[:, SUPPORT] > 0.1)

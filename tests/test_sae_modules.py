# `ontologize.sae` (SAE, BilinearSAE), `ontologize.grouped` (Grouped) and
# `ontologize.blocksparse` (BlockSparse): the SAE baselines as `ontologize`
# modules. The root sae.py trains them through the flat params.npz layout
# every downstream script reads, so the converters between the two layouts
# carry the whole comparison; the behaviour itself is exercised through
# sae.py in test_sae.py and test_gated.py.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.sae import (SAE, BilinearSAE, from_legacy, legacy_encoder,
                            to_legacy)
from ontologize.grouped import Grouped
from ontologize.blocksparse import BlockSparse

D, M = 16, 32
B, G = 4, 8      # BlockSparse: G blocks of B latents


def models():
    return [SAE(d=D, m=M, topk=8), BilinearSAE(d=D, m=M, topk=8),
            SAE(d=D, m=M, topk=0, enc="gated"),
            Grouped(d=D, m=M, groups=8),
            Grouped(d=D, m=M, groups=8, group_fn="softmax"),
            Grouped(d=D, m=M, groups=8, enc="bilinear"),
            BlockSparse(d=D, m=M, block=B, topk=2)]


@pytest.mark.parametrize("model", models(), ids=lambda m: repr(m)[:60])
def test_layout_round_trips(model):
    """`initialize` builds the tree `init` would, and the flat layout
    converts to it and back without loss."""
    tree = model.initialize(jax.random.PRNGKey(0), np.zeros(D))
    ref = model.init(jax.random.PRNGKey(1), jnp.zeros((2, D)))["params"]
    assert jax.tree_util.tree_structure(tree) == \
        jax.tree_util.tree_structure(ref)
    assert all(a.shape == b.shape for a, b in zip(
        jax.tree_util.tree_leaves(tree), jax.tree_util.tree_leaves(ref)))
    flat = to_legacy(tree)
    assert legacy_encoder(flat) == model.enc
    back = from_legacy(flat)
    assert all(np.array_equal(a, b) for a, b in zip(
        jax.tree_util.tree_leaves(tree), jax.tree_util.tree_leaves(back)))


def test_weights_follow_linear_convention():
    """`weights` are (d_out, d_in), as in `Linear`: the decoder is
    (d, m), so the flat layout's decoder ROWS are its columns."""
    tree = SAE(d=D, m=M).initialize(jax.random.PRNGKey(0), np.zeros(D))
    assert tree["encoder"]["weights"].shape == (M, D)
    assert tree["decoder"]["weights"].shape == (D, M)
    assert to_legacy(tree)["W_dec"].shape == (M, D)


def test_renorm_holds_decoder_rows_except_under_softmax():
    tree = SAE(d=D, m=M).initialize(jax.random.PRNGKey(0), np.zeros(D))
    big = {**tree, "decoder": {**tree["decoder"],
                               "weights": 3.0 * tree["decoder"]["weights"]}}
    rows = SAE(d=D, m=M).renorm(big)["decoder"]["weights"]
    assert np.allclose(np.linalg.norm(rows, axis=0), 1.0, atol=1e-5)
    soft = Grouped(d=D, m=M, groups=8, group_fn="softmax")
    assert not soft.unit_rows
    assert soft.renorm(big) is big   # bounded coefficients: rows keep scale


def test_grouped_activation_is_per_group():
    model = Grouped(d=D, m=M, groups=8)
    tree = model.initialize(jax.random.PRNGKey(0), np.zeros(D))
    X = jax.random.normal(jax.random.PRNGKey(2), (64, D))
    z = np.asarray(model.apply({"params": tree}, X, method=Grouped.encode))
    assert ((z.reshape(64, 8, -1) > 0).sum(-1) <= 1).all()


@pytest.mark.parametrize("model", [
    Grouped(d=D, m=M, groups=8, topk=4),          # groups replace top-k
    Grouped(d=D, m=M, groups=8, enc="gated"),     # gate sets its own sparsity
    Grouped(d=D, m=M, groups=5),                  # m not a multiple
    Grouped(d=D, m=M, groups=8, group_fn="max"),
    SAE(d=D, m=M, prefixes=3),
    SAE(d=D, m=M, enc="conv"),
    SAE(d=D, m=M, topk=8, enc="gated"),
    BilinearSAE(d=D, m=M, enc="linear"),
    BlockSparse(d=D, m=M, block=3, topk=2),       # m not a multiple
    BlockSparse(d=D, m=M, block=B, topk=0),       # no block active
    BlockSparse(d=D, m=M, block=B, topk=G + 1),   # more than the blocks
    BlockSparse(d=D, m=M, block=B, topk=2, enc="bilinear"),
    BlockSparse(d=D, m=M, block=B, topk=2, prefixes=2),
])
def test_invalid_configurations_raise(model):
    with pytest.raises(ValueError):
        model.init(jax.random.PRNGKey(0), jnp.zeros((2, D)))


def test_gated_encodes_without_training_settings_but_will_not_train():
    """Downstream scripts encode gated checkpoints with no L1 at hand, so
    construction must not demand one; the loss must, or the gate shuts."""
    model = SAE(d=D, m=M, topk=0, enc="gated")
    tree = model.initialize(jax.random.PRNGKey(0), np.zeros(D))
    X = jax.random.normal(jax.random.PRNGKey(3), (8, D))
    model.apply({"params": tree}, X, method=SAE.encode)
    with pytest.raises(ValueError):
        model.apply({"params": tree}, X, jnp.ones(D), jnp.zeros(M, bool),
                    method=SAE.loss)
    trains = SAE(d=D, m=M, topk=0, enc="gated", s_l1=1e-2, aux_k=0)
    loss, _ = trains.apply({"params": tree}, X, jnp.ones(D),
                           jnp.zeros(M, bool), method=SAE.loss)
    assert np.isfinite(float(loss))


def test_eigenfeatures_need_a_bilinear_encoder():
    tree = SAE(d=D, m=M).initialize(jax.random.PRNGKey(0), np.zeros(D))
    with pytest.raises(ValueError):
        SAE(d=D, m=M).apply({"params": tree}, method=SAE.eigenfeatures)
    model = BilinearSAE(d=D, m=M)
    tree = model.initialize(jax.random.PRNGKey(0), np.zeros(D))
    E = model.apply({"params": tree}, method=BilinearSAE.eigenfeatures)
    assert E.shape == (M, D)


@pytest.mark.parametrize("model", [BilinearSAE(d=D, m=M),
                                   BlockSparse(d=D, m=M, block=B, topk=2)])
def test_resample_refuses_bilinear_and_blocks(model):
    tree = model.initialize(jax.random.PRNGKey(0), np.zeros(D))
    with pytest.raises(ValueError):
        model.resample(tree, (), jnp.zeros((4, D)), jnp.ones(D),
                       np.ones(M, bool), np.random.default_rng(0))


# ---- BlockSparse ----

def block_model(**kw):
    return BlockSparse(d=D, m=M, block=B, topk=2, **kw)


def test_block_code_keeps_the_largest_blocks_whole():
    """The code keeps the `topk` blocks of largest norm, signs and all, and
    zeroes every other block."""
    model = block_model()
    tree = model.initialize(jax.random.PRNGKey(0), np.zeros(D))
    X = jax.random.normal(jax.random.PRNGKey(2), (64, D))
    pre = np.asarray(model.apply({"params": tree}, X,
                                 method=BlockSparse.preacts)).reshape(64, G, B)
    z = np.asarray(model.apply({"params": tree}, X,
                               method=BlockSparse.encode)).reshape(64, G, B)
    on = (z != 0).any(-1)
    top = np.argsort(-(pre ** 2).sum(-1), -1)[:, :2]
    assert all(set(np.flatnonzero(on[i])) == set(top[i]) for i in range(64))
    assert np.array_equal(z[on], pre[on])
    assert (z < 0).any()     # signed, where a ReLU code is not


def test_block_init_decodes_each_blocks_projection():
    """Each decoder block starts as an orthonormal frame at Frobenius norm
    1 and the encoder as its transpose times the block size, so a block
    decodes the projection of the centred input onto its span. At block
    size 1 that is the tied initialization of unit rows."""
    flat = to_legacy(block_model().initialize(jax.random.PRNGKey(0),
                                              np.zeros(D)))
    Dg = np.asarray(flat["W_dec"]).reshape(G, B, D)
    W = np.asarray(flat["W_blk"])                              # (D, G, B)
    assert np.allclose(np.einsum("gid,gjd->gij", Dg, Dg), np.eye(B) / B,
                       atol=1e-5)
    x = np.random.default_rng(0).normal(size=D)
    for g in range(G):
        Q = np.linalg.qr(Dg[g].T)[0]
        assert np.allclose((x @ W[:, g]) @ Dg[g], Q @ (Q.T @ x), atol=1e-4)
    one = to_legacy(BlockSparse(d=D, m=M, block=1, topk=4).initialize(
        jax.random.PRNGKey(0), np.zeros(D)))
    assert np.allclose(np.linalg.norm(one["W_dec"], axis=-1), 1.0, atol=1e-5)
    assert np.allclose(one["W_blk"][:, :, 0], np.asarray(one["W_dec"]).T,
                       atol=1e-6)


def test_block_renorm_caps_blocks_at_the_unit_ball():
    model = block_model()
    tree = model.initialize(jax.random.PRNGKey(0), np.zeros(D))
    scale = jnp.repeat(jnp.array([3.0, 0.5] * (G // 2)), B)    # per latent
    W = tree["decoder"]["weights"] * scale
    out = model.renorm({**tree, "decoder": {**tree["decoder"], "weights": W}})
    Wb = np.asarray(out["decoder"]["weights"]).reshape(D, G, B)
    assert np.allclose(np.sqrt((Wb ** 2).sum((0, 2))), [1.0, 0.5] * (G // 2),
                       atol=1e-5)


def test_block_loss_trains_with_its_own_auxiliary_only():
    """aux-k or an L1 would make a different model, and the runner-ups
    need room, so the loss refuses those; otherwise it adds the runner-up
    term to the reconstruction."""
    tree = block_model().initialize(jax.random.PRNGKey(0), np.zeros(D))
    args = (jax.random.normal(jax.random.PRNGKey(3), (8, D)), jnp.ones(D),
            jnp.zeros(M, bool))
    for bad in (block_model(),                          # aux_k defaults on
                block_model(aux_k=0, s_l1=1e-3),
                BlockSparse(d=D, m=M, block=B, topk=G // 2 + 1, aux_k=0)):
        with pytest.raises(ValueError):
            bad.apply({"params": tree}, *args, method=BlockSparse.loss)
    loss, (mse, fired) = block_model(aux_k=0).apply(
        {"params": tree}, *args, method=BlockSparse.loss)
    assert float(loss) > float(mse) > 0
    f = np.asarray(fired).reshape(G, B)
    assert (f.all(-1) == f.any(-1)).all()               # blocks fire whole


def test_runner_up_blocks_get_gradient():
    """On one sample the selected blocks learn from the reconstruction and
    the next `topk` by norm from the auxiliary term; no other block moves."""
    model = block_model(aux_k=0)
    tree = model.initialize(jax.random.PRNGKey(0), np.zeros(D))
    X = jax.random.normal(jax.random.PRNGKey(4), (1, D))
    pre = np.asarray(model.apply({"params": tree}, X,
                                 method=BlockSparse.preacts)).reshape(G, B)
    order = np.argsort(-(pre ** 2).sum(-1))
    grad = jax.grad(lambda p: model.apply(
        {"params": p}, X, jnp.ones(D), jnp.zeros(M, bool),
        method=BlockSparse.loss)[0])(tree)
    moved = np.abs(np.asarray(grad["W_blk"])).sum((0, 2)) > 0
    assert moved[order[:4]].all() and not moved[order[4:]].any()

# `ontologize.sae` (SAE, BilinearSAE) and `ontologize.grouped` (Grouped):
# the SAE baselines as `ontologize` modules. The root sae.py trains them
# through the flat params.npz layout every downstream script reads, so the
# converters between the two layouts carry the whole comparison; the
# behaviour itself is exercised through sae.py in test_sae.py and
# test_gated.py.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.sae import (SAE, BilinearSAE, from_legacy, legacy_encoder,
                            to_legacy)
from ontologize.grouped import Grouped

D, M = 16, 32


def models():
    return [SAE(d=D, m=M, topk=8), BilinearSAE(d=D, m=M, topk=8),
            SAE(d=D, m=M, topk=0, enc="gated"),
            Grouped(d=D, m=M, groups=8),
            Grouped(d=D, m=M, groups=8, group_fn="softmax"),
            Grouped(d=D, m=M, groups=8, enc="bilinear")]


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


def test_resample_refuses_bilinear():
    model = BilinearSAE(d=D, m=M)
    tree = model.initialize(jax.random.PRNGKey(0), np.zeros(D))
    with pytest.raises(ValueError):
        model.resample(tree, (), jnp.zeros((4, D)), jnp.ones(D),
                       np.ones(M, bool), np.random.default_rng(0))

# experiments/planted-hierarchy/tree.py: the planted tree (a child only
# under its own parent, unless shuffled), the share of a factor's entropy
# a head carries, and the probe that reads every prefix of a stack.
import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "planted-hierarchy"))
from tree import (build, explained, explained_given, fvu,  # noqa: E402
                  fvu_given, head_table, placement, plant, run_model, sample,
                  train)


def tree(n: int, shuffle: bool, scale: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    P, C, owner = plant(rng, 11, 3, 3, 32, scale)
    return (P, C, owner) + sample(rng, P, C, owner, n, 0.1, 0.75, shuffle)


def test_a_child_appears_only_under_its_parent():
    P, C, owner, X, zp, zc, clean = tree(20000, shuffle=False)
    assert C.shape == (10, 32) and not C[0].any()
    assert owner.tolist() == [-1, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    has = zc > 0
    assert (owner[zc[has]] == zp[has]).all()           # the hierarchy
    assert abs(has[zp < 3].mean() - 0.75) < 0.02 and not has[zp >= 3].any()
    assert np.allclose(clean, P[zp] + C[zc])
    assert abs((X - clean).std() - 0.1) < 0.005


def test_shuffling_keeps_the_marginals_and_breaks_the_tree():
    _, _, owner, _, zp, zc, _ = tree(20000, shuffle=True)
    _, _, _, _, _, zc0, _ = tree(20000, shuffle=False)
    assert np.array_equal(np.bincount(zc, minlength=10),
                          np.bincount(zc0, minlength=10))
    has = zc > 0
    assert (owner[zc[has]] == zp[has]).mean() < 0.2
    # in the tree the parent alone carries much of the child: whether one
    # is possible. Shuffled, it carries none
    assert explained(zp, zc) < 0.01 < 0.5 < explained(zp, zc0) < 0.7


def test_explained_given_does_not_credit_what_the_parent_implies():
    _, _, _, _, zp, zc, _ = tree(20000, shuffle=False)
    noise = np.random.default_rng(5).integers(0, 16, 20000)
    assert abs(explained_given(zp, zc, zp)) < 1e-9      # the parent itself
    assert np.isclose(explained_given(zc, zc, zp), 1.0)
    assert explained_given(noise, zc, zp) < 0.05
    # which child, within its parent: all of what the parent leaves open
    within = np.where(zc > 0, 1 + (zc - 1) % 3, 0)
    assert np.isclose(explained_given(within, zc, zp), 1.0)


def test_child_scale_scales_the_child_atoms_alone():
    P1, C1, _ = plant(np.random.default_rng(0), 11, 3, 3, 32, 1.0)
    P3, C3, _ = plant(np.random.default_rng(0), 11, 3, 3, 32, 3.0)
    assert np.allclose(P1, P3) and np.allclose(3.0 * C1, C3, rtol=1e-5)


def test_explained_is_one_for_a_copy_and_zero_for_noise():
    rng = np.random.default_rng(1)
    z = rng.integers(0, 6, 5000)
    assert np.isclose(explained((z + 2) % 6, z), 1.0)   # a renaming
    assert np.isclose(explained(z // 2, z), np.log(3) / np.log(6), atol=0.01)
    assert explained(rng.integers(0, 6, 5000), z) < 0.01
    assert explained(z, np.zeros(5000, int)) == 0.0     # a constant factor


def test_placement_names_the_head_that_carries_each_factor():
    _, _, _, _, zp, zc, _ = tree(8000, shuffle=False)
    noise = np.random.default_rng(2).integers(0, 16, 8000)
    A = np.stack([zp, zc, noise])
    table = head_table(A, zp, zc, np.random.default_rng(3))
    assert np.isclose(table[0]["parent"], 1.0)
    assert np.isclose(table[1]["child"], 1.0)
    assert table[0]["child"] > 0.5 and abs(table[0]["child_new"]) < 1e-9
    assert np.isclose(table[1]["child_new"], 1.0)
    assert table[2]["parent"] < 0.02 and table[0]["parent_null"] < 0.02
    assert table[1]["child_new_null"] < 0.1
    where = placement(table, np.array([0, 1, 1]))
    assert where == {"parent_head": 0, "parent_layer": 0,
                     "child_head": 1, "child_layer": 1}


def test_fvu_given_is_the_cell_mean_reconstruction():
    _, _, _, X, zp, zc, clean = tree(8000, shuffle=False)
    assert fvu(X, X) == 0.0
    both = fvu_given(zp * 10 + zc, X)
    assert both < fvu_given(zp, X) < fvu_given(zc, X) < 1.0
    assert abs(both - fvu(clean, X)) < 0.01             # the noise floor


def test_the_probe_reads_every_prefix_of_the_stack():
    cfg = argparse.Namespace(
        d=32, b=64, lr=1e-3, temperature=1.0, temperature_end=0.03,
        anneal_steps=3, steps=4, p_drop=0.1, sd_k=0.02, sd_f=0.1, e_dec=48,
        k=8, h=2, l=2, select="ste", no_deepsup=False)
    _, _, _, X, _, _, _ = tree(256, shuffle=False)
    hyper, model, state = build(cfg, 0)
    state = train(cfg, hyper, state, X, 0)
    Y, A = run_model(model, state.params, X[:64], 0.03)
    assert Y.shape == (2, 64, 32) and A.shape == (4, 64)
    assert A.min() >= 0 and A.max() < 8
    full = model.apply(state.params, jnp.asarray(X[:64]), temperature=0.03)
    assert np.allclose(Y[-1], np.asarray(full), atol=1e-5)
    assert not np.allclose(Y[0], Y[1])

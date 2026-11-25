# experiments/vertices/simplex.py: the planted-simplex toy. The tests check
# the planted mixtures, the ray position, the matching through cells, the
# projection and the archetypal analysis that is the reference arm, the
# phases of the schedule arms, and that switching a model's selection rule
# keeps its parameters.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "vertices"))
from simplex import (archetypes, hull_point, match_cells,  # noqa: E402
                     mixing, mixtures, phases, ray_position, select_at,
                     simplex_project, with_select)


def planted(rng, k=5, d=8):
    return rng.normal(size=(k, d))


def test_mixtures_are_convex_and_concentrate_at_small_alpha():
    rng = np.random.default_rng(0)
    A = planted(rng)
    X, W, clean = mixtures(rng, A, 4000, 0.2, 0.0)
    assert np.allclose(W.sum(1), 1.0) and (W >= 0).all()
    assert np.allclose(clean, W @ A, atol=1e-5) and np.allclose(X, clean)
    _, W_wide, _ = mixtures(rng, A, 4000, 5.0, 0.0)
    assert W.max(1).mean() > 0.6 > W_wide.max(1).mean()


def test_ray_position():
    rng = np.random.default_rng(1)
    A = planted(rng)
    m = A.mean(0)
    lam, err = ray_position(A, A, m)
    assert np.allclose(lam, 1.0) and np.allclose(err, 0.0)
    lam, err = ray_position(m + 1.5 * (A - m), A, m)
    assert np.allclose(lam, 1.5) and np.allclose(err, 0.5)
    lam, _ = ray_position(np.broadcast_to(m, A.shape), A, m)
    assert np.allclose(lam, 0.0)


def test_match_cells_recovers_a_relabelling():
    rng = np.random.default_rng(2)
    dom = rng.integers(0, 6, 3000)
    perm = rng.permutation(6)
    sel = perm[dom]
    sel[:300] = rng.integers(0, 6, 300)      # a tenth of the rows relabelled
    ents, atoms, acc = match_cells(sel, dom, 6, 6)
    assert (ents == perm[atoms]).all()
    assert 0.88 < acc < 0.94


def test_simplex_project_is_sparsemax():
    import jax.numpy as jnp
    from ontologize.fns.classify import sparsemax
    V = np.random.default_rng(3).normal(size=(50, 7)) * 2.0
    assert np.allclose(simplex_project(V), np.asarray(sparsemax(jnp.asarray(V))),
                       atol=1e-5)


def test_mixing_recovers_planted_weights():
    rng = np.random.default_rng(4)
    A = planted(rng)
    X, W, _ = mixtures(rng, A, 500, 0.5, 0.0)
    assert np.abs(mixing(X, A, iters=2000) - W).max() < 1e-3


def test_hull_point():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 4))
    inside = X[:3].mean(0)
    assert np.allclose(hull_point(inside, X), inside, atol=1e-4)
    far = 100.0 * X[0]
    p = hull_point(far, X)
    # the nearest hull point is no farther than any row
    assert np.linalg.norm(p - far) <= np.linalg.norm(X - far, axis=1).min() + 1e-6


def test_archetypes_recover_the_corners():
    rng = np.random.default_rng(6)
    A = planted(rng, k=4, d=6)
    X, W, _ = mixtures(rng, A, 1500, 0.3, 0.0)
    X = np.vstack([X, A])                    # the corners are rows
    Z = archetypes(X, 4, np.random.default_rng(7), iters=200, restarts=2)
    D = np.linalg.norm(Z[:, None] - A[None], axis=-1)
    tol = 0.05 * np.linalg.norm(A - A.mean(0), axis=1).min()
    # every archetype is at a corner, and every corner has an archetype
    assert D.min(1).max() < tol and D.min(0).max() < tol


def test_phases_and_select_at():
    short, long = (0.75, 0.9), (0.6, 0.75)
    ph = phases("schedule_short", short, long)
    assert [select_at(ph, s, 100) for s in (0, 74, 75, 89, 90, 99)] == \
        ["sparsemax", "sparsemax", "ste", "ste", "sparsemax", "sparsemax"]
    assert phases("schedule_long", short, long)[1] == (0.6, "ste")
    for arm in ("sparsemax", "sparsemax_fixed", "sparsemax_setpoint"):
        assert phases(arm, short, long) == [(0.0, "sparsemax")]
    assert phases("ste", short, long) == [(0.0, "ste")]


def small_cfg(**kw):
    import argparse
    cfg = dict(d=8, b=64, lr=1e-3, temperature=1.0, temperature_end=0.03,
               anneal_steps=10, steps=10, sd_k=0.02, sd_f=0.1, p_drop=0.1,
               short=(0.75, 0.9), long=(0.6, 0.75), k=4, h=1, l=1, e_dec=16,
               n=2, deepsup=False, resid_norm=False, no_const0=False,
               min_rows=4)
    cfg.update(kw)
    return argparse.Namespace(**cfg)


def test_evaluate_reads_every_layer_and_the_corners_only_for_one_head():
    from simplex import build, evaluate
    rng = np.random.default_rng(8)
    A = planted(rng, k=4, d=8)
    X, W, clean = mixtures(rng, A, 512, 0.3, 0.05)
    one = evaluate(small_cfg(), build(small_cfg(), "softmax", 0)[1], 1.0,
                   X, W, clean, A)
    assert "lam_vertex" in one and "acc" in one and "L0_atom_over_mu" in one
    cfg = small_cfg(h=2, l=2, deepsup=True, resid_norm=True)
    many = evaluate(cfg, build(cfg, "softmax", 0)[1], 1.0, X, W, clean, A)
    assert "lam_vertex" not in many and "acc" not in many
    assert {"L0_atom_over_mu", "L1_atom_over_mu", "aom_max"} <= set(many)
    assert many["aom_max"] == max(many["L0_atom_over_mu"],
                                  many["L1_atom_over_mu"])


def test_voronoi_logits_pick_the_nearest_atom():
    from simplex import voronoi_logits
    rng = np.random.default_rng(9)
    X, A = rng.normal(size=(300, 5)), rng.normal(size=(6, 5))
    near = np.linalg.norm(X[:, None] - A[None], axis=-1).argmin(1)
    assert (voronoi_logits(X, A, 2.0).argmax(1) == near).all()


def test_a_voronoi_head_at_the_corners_reads_them():
    from simplex import evaluate_voronoi
    rng = np.random.default_rng(10)
    A = planted(rng, k=4, d=8) * 3.0
    X, W, clean = mixtures(rng, A, 2000, 0.05, 0.01)
    r = evaluate_voronoi(small_cfg(), "ste", A, 1.0, 1.0, X, W, clean, A)
    assert np.isclose(r["lam_vertex"], 1.0) and np.isclose(r["err_vertex"], 0)
    assert r["acc"] > 0.95 and r["support"] == 1.0 and r["decided"] == 1.0
    # the cells' centroids sit inside their corners
    assert r["lam_centroid"] < 1.0


def test_the_voronoi_ste_moves_its_atoms_to_the_cluster_means():
    # from one displaced atom per cluster; like k-means it has local minima
    # from other starts, so a start that misses a cluster is not tested
    from simplex import train_voronoi
    rng = np.random.default_rng(11)
    C = rng.normal(size=(4, 6)) * 4.0
    X = (C[rng.integers(0, 4, 4096)]
         + 0.1 * rng.normal(size=(4096, 6))).astype(np.float32)
    cfg = small_cfg(d=6, k=4, b=128, steps=1500, anneal_steps=1000,
                    voronoi_init="rows", voronoi_beta=1.0, lr=3e-2)
    A0 = C + 1.0 * rng.normal(size=C.shape)
    A, _, _ = train_voronoi(cfg, "ste", X, 0, A0=A0)
    D = np.linalg.norm(A[:, None] - C[None], axis=-1)
    assert np.linalg.norm(A0 - C, axis=1).min() > 1.0
    assert D.min(0).max() < 0.1 and D.min(1).max() < 0.1


def test_with_select_keeps_the_parameters():
    import jax
    from ontologize.training.config import Hyperparams
    hyper = Hyperparams(8, 8, 16, ghost=False)
    model = hyper.ontologizer(8, 8, 8, 4, 1, 1, n=2, gate="none",
                              select="sparsemax", forward="resid",
                              deepsup=False, dtype_str="float32",
                              dtype_p_str="float32")
    state = hyper.init(model, save_each=4)
    hard = with_select(state, "ste")
    assert hard.model.select == "ste" and state.model.select == "sparsemax"
    same = jax.tree_util.tree_map(lambda a, b: bool((a == b).all()),
                                  state.params, hard.params)
    assert all(jax.tree_util.tree_leaves(same))
    X = jax.random.normal(jax.random.PRNGKey(0), (16, 8))
    Y, _, _, _ = hard.apply_fn(hard.params, X, temperature=0.5)
    assert Y.shape == (16, 8)

# experiments/seed-stability/kmeans_seeds.py: k-means seed agreement. The
# tests pin the assignment against brute force, and the agreement on three
# planted cases: separated clusters and a dominant direction reproduce
# across seeds, and an isotropic Gaussian, whose cells can rotate at no
# cost, does not.
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "seed-stability"))
from kmeans_seeds import (assign, energy_dependence, energy_groups,  # noqa: E402
                          fastica, fvu, group_copy, ica_frame, kmeans,
                          marginal_copy, principal, scale_copy, seed_table,
                          shape_stats, to_sources, train_index, trimmed_rows)


def test_assign_is_the_nearest_centre():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6)).astype(np.float32)
    C = rng.normal(size=(7, 6)).astype(np.float32)
    brute = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
    assert np.array_equal(assign(X, C, chunk=128), brute)


def test_separated_clusters_reproduce_and_fit():
    rng = np.random.default_rng(1)
    centres = rng.normal(size=(4, 8)) * 20
    X = (centres[rng.integers(4, size=6000)] + rng.normal(size=(6000, 8))
         ).astype(np.float32)
    t = seed_table(X[:4000], X[4000:], 4, [1, 2, 3])
    assert min(t["nmi"]) > 0.95
    assert max(t["null"]) < 0.01
    assert max(t["fvu"]) < 0.05


def test_a_dominant_direction_pins_a_two_cell_split():
    rng = np.random.default_rng(2)
    X = (rng.normal(size=(6000, 8)) * np.r_[10.0, np.ones(7)]).astype(np.float32)
    t = seed_table(X[:4000], X[4000:], 2, [1, 2, 3])
    assert min(t["nmi"]) > 0.9


def test_an_isotropic_gaussian_does_not_reproduce():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12000, 8)).astype(np.float32)
    t = seed_table(X[:8000], X[8000:], 8, [1, 2, 3])
    sep = rng.normal(size=(8, 8)) * 20
    Y = (sep[rng.integers(8, size=12000)] + rng.normal(size=(12000, 8))
         ).astype(np.float32)
    u = seed_table(Y[:8000], Y[8000:], 8, [1, 2, 3])
    assert np.mean(t["nmi"]) < np.mean(u["nmi"]) - 0.2
    assert np.mean(t["nmi"]) > 10 * np.mean(t["null"])


def test_spherical_kmeans_follows_direction_and_trimming_drops_the_far_rows():
    # four directions, each row at a length spread over a factor of ~20:
    # Euclidean k-means spends its cells on the long rows and lumps most of
    # the short ones into one cell, while spherical k-means finds the rays
    rng = np.random.default_rng(13)
    n, d = 8000, 8
    dirs = np.linalg.qr(rng.standard_normal((d, 4)))[0].T * 8
    lab0 = rng.integers(4, size=n)
    r = np.exp(rng.standard_normal(n))[:, None]
    X = (r * (dirs[lab0] + rng.standard_normal((n, d)))).astype(np.float32)
    from partition import nmi_matrix
    t = seed_table(X[:6000], X[6000:], 4, [1, 2, 3], method="spherical")
    for lab in t["labels"]:
        assert nmi_matrix(lab[None], 4, lab0[6000:][None], 4)[0, 0] > 0.95
    assert min(t["nmi"]) > 0.95 and max(t["largest_cell"]) < 0.35
    e = seed_table(X[:6000], X[6000:], 4, [1, 2, 3])
    assert min(e["largest_cell"]) > 0.8
    keep = trimmed_rows(X, 0.9)
    rad = np.linalg.norm(X - X.mean(0), axis=1)
    assert abs(len(keep) / n - 0.9) < 0.01 and rad[keep].max() <= np.quantile(rad, 0.9)
    # the default path is unchanged: no trimming gives the plain run
    assert np.array_equal(kmeans(X, 4, 5), kmeans(X, 4, 5, seed_rows=None))
    assert seed_table(X[:6000], X[6000:], 4, [1], trim=0.9)["fvu"][0] < 1


def test_kmeans_ends_with_every_cell_used():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 3)).astype(np.float32)
    C = kmeans(X, 20, 0)
    lab = assign(X, C)
    assert len(np.unique(lab)) == 20
    assert 0 < fvu(X, C, lab) < 1


def _square_dependence(Y):
    """The largest |corr(Y_i², Y_j)| over i != j."""
    C = np.corrcoef(np.concatenate([Y ** 2, Y], 1).T)[:Y.shape[1], Y.shape[1]:]
    return np.abs(C - np.diag(np.diag(C))).max()


def test_the_pca_copy_keeps_each_principal_marginal_and_breaks_dependence():
    # z2 is a function of z1 yet uncorrelated with it, and the variances
    # differ, so the principal frame is z's own up to a rotation
    rng = np.random.default_rng(5)
    z1 = 2 * rng.standard_normal(8000)
    Z = np.stack([z1, (z1 ** 2 - 4) / 4, 0.5 * rng.standard_normal(8000)], 1)
    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    X = (Z @ R + 1.0).astype(np.float32)
    T, E, st = marginal_copy(X[:6000], X[6000:], "pca")
    mu, lam, V = principal(X[:6000])
    for orig, copy in ((X[:6000], T), (X[6000:], E)):
        Y, Yc = (orig - mu) @ V, (copy - mu) @ V
        assert np.allclose(np.sort(Y, 0), np.sort(Yc, 0), atol=1e-4)
        assert _square_dependence(Y) > 0.9 > 0.1 > _square_dependence(Yc)
    assert np.allclose(np.cov(T.T), np.cov(X[:6000].T), rtol=0.05, atol=0.05)
    assert np.allclose(st["variance_share"], lam / lam.sum())


def test_fastica_recovers_planted_sources():
    rng = np.random.default_rng(6)
    n = 20000
    S = np.stack([rng.uniform(-1, 1, n), rng.laplace(size=n),
                  rng.exponential(size=n), rng.uniform(-1, 1, n) ** 3], 1)
    S = (S - S.mean(0)) / S.std(0)
    X = S @ rng.standard_normal((4, 4)) + 3.0
    mu, lam, V = principal(X)
    W, it, lim = fastica((X - mu) @ (V / np.sqrt(lam)), seed=0)
    Y = (X - mu) @ (V / np.sqrt(lam)) @ W.T
    C = np.abs(np.corrcoef(np.concatenate([Y, S], 1).T)[:4, 4:])
    assert lim < 1e-4 and it < 200
    assert (C.max(0) > 0.98).all() and (C.max(1) > 0.98).all()
    # started from its own fixed point, it stops at once and stays there
    W2, it2, lim2 = fastica((X - mu) @ (V / np.sqrt(lam)), seed=1, W0=W)
    assert it2 <= 2 and lim2 < 1e-4 and np.abs(np.abs((W2 * W).sum(1)) - 1).max() < 1e-3
    # damped steps, per row or shared, reach the same sources, and the
    # checkpoint sees the run and its steps
    Zs = (X - mu) @ (V / np.sqrt(lam))
    for shared in (False, True):
        seen, steps = [], []
        W3, it3, lim3 = fastica(Zs, seed=2, iters=2000, patience=1, shared=shared,
                                checkpoint_every=2,
                                checkpoint=lambda W, i, l, s: (seen.append(i),
                                                               steps.append(s)))
        Y3 = Zs @ W3.T
        C3 = np.abs(np.corrcoef(np.concatenate([Y3, S], 1).T)[:4, 4:])
        assert lim3 < 1e-4 and (C3.max(0) > 0.98).all()
        assert seen == list(range(2, it3, 2))
        assert np.shape(steps[-1]) == (() if shared else (4,))
    # a continued per-row fit takes its steps back; a shared step needs one
    W4, it4, _ = fastica(Zs, seed=3, W0=W3, patience=1, step0=np.full(4, 0.5))
    assert it4 <= 2
    with pytest.raises(ValueError):
        fastica(Zs, seed=3, patience=1, shared=True, step0=np.full(4, 0.5))


def test_an_ica_fit_continues_only_on_its_own_rows():
    rng = np.random.default_rng(9)
    S = rng.laplace(size=(12000, 3))
    X = (S @ rng.standard_normal((3, 3))).astype(np.float32)
    _, _, st = marginal_copy(X[:10000], X[10000:], "ica", ica_iters=3)
    assert {"mean", "eigenvalues", "eigenvectors", "unmixing"} <= set(st["frame"])
    _, _, st2 = marginal_copy(X[:10000], X[10000:], "ica", ica_init=st["frame"])
    assert st2["ica_last_change"] < 1e-4
    # a continued fit counts the earlier one's iterations and keeps its step
    fit = ica_frame(X[:10000], init={**st["frame"], "step": 0.25}, patience=5)
    assert fit["iterations"] > st["frame"]["iterations"] == 3
    # with no iterations, the saved frame is used as it is
    same = ica_frame(X[:10000], iters=0, init=fit)
    assert np.array_equal(same["unmixing"], fit["unmixing"])
    assert (same["iterations"], same["last_change"]) == (fit["iterations"], fit["last_change"])
    with pytest.raises(ValueError):
        ica_frame(X[2000:], iters=0, init=fit)
    with pytest.raises(ValueError):
        marginal_copy(X[2000:], X[:2000], "ica", ica_init=st["frame"])


def test_the_ica_copy_keeps_an_independent_model_the_pca_copy_breaks():
    # two uniform sources along non-orthogonal directions: the rows fill a
    # parallelogram. Shuffling the sources keeps it; shuffling principal
    # coordinates, which mix the sources, spills rows outside it
    rng = np.random.default_rng(7)
    n = 20000
    S = rng.uniform(-np.sqrt(3), np.sqrt(3), (n, 2))
    A = np.array([[1.0, 0.0], [0.8, 0.6]])
    X = (S @ A).astype(np.float32)
    edge = {}
    for frame in ("pca", "ica"):
        T, E, st = marginal_copy(X[:15000], X[15000:], frame)
        edge[frame] = np.abs(T.astype(np.float64) @ np.linalg.inv(A)).max()
    assert edge["ica"] < np.sqrt(3) + 0.05
    assert edge["pca"] > np.sqrt(3) + 0.3
    assert st["ica_last_change"] < 1e-4


def planted_scale_groups(n, rng):
    """Six Laplace sources mixed at random; sources 0-2 share one scale
    variable (their magnitudes rise together), 3-5 are independent."""
    S = rng.laplace(size=(n, 6))
    S[:, :3] *= np.exp(rng.standard_normal((n, 1)))
    S = (S - S.mean(0)) / S.std(0)
    return S, (S @ rng.standard_normal((6, 6))).astype(np.float32)


def test_magnitude_correlations_find_sources_that_share_a_scale():
    rng = np.random.default_rng(10)
    S, _ = planted_scale_groups(40000, rng)
    R = energy_dependence(S)
    assert R[0, 1] > 0.2 and R[1, 2] > 0.2
    assert np.abs(R[3:, 3:][~np.eye(3, dtype=bool)]).max() < 0.03
    assert np.abs(R[:3, 3:]).max() < 0.03
    g = energy_groups(R, 0.1)
    assert len(set(g[:3])) == 1 and len(set(g[3:])) == 3 and g[3] != g[0]


def test_the_group_copy_keeps_a_group_and_breaks_ties_between_groups():
    rng = np.random.default_rng(11)
    _, X = planted_scale_groups(40000, rng)
    fit = ica_frame(X[:30000], iters=500)
    assert fit["last_change"] < 1e-4
    S = to_sources(X[:30000], fit)
    R = energy_dependence(S)
    g = energy_groups(R, 0.1)
    T, E = group_copy(X[:30000], X[30000:], fit, g)
    Rt = energy_dependence(to_sources(T, fit))
    shared = np.flatnonzero(np.bincount(g)[g] == 3)
    assert len(shared) == 3
    a, b = shared[:2]
    assert abs(Rt[a, b] - R[a, b]) < 0.03                  # kept inside the group
    other = np.flatnonzero(np.bincount(g)[g] == 1)[0]
    assert abs(Rt[a, other]) < 0.03
    singles = group_copy(X[:30000], X[30000:], fit, np.arange(6))[0]
    assert energy_dependence(to_sources(singles, fit))[a, b] < 0.03
    assert np.allclose(to_sources(T, fit).std(0), S.std(0), rtol=1e-6)


def _mean_offdiag(R):
    return R[~np.eye(len(R), dtype=bool)].mean()


def test_the_scale_copy_keeps_a_shared_scale_the_ica_copy_breaks():
    # 128 Laplace sources mixed at random, all multiplied by one scale per
    # row; then the same without the scale. Binning rows by a scale
    # estimated from their own sources adds a magnitude correlation of about
    # 0.75/p (0.024 at 32)
    rng = np.random.default_rng(12)
    n, p = 40000, 128
    A = rng.standard_normal((p, p))
    for shared in (True, False):
        S = rng.laplace(size=(n, p))
        if shared:
            S *= np.exp(0.5 * rng.standard_normal((n, 1)))
        X = ((S / S.std(0)) @ A).astype(np.float32)
        fit = ica_frame(X[:30000], iters=300)
        Y = to_sources(X[:30000], fit)
        T, E = scale_copy(X[:30000], X[30000:], fit)
        Yt = to_sources(T, fit)
        R, Rt = energy_dependence(Y), energy_dependence(Yt)
        Ri = energy_dependence(to_sources(
            marginal_copy(X[:30000], X[30000:], "ica", ica_fit=fit)[0], fit))
        if shared:
            assert _mean_offdiag(R) > 0.1
            assert abs(_mean_offdiag(Rt) - _mean_offdiag(R)) < 0.015
            assert abs(_mean_offdiag(Ri)) < 0.01
        else:
            assert abs(_mean_offdiag(R)) < 0.01 and abs(_mean_offdiag(Rt)) < 0.015
        # each source keeps its values, and the tail is copied on its own
        assert np.allclose(np.sort(Yt, 0), np.sort(Y, 0), atol=1e-4)
        assert E.shape == X[30000:].shape


def test_the_training_rows_are_drawn_as_before_and_a_second_draw_avoids_them():
    a = train_index(1000, 100)
    assert np.array_equal(a, np.sort(np.random.default_rng(0).choice(1000, 100, replace=False)))
    b = train_index(1000, 100, seed=1, exclude=a)
    assert len(np.intersect1d(a, b)) == 0 and len(b) == 100


def test_shape_stats_read_gaussian_uniform_and_two_modes():
    rng = np.random.default_rng(8)
    n = 200000
    Y = np.stack([rng.standard_normal(n), rng.uniform(size=n),
                  rng.standard_normal(n) + 4 * rng.choice([-1, 1], n)], 1)
    st = shape_stats(Y)
    assert abs(st["kurtosis"][0]) < 0.05 and abs(st["bimodality"][0] - 1 / 3) < 0.01
    assert abs(st["kurtosis"][1] + 1.2) < 0.02 and abs(st["bimodality"][1] - 5 / 9) < 0.01
    assert st["bimodality"][2] > 0.8

"""headcoh.py pure stats: planted coherent groups must z-score high,
arbitrary groupings must not, and everything must be deterministic."""
import numpy as np

from headcoh import (group_zscores, mean_pairwise_cos, perm_z,
                     pooled_within_cos, sv_energy, unit)


def _planted(n_groups=8, size=6, d=32, noise=0.3, seed=0):
    """Rows = group centroid + noise: semantically coherent groups."""
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(n_groups, d))
    R = np.repeat(C, size, 0) + noise * rng.normal(size=(n_groups * size, d))
    labels = np.repeat(np.arange(n_groups), size)
    return R, labels


def test_mean_pairwise_cos_matches_naive():
    rng = np.random.default_rng(1)
    R = rng.normal(size=(7, 5))
    Rh = unit(R)
    naive = np.mean([Rh[i] @ Rh[j] for i in range(7) for j in range(7)
                     if i != j])
    assert abs(mean_pairwise_cos(R) - naive) < 1e-9


def test_sv_energy_extremes():
    row = np.random.default_rng(2).normal(size=8)
    assert abs(sv_energy(np.stack([row, 2 * row, -3 * row])) - 1.0) < 1e-6
    assert abs(sv_energy(np.eye(8)) - 1 / 8) < 1e-9


def test_planted_groups_score_high_random_do_not():
    R, labels = _planted()
    pools = np.zeros(len(R), int)
    rows = group_zscores(R, labels, pools, n_null=200, seed=42)
    assert len(rows) == 8
    assert all(r["z_cos"] > 3 for r in rows)
    assert all(r["z_sv"] > 3 for r in rows)

    shuffled = np.random.default_rng(3).permutation(labels)
    rows_r = group_zscores(R, shuffled, pools, n_null=200, seed=42)
    z = np.array([r["z_cos"] for r in rows_r])
    assert abs(z.mean()) < 1.0  # arbitrary partition: no coherence signal


def test_group_zscores_pools_and_small_groups():
    R, labels = _planted(n_groups=4, size=6)
    pools = np.repeat([0, 0, 1, 1], 6)
    labels = labels.copy()
    labels[:2] = -1  # ungrouped rows are ignored
    rows = group_zscores(R, labels, pools, n_null=100, seed=0)
    assert {r["pool"] for r in rows} == {0, 1}
    assert all(r["size"] in (4, 6) for r in rows)


def test_perm_z_planted_vs_shuffled():
    R, labels = _planted(noise=0.4, seed=5)
    pools = np.zeros(len(R), int)
    obs, z = perm_z(R, labels, pools, n_perm=300, seed=7)
    assert obs > 0.3 and z > 3

    shuffled = np.random.default_rng(8).permutation(labels)
    _, z_r = perm_z(R, shuffled, pools, n_perm=300, seed=7)
    assert abs(z_r) < 2.5


def test_pooled_within_cos_skips_singletons():
    E = np.eye(4)
    labels = np.array([0, 0, 1, -1])
    # only the (0,1) pair counts; orthogonal rows -> 0
    assert abs(pooled_within_cos(E, labels)) < 1e-6


def test_determinism():
    R, labels = _planted()
    pools = np.zeros(len(R), int)
    a = group_zscores(R, labels, pools, n_null=50, seed=9)
    b = group_zscores(R, labels, pools, n_null=50, seed=9)
    assert a == b

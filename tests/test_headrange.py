# headrange.py's range-coverage test on planted codes: independent heads
# leave no structurally empty cell and match the shuffle null, an entry pair
# planted never to co-occur is the empty cell found, and the overlap
# contrast reads atoms that share coordinates only where entries exclude.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"
                       / "ste-arm"))
from headrange import (expected, layer_measure, overlap_stats,  # noqa: E402
                       pair_table, structural)


def test_pair_table_counts_joint_winners():
    a = np.array([0, 0, 1, 2, 2, 2])
    b = np.array([1, 1, 0, 2, 0, 2])
    C = pair_table(a, b, 3)
    assert C.sum() == 6
    assert C[0, 1] == 2 and C[2, 2] == 2 and C[1, 0] == 1 and C[2, 0] == 1


def test_independent_heads_have_no_structural_zeros():
    rng = np.random.default_rng(0)
    A = rng.integers(0, 4, (20000, 3))
    r = layer_measure(A, None, 4, 10.0, 2, rng)
    assert r["pairs"] == 3 and r["tested_cells"] == 3 * 16
    assert r["empty_cells"] == 0 and r["null_frac"] == 0
    assert r["pairs_full_coverage"] == 1.0


def test_planted_exclusion_is_the_empty_cell():
    rng = np.random.default_rng(1)
    n, k = 20000, 4
    a = rng.integers(0, k, n)
    b = rng.integers(0, k, n)
    b[(a == 0) & (b == 0)] = 1                # entry 0 of each never together
    f_a = np.bincount(a, minlength=k) / n
    f_b = np.bincount(b, minlength=k) / n
    t, z = structural(pair_table(a, b, k), expected(f_a, f_b, n), 10.0)
    assert t.all()
    assert z.sum() == 1 and z[0, 0]
    r = layer_measure(np.stack([a, b], 1), None, k, 10.0, 3, rng)
    assert r["empty_cells"] == 1 and r["null_frac"] == 0
    assert r["pairs_full_coverage"] == 0.0


def test_overlap_contrast_reads_shared_coordinates_on_empty_cells():
    rng = np.random.default_rng(2)
    n, k, e = 20000, 4, 64
    a = rng.integers(0, k, n)
    b = rng.integers(0, k, n)
    b[(a == 0) & (b == 0)] = 1
    # disjoint random supports, except that the excluded pair shares one
    U = np.zeros((2, k, e))
    for h in range(2):
        for j in range(k):
            U[h, j, (h * k + j) * 8:(h * k + j) * 8 + 8] = 1.0
    U[1, 0] = U[0, 0]
    U /= np.linalg.norm(U, axis=-1, keepdims=True)
    r = layer_measure(np.stack([a, b], 1), U, k, 10.0, 1, rng)
    o = r["overlap"]
    assert o["empty_cells"] == 1 and np.isclose(o["excess_cos"], 1.0)


def test_overlap_stats_is_bin_matched():
    # same excess in each bin, at different co-occurring baselines
    cells = {0: {"z_n": 2, "z_s": 2 * 0.5, "c_n": 100, "c_s": 100 * 0.3,
                 "c_ss": 100 * (0.3 ** 2 + 0.01)},
             1: {"z_n": 1, "z_s": 0.9, "c_n": 50, "c_s": 50 * 0.7,
                 "c_ss": 50 * (0.7 ** 2 + 0.01)}}
    o = overlap_stats(cells)
    assert o["empty_cells"] == 3
    assert np.isclose(o["excess_cos"], 0.2)
    assert np.isclose(o["z"], 0.6 / np.sqrt(3 * 0.01))

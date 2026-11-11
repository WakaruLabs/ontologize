# experiments/pair-diff: codes of the difference of two sentences from one
# document. The tests pin what the data source serves, the differences'
# whitening, and evaldiff.py's seed-match and antipodal statistics.
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "pair-diff"))
from diffs import DiffSource, diff_weights, differences  # noqa: E402
from evaldiff import chance_keep, entropy_bits, seed_match  # noqa: E402


def cache(tmp_path, n=40, d=6):
    X = np.random.default_rng(0).normal(size=(n, d)).astype(np.float32)
    path = tmp_path / "cache.npy"
    np.save(path, X)
    return path, X


def test_source_serves_row_minus_partner(tmp_path):
    path, X = cache(tmp_path)
    rows, partners = np.array([3, 7, 7, 0]), np.array([5, 2, 9, 39])
    src = DiffSource(path, rows, partners)
    assert len(src) == 4
    for m in range(4):
        assert np.allclose(src[m], X[rows[m]] - X[partners[m]])
    src2 = pickle.loads(pickle.dumps(src))       # after a read, as a worker
    assert src2._arr is None and np.allclose(src2[1], src[1])
    try:
        DiffSource(path, rows, partners[:2])
        assert False, "mismatched lengths should raise"
    except ValueError:
        pass


def test_differences_matches_the_source_in_any_row_order(tmp_path):
    path, X = cache(tmp_path)
    rng = np.random.default_rng(1)
    rows, partners = rng.integers(0, 40, 25), rng.integers(0, 40, 25)
    arr = np.load(path, mmap_mode="r")
    D = differences(arr, rows, partners, chunk=7)   # unsorted, repeated rows
    assert np.allclose(D, X[rows] - X[partners])


def test_diff_weights_whiten_and_keep_the_mask_and_scale():
    rng = np.random.default_rng(2)
    sd = np.array([1.0, 2.0, 4.0, 0.5, 3.0])
    D = (rng.normal(size=(20000, 5)) * sd).astype(np.float32)
    base = np.array([0.2, 0.1, 0.0, 0.4, 0.3], np.float32)
    w = diff_weights(D, base)
    assert w[2] == 0.0
    assert np.isclose(w.sum(), base.sum(), rtol=1e-5)
    live = base > 0
    scaled = w[live] * sd[live] ** 2             # inverse variance: constant
    assert np.allclose(scaled, scaled[0], rtol=0.05)


def test_seed_match_finds_a_permuted_copy():
    rng = np.random.default_rng(3)
    s = 10
    perm = rng.permutation(s)
    M = 0.02 * rng.random((s, s))
    M[np.arange(s), perm] = 0.5                  # A's head i is B's perm[i]
    Mn = 0.02 * rng.random((s, s))
    m = seed_match(M, Mn)
    assert m["heads"] == s and np.isclose(m["matched"], 0.5)
    assert m["null"] < 0.03 and m["all_pairs"] < 0.1
    assert m["margin"] > 0.45
    # near-copies of one head match as well under any pairing
    flat = seed_match(np.full((s, s), 0.3), Mn)
    assert np.isclose(flat["matched"], flat["all_pairs"])
    assert flat["margin"] == 0.0


def test_chance_keep_and_entropy():
    A = np.array([[0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 2, 3]])
    B = np.array([[0, 1, 0, 1], [1, 1, 1, 1], [0, 1, 2, 3]])
    assert np.allclose(chance_keep(A, B, 4), [0.5, 0.0, 0.25])
    assert np.allclose(entropy_bits(A, 4), [1.0, 0.0, 2.0])

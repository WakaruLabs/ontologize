# experiments/ste-arm/partition.py: the pairwise NMI between two models'
# head partitions. The contraction is the whole cost of that analysis --
# `np.einsum` without `optimize` runs its own nested loop instead of
# reshaping to a matmul, which at the real size is 81.8s per chunk
# against 0.9s. These pin the numbers so the fast path cannot drift from
# the obvious one, and so the flag cannot be dropped silently.
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "ste-arm"))
from partition import nmi_matrix   # noqa: E402


def naive(A, ka, B, kb):
    """Per-pair NMI, written the obvious way, as the reference."""
    out = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            C = np.zeros((ka, kb))
            np.add.at(C, (a, b), 1.0)
            C /= C.sum()
            pa, pb = C.sum(1), C.sum(0)
            nz = C > 0
            mi = (C[nz] * np.log(C[nz] / np.outer(pa, pb)[nz])).sum()
            h = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
            out[i, j] = mi / max(0.5 * (h(pa) + h(pb)), 1e-12)
    return out


@pytest.fixture(scope="module")
def labels():
    rng = np.random.default_rng(0)
    return (rng.integers(0, 6, size=(5, 4000)),
            rng.integers(0, 6, size=(7, 4000)))


def test_matches_the_naive_reference(labels):
    A, B = labels
    assert np.allclose(nmi_matrix(A, 6, B, 6), naive(A, 6, B, 6), atol=1e-5)


def test_chunking_does_not_change_the_answer(labels):
    A, B = labels
    ref = nmi_matrix(A, 6, B, 6, chunk=32)
    for c in (1, 2, 3):
        assert np.allclose(nmi_matrix(A, 6, B, 6, chunk=c), ref, atol=1e-6)


def test_identical_partitions_score_one(labels):
    A, _ = labels
    assert np.allclose(np.diag(nmi_matrix(A, 6, A, 6)), 1.0, atol=1e-6)


def test_relabeling_a_partition_does_not_change_it(labels):
    """NMI is the point of using it: a head that learned the same split
    under different entry numbers must still match."""
    A, _ = labels
    perm = np.random.default_rng(1).permutation(6)
    assert np.allclose(np.diag(nmi_matrix(A, 6, perm[A], 6)), 1.0, atol=1e-6)


def test_independent_partitions_score_near_zero(labels):
    A, B = labels
    # the finite-sample floor, not exactly 0: at 4000 rows and k=6 the
    # shuffled null in the live analysis sits around 0.004
    assert nmi_matrix(A, 6, B, 6).mean() < 0.02


def test_a_head_that_never_moves_is_finite():
    """A constant head has zero entropy, so the normalizer is zero on
    that side and the guard has to hold."""
    rng = np.random.default_rng(2)
    A = np.stack([rng.integers(0, 6, 2000), np.zeros(2000, int)])
    M = nmi_matrix(A, 6, A, 6)
    assert np.all(np.isfinite(M))

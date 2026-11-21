# experiments/seed-stability/kmeans_structure.py: residual k-means and the
# helpers of the additive arm (whose solver, FAISS's LSQ, is not a project
# dependency). The tests pin the cascade's falling prefix error and a shared
# first stage, the matching (a relabelled, reordered copy reads 1,
# independent codes near 0), and the unpacking of FAISS's packed codes
# (least significant bit first, codes back to back across byte boundaries).
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "seed-stability"))
from kmeans_structure import matched_nmi, rvq_assign, rvq_fit, unpack  # noqa: E402


def test_rvq_prefix_error_falls_and_a_shared_stage_is_kept():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 6)).astype(np.float32)
    Cs = rvq_fit(X, 8, 3, seed=1, iters=50)
    labs, fv = rvq_assign(X, Cs)
    assert labs.shape == (3, 3000)
    assert fv[0] > fv[1] > fv[2]
    shared = rvq_fit(X, 8, 3, seed=2, first=Cs[0], iters=50)
    assert np.array_equal(shared[0], Cs[0])


def test_matched_nmi_reads_a_relabelled_copy_and_noise():
    rng = np.random.default_rng(1)
    A = rng.integers(0, 8, (3, 5000))
    B = np.stack([np.argsort(rng.permutation(8))[A[i]] for i in (2, 0, 1)])
    assert abs(matched_nmi(A, B, 8)[0] - 1) < 1e-6
    assert matched_nmi(A, rng.integers(0, 8, (3, 5000)), 8)[0] < 0.01


def test_unpack_reads_codes_packed_lsb_first_across_bytes():
    rng = np.random.default_rng(2)
    heads, nbits, n = 3, 5, 100
    want = rng.integers(0, 1 << nbits, (heads, n))
    bits = ((want.T[:, :, None] >> np.arange(nbits)) & 1).reshape(n, heads * nbits)
    pad = (-bits.shape[1]) % 8
    bits = np.concatenate([bits, np.zeros((n, pad), bits.dtype)], 1)
    codes = np.packbits(bits.astype(np.uint8), axis=1, bitorder="little")
    assert np.array_equal(unpack(codes, heads, nbits), want)

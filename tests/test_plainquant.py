# experiments/rate-penalty/plainquant.py: the helpers around FAISS's
# quantizers (FAISS itself is not a project dependency). The tests pin the
# FVU at its two ends and the residual code's prefixes, which must be the
# partial sums of the first m codebooks' chosen entries.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "rate-penalty"))
sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "seed-stability"))
from plainquant import fvu, rq_prefix  # noqa: E402


def test_fvu_ends():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(500, 6)).astype(np.float32)
    assert fvu(Z, Z.astype(np.float64)) == 0
    assert abs(fvu(Z, np.broadcast_to(Z.mean(0), Z.shape)) - 1) < 1e-9


def test_rq_prefix_is_the_partial_sum_of_chosen_entries():
    rng = np.random.default_rng(1)
    m, nbits, n, d = 4, 5, 50, 3
    books = rng.normal(size=(m, 1 << nbits, d))
    lab = rng.integers(0, 1 << nbits, (m, n))
    bits = ((lab.T[:, :, None] >> np.arange(nbits)) & 1).reshape(n, m * nbits)
    bits = np.concatenate([bits, np.zeros((n, (-bits.shape[1]) % 8), bits.dtype)], 1)
    codes = np.packbits(bits.astype(np.uint8), axis=1, bitorder="little")
    out = rq_prefix(codes, books, nbits, [2, 4])
    want2 = books[0][lab[0]] + books[1][lab[1]]
    assert np.allclose(out[2], want2)
    assert np.allclose(out[4], want2 + books[2][lab[2]] + books[3][lab[3]])

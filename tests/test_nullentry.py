# experiments/ste-arm/nullentry.py: whether a head spends an entry on
# nothing. A used entry with a near-zero atom is quiet; a near-zero atom
# that is rarely selected is not, nor is a head whose atoms are all small;
# an atom at the head's mean write is quiet only when measured centered.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "ste-arm"))
from nullentry import (concentration, entry_table, head_ratio,  # noqa: E402
                       quiet, top_share)


def planted(rng: np.random.Generator):
    H, k, d, n = 4, 8, 16, 8000
    D = rng.standard_normal((H, k, d))
    D[0, 3] *= 0.01                     # head 0: a used, near-zero atom
    D[1, 5] *= 0.01                     # head 1: near-zero but rare
    D[2] *= 0.01                        # head 2: every atom small
    I = rng.integers(0, k, size=(H, n))
    I[1][I[1] == 5] = 0                 # head 1 never selects entry 5 ...
    I[1, :10] = 5                       # ... except on ten rows
    # head 3: atoms around a large shared offset whose spreads sum to zero,
    # so the head's mean write is the offset; entry 2 writes exactly that
    off = 10 * rng.standard_normal(d)
    S = rng.standard_normal((k, d))
    S[2] = 0
    S[np.arange(k) != 2] -= S[np.arange(k) != 2].mean(0)
    D[3] = off + S
    return D, I


def test_used_near_zero_atom_is_quiet():
    D, I = planted(np.random.default_rng(0))
    t = entry_table(D, I)
    q = quiet(t["norm"], t["usage"], 0.25, 0.25)
    assert q[0].tolist() == [e == 3 for e in range(8)]
    assert not q[1].any()               # rare, so not spent on anything
    assert not q[2].any()               # small throughout, not abstaining


def test_centering_finds_an_entry_at_the_heads_mean():
    D, I = planted(np.random.default_rng(1))
    t = entry_table(D, I)
    assert not quiet(t["norm"], t["usage"], 0.25, 0.25)[3].any()
    qc = quiet(t["cnorm"], t["usage"], 0.25, 0.25)[3]
    assert qc.tolist() == [e == 2 for e in range(8)]


def test_head_ratio_is_against_the_heads_used_median():
    norm = np.array([[1.0, 2.0, 3.0, 100.0]])
    used = np.array([[True, True, True, False]])
    np.testing.assert_allclose(head_ratio(norm, used), norm / 2.0)


def test_a_one_entry_heads_centered_ratio_is_undefined_not_an_error():
    cnorm = np.array([[0.0, 1.0]])                 # its only used entry
    used = np.array([[True, False]])
    assert np.isnan(head_ratio(cnorm, used)).all()


def test_token_concentration_reads_one_for_random_rows():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 50, 20000)             # top 10 hold ~a fifth
    word = tokens == 7                              # one token's rows
    rand = rng.random(len(tokens)) < word.mean()    # as many, at random
    assert top_share(tokens[word]) == 1.0
    assert concentration(tokens, word, rng) > 1.5
    assert abs(concentration(tokens, rand, rng) - 1) < 0.1

# experiments/entry-force/force.py: forcing one entry and reading the
# decodes' script. The tests pin the pure helpers: the script share counts
# letters only, the null pairing never pairs a row with itself, ties rank
# against the entry under test, and the forced rows avoid the script's own
# language and stay in the tail.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "entry-force"))
from force import (SCRIPTS, derangement, pick_rows, rank_of,  # noqa: E402
                   script_share)

from scan import affine, auroc, log_odds  # noqa: E402

MYA = SCRIPTS["mya"]


def test_auroc():
    y = np.array([False, False, True, True])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert auroc(np.array([0.1, 0.9, 0.2, 0.8]), y) == 0.5


def test_affine_reads_the_probe_back():
    rng = np.random.default_rng(0)
    w0, c0 = rng.normal(size=6).astype(np.float32), 0.3
    w, c = affine(lambda U: U @ w0 + c0, 6)
    assert np.allclose(w, w0, atol=1e-6) and abs(c - c0) < 1e-6


def test_log_odds_is_symmetric_and_corrected():
    assert log_odds(np.array([5.0]), 10, np.array([5.0]), 10)[0] == 0.0
    assert log_odds(np.array([0.0]), 10, np.array([0.0]), 1000)[0] > 0
    assert np.isfinite(log_odds(np.array([0.0]), 10, np.array([9.0]), 10)).all()


def test_script_share_counts_letters():
    assert script_share("", MYA) == 0.0
    assert script_share("123 !?", MYA) == 0.0
    assert script_share("hello", MYA) == 0.0
    # "Myanmar" in Burmese: consonants count, vowel signs and the virama do not
    assert script_share("မြန်မာ", MYA) == 1.0
    assert script_share("ab မ", MYA) == 1 / 3


def test_derangement_has_no_fixed_point():
    rng = np.random.default_rng(0)
    for n in (2, 3, 10, 257):
        p = derangement(n, rng)
        assert sorted(p) == list(range(n))
        assert not (p == np.arange(n)).any()


def test_rank_counts_ties_against_the_entry():
    v = np.array([0.1, 0.5, 0.5, 0.2])
    assert rank_of(v, 1) == 2          # tied with entry 2
    assert rank_of(v, 3) == 3
    assert rank_of(np.array([0.9, 0.1]), 0) == 1


def test_rows_avoid_the_language_and_the_prefix():
    langs = np.array(["my", "en", "fr"] * 40)
    rows = pick_rows(langs, 60, 20, "my", seed=0)
    assert len(rows) == len(set(rows)) == 20
    assert (rows >= 60).all()
    assert (langs[rows] != "my").all()
    assert (rows == pick_rows(langs, 60, 20, "my", seed=0)).all()
    en = pick_rows(langs, 60, 10, "my", seed=0, only="en")
    assert (langs[en] == "en").all() and (en >= 60).all()

# textfid.py's chrF implementation: the text-fidelity numbers ride on this
# metric behaving like the standard one (identity 1, disjoint 0, partial
# overlap between, whitespace-insensitive, recall-weighted via beta=2).
import numpy as np

from textfid import chrf


def test_identity_and_disjoint():
    assert chrf("the cat sat", "the cat sat") == 1.0
    assert chrf("aaaa", "zzzz") == 0.0
    assert chrf("", "") == 1.0
    assert chrf("abc", "") == 0.0


def test_partial_overlap_is_between():
    v = chrf("the cat sat on the mat", "the cat sat on the hat")
    assert 0.5 < v < 1.0


def test_whitespace_insensitive():
    assert chrf("thecat", "the cat") == 1.0


def test_more_overlap_scores_higher():
    ref = "she walked to the store"
    assert chrf(ref, "she walked to the shop") > chrf(ref, "he ran home")
    assert chrf(ref, "he ran home") > chrf(ref, "zzzzzz")


def test_recall_weighting():
    # beta=2 favours recall: a hypothesis containing the whole reference
    # (plus extra) beats one containing half the reference exactly
    ref = "abcdefgh"
    assert chrf(ref, "abcdefghijklmnop") > chrf(ref, "abcd")

# experiments/vertices/steer.py: forcing a head to a vertex, an extreme or
# a centroid, read through the decoder. The tests check the pure helpers:
# the draw of a head and another entry, the forced rows at their own and
# at matched lengths, the per-decode measures, and the paired contrasts.
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "vertices"))
from steer import (cond, contrast_pairs, forced_rows, measures,  # noqa: E402
                   pick, summarize)

D = 8


def test_cond_names_the_extreme_at_its_own_length():
    assert cond("extreme", 1.0) == "extreme"
    assert cond("extreme", 2.0) == "extreme@2"
    assert cond("vertex", 1.0) == "vertex@1"
    assert cond("random", 0.5) == "random@0.5"


def test_pick_draws_another_eligible_entry():
    rng = np.random.default_rng(0)
    n, h, k = 4000, 3, 5
    sel = rng.integers(0, k, (n, h))
    ok = np.ones((h, k), bool)
    ok[0] = False                  # head 0 has nothing eligible
    ok[1, 1:] = False              # head 1 can be forced only to entry 0
    ok[2, 4] = False
    heads, ents = pick(sel, ok, rng)
    assert (heads != 0).all()
    assert ok[heads, ents].all()
    assert (ents != sel[np.arange(n), heads]).all()
    # a row already on head 1's one eligible entry goes to head 2
    on0 = sel[:, 1] == 0
    assert (heads[on0] == 2).all()
    # both heads are drawn when both are open
    assert {1, 2} <= set(heads[~on0].tolist())


def test_pick_refuses_a_row_with_nothing_to_force():
    sel = np.zeros((1, 2), np.int64)
    ok = np.zeros((2, 3), bool)
    ok[:, 0] = True                # only the row's own entry is eligible
    with pytest.raises(ValueError):
        pick(sel, ok, np.random.default_rng(1))


def test_forced_rows_at_own_and_matched_lengths():
    rng = np.random.default_rng(2)
    n = 6
    Y = rng.normal(size=(n, D))
    steps = {name: rng.normal(size=(n, D)) * s
             for name, s in (("vertex", 10.0), ("extreme", 1.0),
                             ("centroid", 0.5))}
    rand = rng.normal(size=(n, D))
    out = forced_rows(Y, steps, rand, [0.5, 1.0, 4.0])
    for name, s in steps.items():
        assert np.allclose(out[name], Y + s)
    ref = np.linalg.norm(steps["extreme"], axis=1)
    for name, s in dict(steps, random=rand).items():
        for L in (0.5, 1.0, 4.0):
            d = out[cond(name, L)] - Y
            assert np.allclose(np.linalg.norm(d, axis=1), L * ref)
            cos = (d * s).sum(1) / (np.linalg.norm(d, axis=1)
                                    * np.linalg.norm(s, axis=1))
            assert np.allclose(cos, 1.0)
    assert "extreme@1" not in out
    assert len(out) == 3 + 4 * 3 - 1


def test_measures():
    rng = np.random.default_rng(3)
    n, k = 5, 4
    X = rng.normal(size=(n, D))
    j = np.array([0, 1, 2, 3, 0])
    P = np.full((n, k), 0.1)
    P[np.arange(n), j] = 0.7
    P[4] = [0.1, 0.6, 0.2, 0.1]    # the last row's argmax is not j
    P_given = np.full((n, k), 0.25)
    texts = ["a cat sat", "on the mat", "zzzz", "qqqq", "same"]
    base = ["a cat sat", "on the mat", "xxxx", "yyyy", "same"]
    given = 3.0 * X
    r = measures(texts, 2.0 * X, given, X, P, P_given, j, base,
                 np.full(n, 0.2))
    assert np.allclose(r["p"], [0.7, 0.7, 0.7, 0.7, 0.1])
    assert np.allclose(r["hit"], [1, 1, 1, 1, 0])
    assert np.allclose(r["p_given"], 0.25)
    assert np.allclose(r["effect"], r["p"] - 0.2)
    assert np.allclose(r["collateral"], [0, 0, 1, 1, 0])
    assert np.allclose(r["keep"], 1.0) and np.allclose(r["fidelity"], 1.0)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    assert np.allclose(r["keep0"], (Xn * np.roll(Xn, -1, 0)).sum(1))


def test_summarize_reads_paired_differences():
    rng = np.random.default_rng(4)
    n = 50
    names = ["vertex", "extreme", "centroid"] + [
        cond(d, L) for d in ("vertex", "extreme", "centroid", "random")
        for L in (1.0, 2.0)]
    per = {c: {m: rng.normal(size=n) for m in
               ("p", "hit", "p_given", "effect", "collateral", "keep",
                "keep0", "fidelity")} for c in dict.fromkeys(names)}
    res = summarize(per, [1.0, 2.0])
    d = res["contrasts"]["extreme - vertex"]["effect"]
    want = per["extreme"]["effect"] - per["vertex"]["effect"]
    assert np.isclose(d["mean"], want.mean())
    assert np.isclose(d["se"], want.std(ddof=1) / np.sqrt(n))
    # at length 1 the extreme's step is the extreme itself
    assert ("extreme", "vertex@1") in contrast_pairs([1.0])
    assert "extreme - vertex@1" in res["contrasts"]
    assert "extreme@2 - random@2" in res["contrasts"]

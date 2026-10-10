# experiments/ste-arm/reconseeds.py: whether two runs compute the same
# function or only reach the same error. Equal FVU is the trap these
# statistics exist to see past, so the tests pin what separates the two
# cases: the cross distance and the residual correlation.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "ste-arm"))
from reconseeds import (agreement, assigned, batch_sums,  # noqa: E402
                        best_match, decoder_cos, row_copies, supports)

N, D = 4096, 24


def data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D))
    w = rng.random(D) + 0.5
    var = float((w * (X - X.mean(0)) ** 2).sum())
    return rng, X, w, var


def test_one_function_has_no_cross_distance():
    rng, X, w, var = data()
    Y = X + 0.3 * rng.normal(size=X.shape)
    (r,) = agreement(batch_sums(X, Y[None], Y[None], w), var)
    assert np.isclose(r["cross"], 0.0)
    assert np.isclose(r["resid_corr"], 1.0)
    assert np.isclose(r["fvu_a"], r["fvu_b"])
    assert np.isclose(r["indep"], 2 * r["fvu_a"])


def test_equal_error_by_different_functions_reads_as_independent():
    """Same FVU, unrelated residuals: cross reaches indep, corr near 0."""
    rng, X, w, var = data(1)
    Ya = X + 0.3 * rng.normal(size=X.shape)
    Yb = X + 0.3 * rng.normal(size=X.shape)
    (r,) = agreement(batch_sums(X, Ya[None], Yb[None], w), var)
    assert abs(r["fvu_a"] - r["fvu_b"]) < 0.01
    assert abs(r["resid_corr"]) < 0.05
    assert np.isclose(r["cross"], r["indep"], rtol=0.05)


def test_cross_is_indep_less_twice_the_correlated_part():
    rng, X, w, var = data(2)
    shared = rng.normal(size=X.shape)
    Ya = X + 0.3 * shared + 0.2 * rng.normal(size=X.shape)
    Yb = X + 0.3 * shared + 0.2 * rng.normal(size=X.shape)
    for r in agreement(batch_sums(X, Ya[None], Yb[None], w), var):
        want = r["indep"] - 2 * r["resid_corr"] * np.sqrt(r["fvu_a"]
                                                          * r["fvu_b"])
        assert np.isclose(r["cross"], want)


def test_batches_sum_to_the_whole():
    rng, X, w, var = data(3)
    Ya = X[None] + 0.1 * rng.normal(size=(2, *X.shape))
    Yb = X[None] + 0.1 * rng.normal(size=(2, *X.shape))
    whole = batch_sums(X, Ya, Yb, w)
    parts = sum(batch_sums(X[i:i + 1000], Ya[:, i:i + 1000],
                           Yb[:, i:i + 1000], w) for i in range(0, N, 1000))
    assert np.allclose(whole, parts)
    rows = agreement(whole, var)
    assert [r["prefix"] for r in rows] == [1, 2]


def test_a_relabeled_dictionary_is_all_copies():
    rng = np.random.default_rng(4)
    m = 40
    Da = rng.normal(size=(m, D))
    perm = rng.permutation(m)
    Db = 3.0 * Da[perm]                       # scale must not matter
    C = decoder_cos(Da, Db)
    near = [0.9, 0.7]
    b = best_match(C, np.ones(m), near)
    assert np.isclose(b["median"], 1.0) and b["share_gt_0.9"] == 1.0
    a = assigned(C, near)
    assert np.isclose(a["mean"], 1.0) and a["share_gt_0.7"] == 1.0
    # firing-rate weighting: a perfect copy that never fires adds nothing
    C2 = C.copy()
    C2[0] = 0.0
    rate = np.ones(m)
    rate[0] = 0.0
    assert np.isclose(best_match(C2, rate, near)["rate_weighted_mean"], 1.0)


def test_row_copies_counts_only_copies_active_on_the_same_row():
    copy = np.zeros((3, 3), dtype=bool)
    copy[0, 2] = True
    act_a = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=bool)
    act_b = np.array([[0, 0, 1], [1, 1, 1], [1, 1, 0]], dtype=bool)
    # row 0: latent 0's copy (2) is active, latent 1 has none -> 0.5;
    # row 1: A inactive, skipped; row 2: latent 0's copy is off -> 0
    assert np.isclose(row_copies(act_a, act_b, copy), 0.25)


def test_supports_of_a_relabeled_sae():
    rng = np.random.default_rng(5)
    m, n = 30, 500
    Da = rng.normal(size=(m, D))
    perm = rng.permutation(m)
    Za = np.maximum(rng.normal(size=(n, m)) - 1.0, 0.0)
    Za[:, 7] = 0.0                            # never fires: not live
    s = supports(Za, Za[:, perm], Da, Da[perm], [0.9])
    assert s["live_a"] == s["live_b"] == m - 1
    assert np.isclose(s["best"]["mean"], 1.0)
    assert np.isclose(s["assigned"]["mean"], 1.0)
    assert np.isclose(s["rows"]["share_gt_0.9"], 1.0)
    assert np.isclose(s["l0_a"], s["l0_b"])

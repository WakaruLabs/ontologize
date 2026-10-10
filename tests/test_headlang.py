# headlang.py's probes on one-hot head codes: the normal equations it
# counts from the codes must be the ones the explicit design gives, the
# penalty it chooses must be chosen on the validation rows alone, and the
# completeness rows must be the head sets they are labeled as.
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"
                       / "ste-arm"))
from headlang import (choose_ridge, code_gram, code_probe,  # noqa: E402
                      code_scores, completeness_sets, dense_probe,
                      head_features, onehot_heads, ridge_probe, ridge_solve)


def coded(n=600, H=5, k=4, n_cls=3, seed=0):
    """Random codes where head 1's entry carries the label."""
    rng = np.random.default_rng(seed)
    A = rng.integers(0, k, (n, H))
    y = rng.integers(0, n_cls, n)
    A[:, 1] = np.where(rng.random(n) < 0.8, y, A[:, 1])
    return A, y, k, n_cls


def design(A, heads, k):
    F = onehot_heads(A, heads, k)
    return np.concatenate([F, np.ones((len(A), 1))], 1)


def test_code_gram_is_the_designs_normal_equations():
    A, y, k, n_cls = coded()
    G, B = code_gram(A, y, k, n_cls)
    F = design(A, range(A.shape[1]), k)
    np.testing.assert_array_equal(G, F.T @ F)
    np.testing.assert_array_equal(B, F.T @ np.eye(n_cls)[y])


def test_head_subset_reads_its_rows_and_columns():
    A, y, k, n_cls = coded()
    G, B = code_gram(A, y, k, n_cls)
    heads = [3, 1]
    cols = head_features(heads, k, len(G))
    F = design(A, heads, k)
    np.testing.assert_array_equal(G[np.ix_(cols, cols)], F.T @ F)
    np.testing.assert_array_equal(B[cols], F.T @ np.eye(n_cls)[y])


def test_code_scores_is_the_design_times_weights():
    A, _, k, n_cls = coded()
    heads = [4, 0, 2]
    W = np.random.default_rng(1).normal(size=(len(heads) * k + 1, n_cls))
    np.testing.assert_allclose(code_scores(A, heads, k, W),
                               design(A, heads, k) @ W)


def test_fixed_penalty_probe_matches_ridge_probe():
    A, y, k, n_cls = coded(n=800)
    Atr, Ate, ytr, yte = A[:600], A[600:], y[:600], y[600:]
    full = code_gram(Atr, ytr, k, n_cls)
    heads = [1, 2]
    got, lam = code_probe(heads, k, None, full, Atr[:0], ytr[:0], Ate, yte,
                          [0.5])
    want = ridge_probe(onehot_heads(Atr, heads, k), ytr,
                       onehot_heads(Ate, heads, k), yte, n_cls, 0.5)
    assert lam == 0.5
    assert got == pytest.approx(want)


def test_ridge_solve_solves_the_penalized_system():
    rng = np.random.default_rng(2)
    F = rng.normal(size=(50, 6))
    B = rng.normal(size=(6, 3))
    W = ridge_solve(F.T @ F, B, 0.3)
    np.testing.assert_allclose((F.T @ F + 0.3 * np.eye(6)) @ W, B, atol=1e-10)


def test_choose_ridge_takes_the_best_validation_score_first_on_ties():
    G, B = np.eye(3), np.ones((3, 2))
    score = {1.0: 0.2, 10.0: 0.7, 100.0: 0.7}
    # each penalty's solution is B / (1 + lam); recover lam from it
    pick = choose_ridge(G, B, lambda W: score[round(1 / W[0, 0] - 1)],
                        [1.0, 10.0, 100.0])
    assert pick == 10.0
    assert choose_ridge(G, B, lambda W: 1 / 0, [5.0]) == 5.0


def test_validated_probe_never_scores_penalties_on_test_rows():
    """With a penalty grid, the choice depends only on the validation rows:
    scrambling the test labels must not change it."""
    A, y, k, n_cls = coded(n=1000, seed=3)
    Atr, ytr, Ate, yte = A[:800], y[:800], A[800:], y[800:]
    cut = 600
    fit = code_gram(Atr[:cut], ytr[:cut], k, n_cls)
    full = code_gram(Atr, ytr, k, n_cls)
    lams = [1e-2, 1.0, 100.0]
    _, lam = code_probe([1], k, fit, full, Atr[cut:], ytr[cut:], Ate, yte,
                        lams)
    _, lam_scrambled = code_probe([1], k, fit, full, Atr[cut:], ytr[cut:],
                                  Ate, np.roll(yte, 1), lams)
    assert lam == lam_scrambled


def test_dense_probe_recovers_a_linear_label():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(2000, 8))
    y = (X @ rng.normal(size=(8, 3))).argmax(1)
    acc, lam = dense_probe(X[:1500], y[:1500], X[1500:], y[1500:], 3,
                           [1e-2, 1.0, 1e4], 0.25)
    assert acc > 0.8
    assert lam in (1e-2, 1.0)


def test_completeness_sets_around_c():
    layer = np.repeat(np.arange(3), 4)              # 3 layers x 4 heads
    order = np.array([0, 1, 5, 2, 3, 4, 6, 7, 8, 9, 10, 11])
    sets = completeness_sets(order, layer, 2, np.random.default_rng(0))
    C = set(sets["C: top 2 by NMI"].tolist())
    assert C == {0, 1}
    assert set(sets["ranks 3-4"].tolist()) == {5, 2}
    assert set(sets["layer 0 minus C"].tolist()) == {2, 3}
    rand = set(sets["2 random deeper heads"].tolist())
    assert len(rand) == 2 and all(layer[h] > 0 for h in rand)
    assert set(sets["layers 1-2"].tolist()) == set(range(4, 12))
    assert set(sets["all heads minus C"].tolist()) == set(range(12)) - C
    assert len(sets["all heads"]) == 12


def test_completeness_sets_one_layer_drops_empty_and_repeated_sets():
    layer = np.zeros(6, int)
    sets = completeness_sets(np.arange(6), layer, 2, np.random.default_rng(0))
    # no deeper heads, and the rest of layer 0 is every head but C
    assert list(sets) == ["C: top 2 by NMI", "ranks 3-4", "layer 0 minus C",
                          "all heads"]

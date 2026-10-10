# langprobe.py's selection and probing machinery: a planted
# language-indicator latent must be found by the t-statistic and yield
# near-perfect held-out F1; threshold and logistic helpers pinned on
# hand-checkable cases.
import numpy as np

import langprobe

N, F, L = 3000, 40, 4


def planted(seed=0):
    """Latent 7 indicates language 1 (plus noise); the rest are noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, L, N)
    A = rng.normal(size=(N, F)).astype(np.float32) * 0.1
    A[:, 7] += (y == 1).astype(np.float32)
    return A, y


def moments(A, y):
    onehot = np.eye(L, dtype=np.float64)[y]
    return onehot.T @ A, onehot.T @ (A * A), np.bincount(y, minlength=L)


def test_t_stats_finds_planted_latent():
    A, y = planted()
    S, S2, cnt = moments(A, y)
    T = langprobe.t_stats(S, S2, cnt, N)
    assert T.shape == (L, F)
    assert np.argmax(np.abs(T[1])) == 7
    assert T[1, 7] > 10          # decisive, not marginal
    # one-vs-rest: the same latent is a legitimate ANTI-indicator of the
    # other languages (it firing implies lang 1), but nothing else is
    assert T[0, 7] < -10
    assert np.abs(np.delete(T[0], 7)).max() < 6


def test_one_sparse_probe_end_to_end():
    A, y = planted(1)
    A_te, y_te = planted(2)
    ytr = (y == 1).astype(np.float32)
    yte = (y_te == 1).astype(np.float32)
    _, thr = langprobe.best_f1_threshold(A[:, 7], ytr)
    f1 = langprobe.f1_at(A_te[:, 7], yte, thr)
    assert f1 > 0.9


def test_best_f1_threshold_hand_case():
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    f1, thr = langprobe.best_f1_threshold(scores, y)
    assert f1 == 1.0
    assert 0.3 < thr <= 0.8      # separates the classes


def test_logistic_probe_beats_single_latent_on_conjunction():
    # language = latent 3 AND latent 5 -> no single latent suffices
    rng = np.random.default_rng(3)
    A = rng.normal(size=(4000, 8)).astype(np.float32)
    y = ((A[:, 3] > 0) & (A[:, 5] > 0)).astype(np.float32)
    score = langprobe.fit_logistic(A[:2000, [3, 5]], y[:2000])
    _, thr = langprobe.best_f1_threshold(score(A[:2000, [3, 5]]), y[:2000])
    f1_2 = langprobe.f1_at(score(A[2000:, [3, 5]]), y[2000:], thr)
    _, thr1 = langprobe.best_f1_threshold(A[:2000, 3], y[:2000])
    f1_1 = langprobe.f1_at(A[2000:, 3], y[2000:], thr1)
    assert f1_2 > f1_1 + 0.05
    assert f1_2 > 0.75


def test_only_rows_keeps_the_listed_languages_in_order():
    langs = np.array(["en", "fr", "de", "fr", "zh", "en", "it"])
    rows = np.arange(1, 7)
    kept = langprobe.only_rows(langs, rows, ["fr", "en"])
    assert kept.tolist() == [1, 3, 5]
    assert set(langs[kept]) == {"fr", "en"}


def test_only_rows_refuses_a_language_with_no_rows():
    langs = np.array(["en", "fr", "de"])
    try:
        langprobe.only_rows(langs, np.arange(3), ["en", "pt"])
    except SystemExit as e:
        assert "pt" in str(e)
    else:
        raise AssertionError("a missing language was not refused")

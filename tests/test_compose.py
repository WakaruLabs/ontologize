"""compose.py pure stats: exact additivity must score cos 1 / rel 0,
planted non-additivity must be recovered exactly, and the whitened metric
must weight the error by sqrt(w)."""
import numpy as np

from compose import additivity_stats, sample_singles


def test_exact_additivity():
    rng = np.random.default_rng(0)
    D1, D2 = rng.normal(size=(2, 16, 8))
    cos, rel = additivity_stats(D1, D2, D1 + D2)
    assert abs(cos - 1.0) < 1e-6
    assert rel < 1e-6


def test_planted_orthogonal_error():
    # D12 = S + e with e orthogonal to S and ||e|| = 0.5 ||S|| per row:
    # rel = 0.5 exactly, cos = 1/sqrt(1.25).
    rng = np.random.default_rng(1)
    D1, D2 = rng.normal(size=(2, 16, 8))
    S = D1 + D2
    e = rng.normal(size=S.shape)
    e -= S * (e * S).sum(-1, keepdims=True) / (S * S).sum(-1, keepdims=True)
    e *= 0.5 * np.linalg.norm(S, axis=-1, keepdims=True) \
        / np.linalg.norm(e, axis=-1, keepdims=True)
    cos, rel = additivity_stats(D1, D2, S + e)
    assert abs(rel - 0.5) < 1e-6
    assert abs(cos - 1 / np.sqrt(1.25)) < 1e-6


def test_whitened_metric_reweights():
    # Error confined to one dimension: up-weighting that dimension must
    # raise the relative error; killing it must zero the error.
    D1 = np.zeros((4, 3))
    D2 = np.eye(3)[None, 0] * np.ones((4, 1))
    E = np.zeros((4, 3))
    E[:, 2] = 1.0
    w_up = np.array([1.0, 1.0, 4.0])
    w_off = np.array([1.5, 1.5, 0.0])
    _, rel_raw = additivity_stats(D1, D2, D1 + D2 + E)
    _, rel_up = additivity_stats(D1, D2, D1 + D2 + E, w_up)
    _, rel_off = additivity_stats(D1, D2, D1 + D2 + E, w_off)
    assert rel_up > rel_raw
    assert rel_off < 1e-9


def test_sample_singles_distinct_heads_per_layer():
    singles = sample_singles(l=3, h=4, k=8, per_layer=4, seed=0)
    assert len(singles) == 12
    for li in range(3):
        heads = [hi for l_, hi, _ in singles if l_ == li]
        assert len(set(heads)) == len(heads) == 4
    assert all(0 <= ki < 8 for _, _, ki in singles)

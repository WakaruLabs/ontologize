# experiments/recipe-ladder/noiseshare.py: how much of each layer's input
# is training noise. The statistics are pinned on hand-checkable arrays, and
# the replayed forward on a small stack: layer 0's input never carries
# noise, feature noise reaches every later layer's input, and the clean
# replay is the model's own deterministic forward.
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "recipe-ladder"))
from noiseshare import flip_rate, shape_dist, share, trace  # noqa: E402


def test_share_is_noise_over_signal_per_layer():
    Yc = np.ones((2, 4, 3))
    Yn = Yc.copy()
    Yn[1] += 0.5                      # layer 1: each row off by 0.5 in 3 coords
    np.testing.assert_allclose(share(Yn, Yc), [0.0, 0.25])


def test_shape_distance_reads_angle_not_scale():
    U = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    assert shape_dist(3.0 * U, U)[0] < 1e-12
    assert np.isclose(shape_dist(U[:, ::-1], U)[0], 1.0)


def test_flip_rate_counts_heads():
    k = 3
    c = np.zeros((1, 2, 2 * k))
    c[0, :, [0, k]] = 1.0             # both rows: head 0 -> 0, head 1 -> 0
    n = c.copy()
    n[0, 1, :k] = [0.0, 1.0, 0.0]     # row 1, head 0 moves to entry 1
    assert flip_rate(n, c, k)[0] == 0.25


def test_feature_noise_reaches_only_later_layers(build, X):
    model, params = build(noise_F="featvar", noise_K="batchnorm")
    rng = jax.random.PRNGKey(3)
    go = lambda **kw: [np.asarray(a) for a in model.apply(
        params, X, rng, method=trace, temperature=0.5,
        **{"sd_K": 0.0, "sd_F": 0.0, "p_drop": 0.0, **kw})]
    Yc, Uc, Kc = go()
    Yc2, _, Kc2 = go()
    assert (Yc == Yc2).all() and (Kc == Kc2).all()
    Y, U, K = go(sd_F=0.5)
    s = share(Y, Yc)
    assert s[0] == 0.0 and (s[1:] > 0.0).all()
    assert flip_rate(K, Kc, model.k)[0] == 0.0
    Y, _, K = go(sd_K=1.0)
    assert share(Y, Yc)[0] == 0.0 and flip_rate(K, Kc, model.k)[0] > 0.0

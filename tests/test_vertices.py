# experiments/vertices/place.py: where a head's atoms sit against its cells.
# The tests plant cells whose centroid and edge are known, and check that
# the least-squares atom, the hard-decode price, the share of rows past an
# atom, its reach and the trimmed extreme read them; and the two
# nearest-row and best-cosine helpers against brute force.
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "vertices"))
from place import (best_cos, centroids, nn_dist, place_layer,  # noqa: E402
                   price, ray, run_temperature, shaped_random, wquantile)

D = 6


def planted_layer(rng, expand=1.0, common=0.0, n=8000, h=2, k=4):
    """One layer of h hard heads. Rows are the heads' true atoms times a
    gain, plus noise, plus `common` times the gain along one direction.
    The model's atoms are the true ones pushed out from their head's mean
    by `expand`."""
    true = rng.normal(size=(h, k, D)) * 3.0
    mean = true.mean(1, keepdims=True)
    model = mean + expand * (true - mean)
    sel = rng.integers(0, k, (n, 1, h))
    g = rng.uniform(0.5, 1.5, (n, 1))
    pick = lambda A: A[np.arange(h), sel[:, 0]]           # noqa: E731
    Y = g * pick(model).sum(1)
    m = np.zeros(D)
    m[0] = common
    Xw = g * pick(true).sum(1) + g * m + 0.3 * rng.normal(size=(n, D))
    view = dict(A=model[None], sel=sel, P=None, G=g, Y=Y)
    var = float(((Xw - Xw.mean(0)) ** 2).sum())
    return view, Xw, var


def test_place_layer_reads_centroid_atoms_as_centroids():
    view, Xw, var = planted_layer(np.random.default_rng(8))
    r, E, M = place_layer(view, Xw, 0, var, 10)
    assert r["off"]["median"] < 0.1
    assert abs(r["atom_over_mu"]["median"] - 1.0) < 0.05
    assert np.isfinite(E).all() and np.isfinite(M).all()


def test_place_layer_returns_the_planted_centroids():
    # the model's atoms are pushed out from their head's mean, and the
    # centroids it returns are the planted atoms they were pushed from
    view, Xw, var = planted_layer(np.random.default_rng(11), expand=1.5)
    _, _, M = place_layer(view, Xw, 0, var, 10)
    A = view["A"][0]
    mean = A.mean(1, keepdims=True)
    true = mean + (A - mean) / 1.5
    miss = np.linalg.norm(M - true, axis=-1)
    push = np.linalg.norm(A - true, axis=-1)
    assert miss.max() < 0.1 * push.min()


def test_place_layer_sets_a_common_offset_apart():
    view, Xw, var = planted_layer(np.random.default_rng(9), common=2.0)
    r, _, _ = place_layer(view, Xw, 0, var, 10)
    # every head's own refit would take the whole shared component, so the
    # uncentred distance is large while the placement is unchanged
    assert r["off_raw"]["median"] > 1.0
    assert r["off"]["median"] < 0.1
    assert r["common"]["median"] > 1.0
    assert abs(r["atom_over_mu"]["median"] - 1.0) < 0.05


def test_place_layer_reads_atoms_pushed_outward():
    view, Xw, var = planted_layer(np.random.default_rng(10), expand=1.5)
    r, _, _ = place_layer(view, Xw, 0, var, 10)
    assert abs(r["atom_over_mu"]["median"] - 1.5) < 0.1
    assert r["off"]["median"] > 0.5
    assert r["beyond"]["median"] < 0.5


def test_centroid_is_the_weighted_least_squares_atom():
    rng = np.random.default_rng(0)
    n, k = 500, 4
    sel = rng.integers(0, k - 1, n)          # entry k-1 is never selected
    g = rng.uniform(0.5, 2.0, n)
    t = rng.normal(size=(n, D))
    mass, mu, spread = centroids(t, g, sel, k)
    for j in range(k - 1):
        m = sel == j
        assert np.isclose(mass[j], (g[m] ** 2).sum())
        want = (g[m, None] * t[m]).sum(0) / (g[m] ** 2).sum()
        assert np.allclose(mu[j], want)
        z = t[m] / g[m, None]
        rms = np.sqrt(((g[m] ** 2) * ((z - want) ** 2).sum(1)).sum()
                      / (g[m] ** 2).sum())
        assert np.isclose(spread[j], rms)
    assert mass[k - 1] == 0 and np.isnan(mu[k - 1]).all()


def test_price_is_the_hard_decode_excess_over_the_centroid():
    rng = np.random.default_rng(1)
    n = 300
    g = rng.uniform(0.5, 2.0, n)
    t = rng.normal(size=(n, D)) + 3.0
    mass, mu, _ = centroids(t, g, np.zeros(n, np.int64), 1)
    a = mu[0] + rng.normal(size=D)
    sse = lambda atom: ((t - g[:, None] * atom) ** 2).sum()  # noqa: E731
    assert np.isclose(sse(a) - sse(mu[0]), price(mass[0], mu[0], a))
    assert price(mass[0], mu[0], mu[0]) == 0.0


def planted_cell(rng, n=20000):
    """A cell whose rows project uniformly on [0, 2] along the ray from the
    origin, plus noise across it: centroid at 1, 95th percentile at 1.9."""
    u = np.zeros(D)
    u[0] = 1.0
    s = rng.uniform(0.0, 2.0, n)
    z = s[:, None] * u + 0.3 * rng.normal(size=(n, D)) * (1 - u)
    return z, np.ones(n), np.zeros(D), u


def test_ray_reads_centroid_edge_and_beyond():
    rng = np.random.default_rng(2)
    z, w, c, u = planted_cell(rng)
    beyond, reach, e = ray(z, w, c, c + 1.0 * u)
    assert abs(beyond - 0.5) < 0.02
    assert abs(reach - 1.0 / 1.9) < 0.02
    beyond, reach, e = ray(z, w, c, c + 1.9 * u)
    assert abs(beyond - 0.05) < 0.01 and abs(reach - 1.0) < 0.02
    beyond, reach, _ = ray(z, w, c, c + 4.0 * u)
    assert beyond == 0.0 and reach > 2.0
    # the trimmed extreme sits between the 80th and 95th percentiles
    assert 1.6 < e @ u < 1.9


def test_ray_weights_rows():
    rng = np.random.default_rng(3)
    z, w, c, u = planted_cell(rng)
    w = np.where(z[:, 0] > 1.0, 3.0, 1.0)    # the far half weighs 3x
    beyond, _, _ = ray(z, w, c, c + 1.0 * u)
    assert abs(beyond - 0.75) < 0.02


def test_ray_at_the_centre_is_undefined():
    rng = np.random.default_rng(4)
    z, w, c, _ = planted_cell(rng, 100)
    beyond, reach, e = ray(z, w, c, c)
    assert np.isnan(beyond) and np.isnan(reach) and np.isnan(e).all()


def test_wquantile():
    x = np.arange(10.0)
    assert wquantile(x, np.ones(10), 0.5) == 4.0
    w = np.zeros(10)
    w[7] = 1.0
    assert wquantile(x, w, 0.1) == 7.0


def test_nn_dist_matches_brute_force():
    rng = np.random.default_rng(5)
    Q, R = rng.normal(size=(37, D)), rng.normal(size=(211, D))
    want = np.sqrt(((Q[:, None] - R[None]) ** 2).sum(-1)).min(1)
    assert np.allclose(nn_dist(Q, R, chunk=8), want)
    assert np.allclose(nn_dist(R[:5], R), 0.0, atol=1e-6)


def test_best_cos():
    rng = np.random.default_rng(6)
    R = rng.normal(size=(400, 64))
    assert np.allclose(best_cos(2.5 * R[:3], R), 1.0)
    # random directions in 64 dimensions reach far less than 1
    assert best_cos(rng.normal(size=(50, 64)), R).max() < 0.7


def test_shaped_random_follows_the_rows_covariance():
    rng = np.random.default_rng(7)
    scale = np.array([10.0] + [1.0] * (D - 1))
    ref = rng.normal(size=(5000, D)) * scale
    v = shaped_random(ref, 4000, rng)
    sd = v.std(0)
    assert sd[0] / sd[1:].mean() > 7.0


def test_run_temperature(tmp_path):
    log = tmp_path / "log.jsonl"
    hyper = {"temperature": 1.0, "temperature_end": 0.03}
    log.write_text(json.dumps({"type": "env_config", "hyper": hyper}) + "\n"
                   + "not json\n")
    assert run_temperature(str(tmp_path)) == 0.03
    hyper = {"temperature": 0.00015, "temperature_end": None}
    log.write_text(json.dumps({"type": "env_config", "hyper": hyper}) + "\n")
    assert run_temperature(str(tmp_path)) == 0.00015

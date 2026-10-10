# steerembed.py's supervised baselines: the difference-of-means and probe
# vectors it scores alongside the model's own steering directions. They
# are only a fair baseline if they recover a direction that is there and
# never see the rows they are about to steer. Then its data-shaped
# controls, the on/off split of a direction by the data's principal
# subspaces, and the geometry it reports for every direction.
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"
                       / "ste-arm"))
from steerembed import (controls, data_metric, partners, pct,  # noqa: E402
                        principal_ranks, split_dirs, step_geometry,
                        supervised_dirs)

D, N = 32, 4096


def planted(seed=0):
    """Unit-ish rows with a rare class (~5%) defined by one direction."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=D)
    u /= np.linalg.norm(u)
    X = rng.normal(size=(N, D)) / np.sqrt(D)
    y = X @ u > np.quantile(X @ u, 0.95)
    return jnp.asarray(X, jnp.float32), y, u


def test_both_vectors_recover_the_planted_direction():
    X, y, u = planted()
    dm, probe = supervised_dirs(X, y, np.array([], int))
    dm = np.asarray(dm) / np.linalg.norm(np.asarray(dm))
    assert dm @ u > 0.95
    assert np.asarray(probe) @ u > 0.95
    assert np.isclose(np.linalg.norm(np.asarray(probe)), 1.0, atol=1e-5)


def test_held_out_rows_do_not_enter_the_fit():
    """Corrupting only the excluded rows must leave both vectors unchanged."""
    X, y, _ = planted(1)
    excl = np.arange(64)
    dm0, p0 = supervised_dirs(X, y, excl)
    X2 = X.at[excl].set(100.0 * jnp.ones((len(excl), D)))
    dm1, p1 = supervised_dirs(X2, y, excl)
    assert np.allclose(dm0, dm1, atol=1e-6)
    assert np.allclose(p0, p1, atol=1e-6)


def anisotropic(d=64, n=20000, seed=0):
    """Gaussian rows whose standard deviations run from 4 to 1/4 along a
    random rotation, so the anisotropy is in the correlations."""
    rng = np.random.default_rng(seed)
    R = np.linalg.qr(rng.normal(size=(d, d)))[0]
    return (rng.normal(size=(n, d)) * np.geomspace(4.0, 0.25, d)) @ R.T


def test_metric_roots_invert_the_covariance():
    X = anisotropic()
    m = data_metric(X, 1 / X.var(0), np.random.default_rng(1))
    S = np.cov(X, rowvar=False)
    assert np.all(np.diff(m.lam) <= 0)
    assert np.allclose(m.Sh @ m.Sh, S, atol=1e-8)
    assert np.allclose(m.Sih @ S @ m.Sih, np.eye(X.shape[1]), atol=1e-6)


def test_floor_bounds_the_inverse_root_on_a_dead_direction():
    X = anisotropic(d=16)
    X = np.concatenate([X, np.full((len(X), 1), 0.3)], axis=1)
    m = data_metric(X, np.ones(X.shape[1]), np.random.default_rng(1))
    top = np.linalg.eigvalsh(m.Sih).max()
    assert np.isfinite(m.Sih).all()
    assert np.isclose(top, 1 / np.sqrt(1e-6 * m.lam[0]), rtol=1e-6)


def test_principal_ranks_are_the_fewest_components_holding_the_fraction():
    lam = np.array([4.0, 3.0, 2.0, 1.0])
    assert principal_ranks(lam, [0.4, 0.5, 0.7, 0.71, 1.0]) == {
        0.4: 1, 0.5: 2, 0.7: 2, 0.71: 3, 1.0: 4}


def test_partners_are_uniform_over_the_other_rows():
    rng = np.random.default_rng(0)
    idx = np.arange(50)
    for _ in range(100):
        p = partners(idx, 50, rng)
        assert (p != idx).all() and (p >= 0).all() and (p < 50).all()
    seen = {int(partners(np.array([3]), 5, rng)[0]) for _ in range(200)}
    assert seen == {0, 1, 2, 4}


def test_split_parts_are_orthogonal_and_rebuild_the_direction():
    X = anisotropic()
    m = data_metric(X, 1 / X.var(0), np.random.default_rng(1))
    ranks = principal_ranks(m.lam, [0.5, 0.8])
    delta = np.random.default_rng(2).normal(size=(8, X.shape[1]))
    parts = split_dirs(jnp.asarray(delta, jnp.float32),
                       jnp.asarray(m.Q, jnp.float32), ranks)
    for p, r in ranks.items():
        on = np.asarray(parts[f"on{pct(p)}"], np.float64)
        off = np.asarray(parts[f"off{pct(p)}"], np.float64)
        Qr = m.Q[:, :r]
        assert np.allclose(np.linalg.norm(on, axis=1), 1, atol=1e-5)
        assert np.allclose(np.linalg.norm(off, axis=1), 1, atol=1e-5)
        assert np.allclose(on - (on @ Qr) @ Qr.T, 0, atol=1e-5)
        assert np.allclose(off @ Qr, 0, atol=1e-5)
        rebuilt = ((delta * on).sum(1, keepdims=True) * on
                   + (delta * off).sum(1, keepdims=True) * off)
        assert np.allclose(rebuilt, delta, atol=1e-4)


def test_controls_have_the_shapes_the_geometry_says():
    """A step shaped like the data reads about 1 in both lengths; an
    isotropic one carries the mean variance on average and reads
    sqrt(mean(1/lam) mean(lam)) in the Mahalanobis metric."""
    X = anisotropic()
    rng = np.random.default_rng(1)
    m = data_metric(X, 1 / X.var(0), rng)
    ranks = principal_ranks(m.lam, [0.5])
    b = 4096
    c = controls(jax.random.normal(jax.random.PRNGKey(0), (b, X.shape[1])),
                 jnp.asarray(X[:b], jnp.float32),
                 jnp.asarray(X[partners(np.arange(b), len(X), rng)],
                             jnp.float32),
                 jnp.asarray(m.Sh, jnp.float32),
                 jnp.asarray(m.Sih, jnp.float32))
    g = {k: step_geometry(np.asarray(v), m, ranks) for k, v in c.items()}
    med = {k: {s: np.median(v) for s, v in gk.items()} for k, gk in g.items()}
    for k in ("cov", "diff"):
        assert abs(med[k]["mahalanobis"] - 1) < 0.05
        assert abs(med[k]["whitened"] - 1) < 0.05
    iso = np.sqrt(np.mean(1 / m.lam) * np.mean(m.lam))
    assert abs(med["random"]["mahalanobis"] / iso - 1) < 0.1
    assert abs(g["random"]["var_share"].mean() - 1) < 0.05
    assert (med["cov"]["var_share"] > med["random"]["var_share"]
            > med["icov"]["var_share"])
    assert med["cov"]["top50"] > med["random"]["top50"] > med["icov"]["top50"]

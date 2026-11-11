# experiments/ste-arm/condrefit.py: a layer's atoms refit within the cells
# of an earlier head. The tests plant a layer whose true atoms do depend
# on an earlier cell, where the conditional refit must beat a random
# partition, and one whose atoms do not, where it must not.
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "ste-arm"))
from condrefit import (design, eta2, like, predict, refit, score,  # noqa: E402
                       solve_toward)

H, K, D, C = 3, 4, 12, 5


def planted(n: int, rng: np.random.Generator, depends: float):
    """A layer of H heads whose atoms are shared (`depends` 0) or shifted
    per cell of an earlier head by that much. Returns selections, gains,
    the earlier head's cells, the target and the shared atoms."""
    base = rng.normal(size=(H * K, D)).astype(np.float32)
    shift = depends * rng.normal(size=(C, H * K, D)).astype(np.float32)
    I = rng.integers(0, K, (H, n))
    g = rng.uniform(0.5, 1.5, n).astype(np.float32)
    cell = rng.integers(0, C, n)
    cols = (I + np.arange(H)[:, None] * K).T
    T = (base[cols] + shift[cell[:, None], cols]).sum(1) * g[:, None]
    T = T + 0.05 * rng.normal(size=T.shape).astype(np.float32)
    return I, g, cell, T.astype(np.float32), base


def fvus(depends: float, seed: int):
    rng = np.random.default_rng(seed)
    I, g, cell, T, base = planted(6000, rng, depends)
    Ie, ge, celle, Te, _ = (a[..., 4000:] if a.ndim == 2 and a.shape[0] == H
                            else a[4000:] for a in (I, g, cell, T, base))
    If, gf, cellf, Tf = I[:, :4000], g[:4000], cell[:4000], T[:4000]
    cells = {"global": np.zeros(4000, np.int64), "cond": cellf,
             "random": rng.permutation(cellf)}
    A = refit(If, gf, K, Tf, cells, np.zeros_like(base), 1e-6)
    evalc = {"global": np.zeros(2000, np.int64), "cond": celle,
             "random": like(cellf, 2000, rng)}
    var = float(((Te - Te.mean(0)) ** 2).sum())
    return {n: score(A[n], Ie, ge, evalc[n], Te, K) / var for n in A}


def test_design_times_atoms_is_the_layer_output():
    rng = np.random.default_rng(0)
    I, g, cell, T, base = planted(50, rng, 0.0)
    Phi = design(I, g, K)
    assert Phi.shape == (50, H * K) and (Phi != 0).sum(1).tolist() == [H] * 50
    cols = (I + np.arange(H)[:, None] * K).T
    assert np.allclose(Phi @ base, base[cols].sum(1) * g[:, None], atol=1e-5)
    P = predict(jnp.asarray(base)[None], jnp.asarray(cols), jnp.asarray(g),
                jnp.zeros(50, jnp.int32))
    assert np.allclose(np.asarray(P), Phi @ base, atol=1e-5)


def test_conditional_refit_wins_only_when_atoms_depend_on_the_cell():
    dep = fvus(0.5, 1)
    assert dep["cond"] < 0.01                       # recovers the planted code
    assert dep["cond"] < 0.2 * dep["global"]
    assert dep["random"] > 0.9 * dep["global"]      # a blind partition does not
    ind = fvus(0.0, 2)
    assert ind["global"] < 0.01
    assert abs(ind["cond"] - ind["random"]) < 0.002
    assert ind["cond"] >= ind["global"] - 1e-4


def test_an_entry_a_cell_never_selects_keeps_the_prior_atom():
    rng = np.random.default_rng(3)
    a, d = 4, 3
    A_0 = rng.normal(size=(a, d)).astype(np.float32)
    G = np.zeros((2, a, a), np.float32)
    B = np.zeros((2, a, d), np.float32)
    G[0] = np.diag([5.0, 5.0, 0.0, 0.0])            # entries 2, 3 unseen
    B[0, :2] = 5.0 * np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    A = np.asarray(solve_toward(jnp.asarray(G), jnp.asarray(B),
                                jnp.asarray(A_0), 1e-6))
    assert np.allclose(A[0, :2], [[1, 2, 3], [4, 5, 6]], atol=1e-3)
    assert np.allclose(A[0, 2:], A_0[2:], atol=1e-5)
    assert np.allclose(A[1], A_0, atol=1e-5)        # a cell with no rows


def test_eta2_reads_between_cell_variance():
    rng = np.random.default_rng(4)
    cell = rng.integers(0, C, 3000)
    means = rng.normal(size=(C, D))
    T = means[cell] + 0.1 * rng.normal(size=(3000, D))
    assert eta2(T, cell) > 0.95
    assert eta2(T, rng.permutation(cell)) < 0.01
    drawn = like(cell, 5000, rng)
    assert np.allclose(np.bincount(drawn) / 5000, np.bincount(cell) / 3000,
                       atol=0.03)

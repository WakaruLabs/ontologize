# experiments/seed-probes/complement.py: how much of what one code
# explains about the input another explains already. A relabelled copy
# must read full overlap, a code of independent information none, and the
# block-assembled centered ridge must be the ridge on centered features.
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "seed-probes"))
from complement import moments, overlap, readout, system  # noqa: E402
from translate import onehot  # noqa: E402

K = 6


def planted(n: int, seed: int):
    """Two independent codes of 3 heads and a target that is a sum of an
    atom per entry of each, plus noise."""
    rng = np.random.default_rng(seed)
    A = rng.integers(0, K, (n, 3))
    X = rng.integers(0, K, (n, 3))
    WA, WX = rng.standard_normal((3, K, 8)), rng.standard_normal((3, K, 8))
    Y = (WA[np.arange(3), A].sum(1) + WX[np.arange(3), X].sum(1)
         + 0.3 * rng.standard_normal((n, 8)))
    return A, X, Y.astype(np.float32)


def fit_all(codes, Y, conds, n_fit, n_val, lams=(1e-6,)):
    others = [c for c in codes if c != "A"]
    pairs = ([("A", "A")] + [(x, x) for x in others]
             + [("A", x) for x in others])
    Yt = jnp.asarray(Y - Y[:n_fit].mean(0))
    S, FY, G, sy, nf = moments(codes, pairs, Yt, K, 0, n_fit, 1000)
    return {"+".join(c): readout(codes, c, S, FY, G, sy, nf, Yt, K, n_val,
                                 len(Y), lams, 1000) for c in conds}


def evs(res):
    return {c: 1 - r["test_fvu"] for c, r in res.items()}


def test_a_relabelled_copy_overlaps_fully():
    A, _, Y = planted(6000, 0)
    rng = np.random.default_rng(1)
    relabel = np.stack([rng.permutation(K) for _ in range(3)])
    copy = relabel[np.arange(3), A]
    codes = {"A": jnp.asarray(A), "copy": jnp.asarray(copy)}
    ev = evs(fit_all(codes, Y, [("A",), ("copy",), ("A", "copy")],
                     4000, 1000))
    assert abs(ev["A"] - ev["copy"]) < 1e-3
    assert abs(overlap(ev["A"], ev["copy"], ev["A+copy"]) - 1) < 0.01


def test_independent_information_does_not_overlap():
    A, X, Y = planted(6000, 2)
    codes = {"A": jnp.asarray(A), "X": jnp.asarray(X)}
    ev = evs(fit_all(codes, Y, [("A",), ("X",), ("A", "X")], 4000, 1000))
    assert ev["A+X"] > ev["A"] + 0.3
    assert abs(overlap(ev["A"], ev["X"], ev["A+X"])) < 0.05


def test_block_system_is_the_ridge_on_centered_features():
    A, X, Y = planted(500, 3)
    codes = {"A": jnp.asarray(A), "X": jnp.asarray(X)}
    Yt = jnp.asarray(Y)
    # 400 rows in chunks of 64: the last chunk is padded
    S, FY, G, sy, n = moments(codes, [("A", "A"), ("X", "X"), ("A", "X")],
                              Yt, K, 0, 400, 64)
    Gc, Bc, mu = system(("A", "X"), S, FY, G, sy, n)
    assert Gc.dtype == np.float64
    F = np.concatenate([np.asarray(onehot(codes[c][:400], K, False), np.float64)
                        for c in "AX"], 1)
    Fc, y = F - F.mean(0), Y[:400].astype(np.float64)
    assert np.allclose(np.asarray(Gc), Fc.T @ Fc, atol=1e-8)
    assert np.allclose(np.asarray(Bc), Fc.T @ (y - y.mean(0)), atol=1e-6)
    assert np.allclose(np.asarray(mu), F.mean(0))


def test_the_penalty_is_chosen_on_validation_rows():
    A, X, Y = planted(3000, 4)
    codes = {"A": jnp.asarray(A), "X": jnp.asarray(X)}
    res = fit_all(codes, Y, [("A",)], 2000, 500, lams=(1e-6, 10.0))
    assert res["A"]["lambda"] == 1e-6

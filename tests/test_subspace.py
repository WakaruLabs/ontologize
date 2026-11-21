# experiments/seed-stability/subspace.py: seed agreement as subspaces. The
# tests pin the normalized projection similarity phi (1 for nested
# subspaces, 0 for orthogonal ones, max(i, j)/d for random ones), the
# energy share EV, and the transfer curves on planted sets: two seeds whose
# sets share a subspace read cross = own, and sets in unrelated subspaces
# read cross at chance.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "seed-stability"))
from subspace import (basis, best_cos, eig_basis, ev, input_table, phi,  # noqa: E402
                      phi_table, transfer)


def _orth(d, q, rng):
    return np.linalg.qr(rng.normal(size=(d, q)))[0]


def test_phi_nested_orthogonal_and_random():
    rng = np.random.default_rng(0)
    d = 400
    A = _orth(d, 50, rng)
    assert np.isclose(phi(A, A, 10, 10), 1.0)
    assert np.isclose(phi(A, A, 5, 20), 1.0)          # nested prefixes
    E = np.eye(d)
    assert phi(E[:, :10], E[:, 10:30], 10, 20) == 0.0
    vals = [phi(_orth(d, 20, rng), _orth(d, 20, rng), 20, 20) for _ in range(50)]
    assert abs(np.mean(vals) - 20 / d) < 0.01


def test_ev_shares_energy():
    E = np.eye(5)
    X = np.array([[3.0, 4.0, 0, 0, 0]])
    assert np.isclose(ev(X, E[:, :1]), 9 / 25)
    assert np.isclose(ev(X, E[:, :2]), 1.0)


def test_bases_are_ordered():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 6)) * np.array([5, 4, 3, 2, 1, 0.5])
    Q = basis(X)
    assert np.allclose(np.abs(Q[:, 0]), np.eye(6)[0], atol=0.1)
    V = eig_basis(X.T @ X)
    assert np.allclose(np.abs(V[:, 0] @ Q[:, 0]), 1.0, atol=1e-6)


def test_transfer_reads_shared_and_unrelated_subspaces():
    rng = np.random.default_rng(2)
    d = 64
    shared = _orth(d, 4, rng)
    unit = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
    sets = {("a", "U"): unit(rng.normal(size=(60, 4)) @ shared.T),
            ("b", "U"): unit(rng.normal(size=(80, 4)) @ shared.T),
            ("a", "S"): unit(rng.normal(size=(60, 4)) @ _orth(d, 4, rng).T),
            ("b", "S"): unit(rng.normal(size=(60, 4)) @ _orth(d, 4, rng).T)}
    out = transfer(sets, np.eye(d), (4,), draws=3, seed=0)
    U, S = out["U"]["curves"]["a"][4], out["S"]["curves"]["a"][4]
    assert out["U"]["n"] == 60
    assert np.isclose(U["own"], 1.0) and np.isclose(U["cross"], 1.0)
    assert np.isclose(S["own"], 1.0) and S["cross"] < 0.4
    assert np.isclose(U["chance"], 4 / d)


def test_phi_table_and_best_cos():
    rng = np.random.default_rng(3)
    A = _orth(32, 32, rng)
    t = phi_table(A, A, np.eye(32), (4,))
    assert np.isclose(t[4]["ab"], 1.0) and np.isclose(t[4]["ab_x"], 8.0)
    D = np.eye(3)
    assert np.allclose(best_cos(D, D[[0, 1]]), [1.0, 1.0, 0.0])


def test_input_table_reads_contributions_that_follow_their_inputs():
    # Each seed's contribution spans the top of its own input; the inputs
    # share their top 4 directions, so the contributions agree across seeds
    # exactly as much as each agrees with the other seed's input.
    rng = np.random.default_rng(4)
    d = 32
    top = _orth(d, 4, rng)
    rest = lambda: np.linalg.qr(np.hstack([top, rng.normal(size=(d, d - 4))]))[0]
    A_in, B_in = rest(), rest()
    t = input_table(A_in, B_in, A_in, B_in, (4, 8))
    assert np.isclose(t[4]["a_in"], 1.0) and np.isclose(t[4]["in_ab"], 1.0)
    assert np.isclose(t[4]["a_inb"], 1.0) and np.isclose(t[4]["a_in_x"], 8.0)
    assert t[8]["in_ab"] < 0.9                  # beyond the shared top

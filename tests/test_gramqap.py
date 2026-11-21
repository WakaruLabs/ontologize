# experiments/ste-arm/gramqap.py: pairing two heads' entries from their
# Grams alone. A head whose atoms are rotated and relabelled must be found
# by the QAP and missed by the match on atoms; independent heads must read
# near chance; the best order never reads above the atom match.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "ste-arm"))
from atomgram import entry_match, gram, matched_rms, pair_ratio  # noqa: E402
from gramqap import (deviation, entries_recovered, qap_distances,  # noqa: E402
                     qap_order, rms)

H, K, D = 5, 12, 10


def rotated_copies(seed: int, noise: float = 0.01):
    """Heads of random atoms, and each head's atoms relabelled, rotated by
    a random orthogonal map and slightly perturbed: (A, B, B's head order,
    each B head's entry order)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((H, K, D))
    heads = rng.permutation(H)
    entries = np.stack([rng.permutation(K) for _ in range(H)])
    B = np.empty_like(A)
    for j in range(H):
        R = np.linalg.qr(rng.standard_normal((D, D)))[0]
        B[j] = A[heads[j]][entries[j]] @ R + noise * rng.standard_normal((K, D))
    return A, B, heads, entries


def test_deviation_zeroes_the_mean_and_diagonal():
    G = gram(np.random.default_rng(0).standard_normal((2, K, D)))
    Dv = deviation(G)
    off = ~np.eye(K, dtype=bool)
    assert np.allclose(Dv[:, off].mean(-1), 0)
    assert np.allclose(np.einsum("hkk->hk", Dv), 0)


def test_qap_recovers_a_relabelled_gram():
    rng = np.random.default_rng(1)
    G = gram(rng.standard_normal((1, K, D)))[0]
    p = rng.permutation(K)
    c, _ = qap_order(G, G[np.ix_(p, p)], 4, rng)
    assert rms(G, G[np.ix_(p, p)][np.ix_(c, c)]) < 1e-9
    assert (p[c] == np.arange(K)).all()


def test_qap_finds_rotated_heads_the_atom_match_misses():
    A, B, heads, _ = rotated_copies(2)
    Ga, Gb = gram(A), gram(B)
    perm = entry_match(A, B)
    D_m = matched_rms(Ga, Gb, perm)
    D_g, D_b, _ = qap_distances(Ga, Gb, perm, 4, 0)
    assert pair_ratio(D_g, False) > 5
    assert pair_ratio(D_m, False) < 2
    # the true pairing of heads is the one the QAP's distances choose
    true = np.argsort(heads)
    assert (D_g[np.arange(H), true] < 0.02).all()


def test_independent_heads_read_near_chance():
    rng = np.random.default_rng(3)
    A, B = rng.standard_normal((H, K, D)), rng.standard_normal((H, K, D))
    Ga, Gb = gram(A), gram(B)
    D_g, _, _ = qap_distances(Ga, Gb, entry_match(A, B), 4, 0)
    assert pair_ratio(D_g, False) < 1.3


def test_best_never_reads_above_the_atom_match():
    A, B, _, _ = rotated_copies(4, noise=0.5)
    Ga, Gb = gram(A), gram(B)
    perm = entry_match(A, B)
    D_m = matched_rms(Ga, Gb, perm)
    _, D_b, _ = qap_distances(Ga, Gb, perm, 2, 0)
    assert (D_b <= D_m + 1e-12).all()


def test_entries_recovered_reads_identity_orders():
    orders = np.tile(np.arange(K), (H, H, 1))
    assert entries_recovered(orders) == 1.0
    orders[0, 0] = orders[0, 0][::-1]
    assert entries_recovered(orders) == 1 - 1 / H

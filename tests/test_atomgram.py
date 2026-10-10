# experiments/ste-arm/atomgram.py: comparing heads' atom Grams modulo the
# order each head stores its entries in. Relabeling entries changes
# nothing a model computes, so every comparison has to say how it fixes
# that order; these pin each helper's handling of it.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "ste-arm"))
from atomgram import (atom_agreement, canonical, centered,  # noqa: E402
                      entry_match, gram, matched_rms, offdiag_rms,
                      pair_ratio, paired_mask, recovers, reorder)

H, K, D = 4, 6, 16


def atoms(seed=0):
    return np.random.default_rng(seed).normal(size=(H, K, D))


def relabeled(A, seed=1):
    """A with each head's entries in a random order, and those orders."""
    rng = np.random.default_rng(seed)
    perm = np.stack([rng.permutation(K) for _ in range(H)])
    return np.take_along_axis(A, perm[:, :, None], 1), perm


def test_gram_is_a_cosine_matrix():
    G = gram(atoms())
    assert np.allclose(np.einsum("hkk->hk", G), 1.0)
    assert np.allclose(G, G.transpose(0, 2, 1))
    assert np.all(np.abs(G) <= 1 + 1e-12)


def test_centered_removes_the_usage_weighted_mean():
    A = atoms()
    u = np.random.default_rng(2).random((H, K))
    C = centered(A, u)
    assert np.allclose(np.einsum("hk,hkd->hd", u, C), 0.0)


def test_canonical_order_is_relabeling_invariant():
    """Sorting by mean cosine reads the same Gram whatever the stored
    order, when the mean cosines are distinct."""
    A = atoms()
    B, _ = relabeled(A)
    Ga, Gb = gram(A), gram(B)
    assert np.allclose(reorder(Ga, canonical(Ga)), reorder(Gb, canonical(Gb)))
    mean = (Ga.sum(-1) - 1) / (K - 1)
    assert np.all(np.diff(np.take_along_axis(mean, canonical(Ga), 1), axis=1)
                  <= 0)


def test_offdiag_rms_ignores_the_diagonal():
    rng = np.random.default_rng(3)
    G1, G2 = rng.normal(size=(2, K, K)), rng.normal(size=(3, K, K))
    off = ~np.eye(K, dtype=bool)
    want = np.sqrt(((G1[:, None] - G2[None])[:, :, off] ** 2).mean(-1))
    assert np.allclose(offdiag_rms(G1, G2), want)
    G3 = G1.copy()
    G3[:, np.arange(K), np.arange(K)] += 5.0
    assert np.allclose(np.diag(offdiag_rms(G1, G3)), 0.0, atol=1e-6)


def test_entry_match_undoes_a_relabeling():
    A = atoms()
    B, perm = relabeled(A)
    P = entry_match(A, B)
    for i in range(H):
        # B[i][P[i, i]] lines up with A[i]: entry P[i, i, r] of B is A's r
        assert np.array_equal(perm[i][P[i, i]], np.arange(K))
    Ga, Gb = gram(A), gram(B)
    assert np.allclose(np.diag(matched_rms(Ga, Gb, P)), 0.0, atol=1e-12)
    # stored order compares the relabeled Grams entry by entry, and fails
    assert np.all(np.diag(offdiag_rms(Ga, Gb)) > 0.1)


def test_pairing_is_the_identity_within_a_run_and_assigned_across():
    rng = np.random.default_rng(4)
    D = 1.0 + rng.random((H, H))
    swap = np.array([2, 0, 3, 1])
    D[np.arange(H), swap] = 0.1
    assert np.array_equal(np.argwhere(paired_mask(D, True))[:, 1],
                          np.arange(H))
    assert np.array_equal(np.argwhere(paired_mask(D, False))[:, 1], swap)
    m = paired_mask(D, False)
    assert np.isclose(pair_ratio(D, False), D[~m].mean() / 0.1)
    assert recovers(D) == 0.0
    assert recovers(D[:, swap]) == 1.0           # minima on the diagonal


def test_atom_agreement_matches_relabeled_atoms_and_skips_dead_ones():
    A = atoms()
    B, _ = relabeled(A)
    live = np.ones((H, K), dtype=bool)
    paired, other, stored = atom_agreement(A, B, live, live, False)
    assert np.isclose(paired, 1.0)
    assert other < 0.5
    assert np.isnan(stored)
    # within a run the stored order is the correspondence
    _, _, stored = atom_agreement(A, A, live, live, True)
    assert np.isclose(stored, 1.0)
    # a dead entry's atom never enters the match
    B2 = A.copy()
    B2[:, 0] = -A[:, 0]
    dead = live.copy()
    dead[:, 0] = False
    assert np.isclose(atom_agreement(A, B2, dead, dead, True)[2], 1.0)
    assert atom_agreement(A, B2, live, live, True)[2] < 1.0

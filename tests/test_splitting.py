# splitting.py's containment analysis: planted parent/child structure
# (children partition each parent's firing set) must be recovered with the
# right multiplicity, full coverage, zero overlap, and duplicates
# classified as matches rather than children.
import numpy as np
import jax.numpy as jnp
import pytest

import splitting

N = 1200
M1, M2 = 4, 13          # 4 parents; 3 children each + 1 duplicate of parent 0
SPAN = N // M1          # each parent fires on its own quarter of the rows


def planted():
    Fa = np.zeros((N, M1), np.float32)
    Fb = np.zeros((N, M2), np.float32)
    for i in range(M1):
        rows = np.arange(i * SPAN, (i + 1) * SPAN)
        Fa[rows, i] = 1.0
        for c in range(3):  # children partition the parent's rows in thirds
            Fb[rows[c::3], i * 3 + c] = 1.0
    Fb[:SPAN, 12] = 1.0     # exact duplicate of parent 0
    return Fa, Fb


def stats(Fa, Fb):
    return Fa.T @ Fb, Fb.T @ Fb, Fa.sum(0), Fb.sum(0)


def test_split_children_recovers_planted_structure():
    Fa, Fb = planted()
    C, Cbb, na, nb = stats(Fa, Fb)
    children, matches = splitting.split_children(C, na, nb, tau=0.5,
                                                 min_fires=10)
    for i in range(M1):
        assert set(children[i]) == {3 * i, 3 * i + 1, 3 * i + 2}, i
    assert list(matches[0]) == [12]      # the duplicate is a match...
    assert 12 not in children[0]         # ...not a proper child
    assert all(len(matches[i]) == 0 for i in range(1, M1))


def test_coverage_and_overlap_of_planted_partition():
    Fa, Fb = planted()
    C, Cbb, na, nb = stats(Fa, Fb)
    children, _ = splitting.split_children(C, na, nb, 0.5, 10)

    M = np.zeros((M2, M1), np.float32)
    for i, ch in enumerate(children):
        M[ch, i] = 1.0
    cover = splitting.union_cover_counts(
        jnp.asarray(Fa), jnp.asarray(Fb), jnp.asarray(M)) / na
    assert np.allclose(cover, 1.0)       # children tile the parent exactly

    ov = splitting.child_overlap(Cbb, nb, children)
    assert np.allclose(ov, 0.0)          # and are mutually disjoint


def test_min_fires_excludes_rare_children():
    Fa, Fb = planted()
    C, Cbb, na, nb = stats(Fa, Fb)
    children, _ = splitting.split_children(C, na, nb, 0.5,
                                           min_fires=SPAN)  # > child rate
    assert all(len(c) == 0 for c in children)


def test_match_fractions():
    Fa, Fb = planted()
    C, Cbb, na, nb = stats(Fa, Fb)
    _, matches = splitting.split_children(C, na, nb, 0.5, 10)
    live_a = na >= 10
    fa, fb = splitting.match_fractions(matches, live_a, nb, 10)
    assert fa == 0.25       # only parent 0 has its duplicate
    assert fb == 1.0 / 13   # only the duplicate latent is matched


def test_absorption_candidates():
    rng = np.random.default_rng(0)
    W_a = rng.normal(size=(2, 16)).astype(np.float32)
    W_b = rng.normal(size=(3, 16)).astype(np.float32)
    W_b[0] = W_a[0]                       # geometric twin of parent 0
    An = W_a / np.linalg.norm(W_a, axis=1, keepdims=True)
    Bn = W_b / np.linalg.norm(W_b, axis=1, keepdims=True)
    cos = An @ Bn.T
    pa = np.zeros((2, 3))                 # ...that never co-fires
    nb = np.full(3, 100.0)
    got = splitting.absorption_candidates(cos, pa, nb, min_fires=10)
    assert got[0] == 1 and got[1] == 0


def unit_rows(W):
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def test_decoder_shares_read_a_reseeded_copy_as_shared():
    # B is A's decoder permuted and slightly perturbed: every latent has a
    # near twin; an unrelated random decoder in 64 dimensions has none
    rng = np.random.default_rng(1)
    A = unit_rows(rng.normal(size=(20, 64)))
    B = unit_rows(A[rng.permutation(20)] + 0.05 * rng.normal(size=(20, 64)))
    live = np.ones(20, bool)
    shares = splitting.decoder_shares(A @ B.T, live, live, [0.7, 0.9])
    assert shares[0.7] == (1.0, 1.0) and shares[0.9] == (1.0, 1.0)
    R = unit_rows(rng.normal(size=(20, 64)))
    assert splitting.decoder_shares(A @ R.T, live, live, [0.7])[0.7] == (0.0, 0.0)
    # dead latents are left out on both sides: A's live half all have twins,
    # but only half of B's latents have a twin among them
    half = np.arange(20) < 10
    assert splitting.decoder_shares(A @ B.T, half, live, [0.7])[0.7] == (1.0, 0.5)


def test_pw_mcc_is_one_for_a_signed_permutation_and_low_for_random():
    rng = np.random.default_rng(2)
    A = unit_rows(rng.normal(size=(30, 64)))
    perm = rng.permutation(30)
    sign = np.where(rng.random(30) < 0.5, -1.0, 1.0)[:, None]
    mcc, matched = splitting.pw_mcc(A @ (sign * A[perm]).T)
    assert mcc == pytest.approx(1.0) and len(matched) == 30
    R = unit_rows(rng.normal(size=(30, 64)))
    assert splitting.pw_mcc(A @ R.T)[0] < 0.5


def test_max_corr_from_running_sums():
    # B's latent 0 copies A's latent 1 up to scale and shift; B's latent 1
    # is independent noise; A's latent 2 never varies (no correlation)
    rng = np.random.default_rng(3)
    n = 4000
    Aa = np.stack([rng.random(n), rng.random(n), np.full(n, 0.5)], 1)
    Ab = np.stack([3.0 * Aa[:, 1] + 1.0, rng.random(n)], 1)
    best_a, best_b = splitting.max_corr(n, Aa.sum(0), Ab.sum(0), (Aa ** 2).sum(0),
                                        (Ab ** 2).sum(0), Aa.T @ Ab)
    assert best_a[1] == pytest.approx(1.0) and best_b[0] == pytest.approx(1.0)
    assert abs(best_a[0]) < 0.1 and np.isnan(best_a[2])
    np.testing.assert_allclose(best_a[:2], [max(np.corrcoef(Aa[:, 0], Ab[:, j])[0, 1]
                                                for j in range(2)), 1.0], atol=1e-9)

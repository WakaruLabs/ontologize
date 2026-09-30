# splitting.py's containment analysis: planted parent/child structure
# (children partition each parent's firing set) must be recovered with the
# right multiplicity, full coverage, zero overlap, and duplicates
# classified as matches rather than children.
import numpy as np
import jax.numpy as jnp

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

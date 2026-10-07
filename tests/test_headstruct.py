# headstruct.py's discovery machinery: latents with paradigmatic structure
# (fire in the same contexts, never together) must be recovered as groups,
# and the head metrics must separate a true categorical partition from a
# size-matched random one.
import numpy as np
import jax.numpy as jnp

import headstruct

M, G, S, N = 40, 8, 5, 4096  # latents, planted groups, group size, samples


def planted_fires(seed=0):
    """Perfect categorical data: each sample fires exactly one member of
    each planted group (latent g*S+j belongs to group g)."""
    rng = np.random.default_rng(seed)
    F = np.zeros((N, M), np.float32)
    for g in range(G):
        F[np.arange(N), g * S + rng.integers(0, S, N)] = 1.0
    return F


def test_discovery_recovers_planted_groups():
    F = planted_fires()
    C = F.T @ F
    ii, jj, vals = headstruct.affinity_edges(C, F.sum(0), N, min_fire=1e-3,
                                             topn=16)
    labels = headstruct.greedy_groups(ii, jj, vals, M, cap=S)

    planted = np.arange(M) // S
    assert (labels >= 0).all()  # every latent grouped
    for g in range(G):
        members = labels[planted == g]
        assert (members == members[0]).all(), f"group {g} split"
    # no two planted groups merged
    assert len(np.unique(labels)) == G


def test_greedy_groups_respects_cap():
    F = planted_fires(1)
    C = F.T @ F
    ii, jj, vals = headstruct.affinity_edges(C, F.sum(0), N, topn=16)
    for cap in (3, 5):
        labels = headstruct.greedy_groups(ii, jj, vals, M, cap)
        sizes = np.bincount(labels[labels >= 0])
        assert sizes.max() <= cap


def test_metrics_separate_true_partition_from_null():
    F = planted_fires(2)
    z = jnp.asarray(F)  # activation == fire indicator
    planted = np.arange(M) // S

    rng = np.random.default_rng(0)
    null = planted[rng.permutation(M)]

    def score(labels):
        mom = headstruct.group_moments(z, z, labels, G)
        return headstruct.metrics(*(np.asarray(a) for a in mom), N)

    exh, exc, cv = score(planted)
    assert np.allclose(exh, 1.0)   # every group fires on every sample
    assert np.allclose(exc, 1.0)   # exactly one member at a time
    assert np.allclose(cv, 0.0, atol=1e-4)  # summed activation constant

    exh_n, exc_n, cv_n = score(null)
    # a random partition fires multiple members together and misses samples
    assert exc_n.mean() > 1.2
    assert cv_n.mean() > 0.2
    assert exh_n.mean() < 1.0


def random_graph(n, seed):
    rng = np.random.default_rng(seed)
    D = rng.random((n, n)).astype(np.float32)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


def test_soft_modularity_hard_matches_newman():
    # one-hot cells over a sparse group: soft Q must equal the textbook
    # sum over the fired subgraph of (D_ij - gamma k_i k_j / 2m) delta(c_i, c_j)
    n, gamma = 60, 1.3
    rng = np.random.default_rng(3)
    D = random_graph(n, 4)
    c = rng.integers(0, 3, n)
    fired = rng.random(n) < 0.7
    z = np.zeros((n, 3), np.float32)
    z[np.arange(n)[fired], c[fired]] = rng.random(fired.sum()) + 0.5
    Q = headstruct.soft_modularity(jnp.asarray(D), jnp.asarray(z),
                                   jnp.asarray((z > 0).astype(np.float32)),
                                   np.zeros(3, np.int64), 1, gamma)

    Ds = D[np.ix_(fired, fired)]
    cs = c[fired]
    k = Ds.sum(1)
    two_m = k.sum()
    same = cs[:, None] == cs[None, :]
    ref = ((Ds - gamma * np.outer(k, k) / two_m) * same).sum() / two_m
    assert np.allclose(Q[0], ref, atol=1e-5)


def test_soft_modularity_one_cell_is_zero_and_silent_group_nan():
    n = 50
    D = jnp.asarray(random_graph(n, 5))
    z = np.zeros((n, 4), np.float32)
    z[:, 0] = 1.0                   # group 0 = latents 0,1: always latent 0
    labels = np.array([0, 0, 1, 1])  # group 1 = latents 2,3: never fires
    Q = headstruct.soft_modularity(D, jnp.asarray(z), jnp.asarray(z), labels, 2)
    assert np.isclose(Q[0], 0.0, atol=1e-6)
    assert np.isnan(Q[1])


def test_knn_graph_is_symmetric_sparse_and_keeps_weights():
    n, k = 40, 5
    D = jnp.asarray(random_graph(n, 7))
    K = np.asarray(headstruct.knn_graph(D, k))
    assert np.allclose(K, K.T)
    assert (np.diag(K) == 0).all()
    assert ((K > 0).sum(1) >= k).all()
    assert (K > 0).sum() < n * (n - 1)          # actually sparser than dense
    nz = K > 0
    assert np.allclose(K[nz], np.asarray(D)[nz])  # kernel weights kept
    # each row's own k nearest all survive
    top = np.argsort(-np.asarray(D), 1)[:, :k]
    assert nz[np.arange(n)[:, None], top].all()


def test_knn_sharpens_modularity_of_true_clusters():
    # two overlapping blobs on the sphere: under the dense heat kernel every
    # pair has similar weight and the true split scores little; the kNN
    # graph exposes it
    from ontologize.fns.pwak import affinity
    rng = np.random.default_rng(8)
    n, d = 400, 16
    c = rng.integers(0, 2, n)
    X = np.eye(d)[c] + 0.6 * rng.standard_normal((n, d))
    D = affinity(jnp.asarray(X, jnp.float32), 0.2)
    z = np.zeros((n, 2), np.float32)
    z[np.arange(n), c] = 1.0
    z = jnp.asarray(z)
    labels = np.zeros(2, np.int64)
    dense = headstruct.soft_modularity(D, z, z, labels, 1)[0]
    knn = headstruct.soft_modularity(headstruct.knn_graph(D, 10), z, z,
                                     labels, 1)[0]
    assert knn > dense + 0.1
    assert knn > 0.3


def test_affinity_edges_block_keeps_pairs_inside_blocks():
    F = planted_fires(10)
    C = F.T @ F
    block = 2 * S                     # two planted groups per block
    ii, jj, _ = headstruct.affinity_edges(C, F.sum(0), N, topn=16, block=block)
    assert len(ii) > 0
    assert (ii // block == jj // block).all()
    labels = headstruct.greedy_groups(*headstruct.affinity_edges(
        C, F.sum(0), N, topn=16, block=block), M, S)
    for g in np.unique(labels[labels >= 0]):
        assert len(np.unique(np.arange(M)[labels == g] // block)) == 1


def test_block_modularity_handles_arbitrary_group_ids():
    # same layout as the per-layer test below, but groups numbered out of
    # block order and one block holding no group
    rng = np.random.default_rng(11)
    n, k = 300, 3
    cells = rng.integers(0, k, n)
    z = np.zeros((n, 3 * k), np.float32)
    z[np.arange(n), k + cells] = 1.0          # only block 1 fires
    z = jnp.asarray(z)
    D = np.where(cells[:, None] == cells[None, :], 1.0, 0.0).astype(np.float32)
    np.fill_diagonal(D, 0.0)
    D = jnp.asarray(D)
    labels = np.array([-1] * k + [4] * k + [-1] * k)
    Q = headstruct.block_modularity([D, D, D], z, z, labels, 5, k)
    assert Q.shape == (5,)
    assert Q[4] > 0.6
    assert np.isnan(Q[:4]).all()


def test_layer_modularity_scores_each_layer_on_its_own_graph():
    # two layers of one head with three entries each, each layer's cells
    # independent of the other's; graph L links samples sharing layer L's
    # cell, so the right pairing scores high and the swapped one near 0
    rng = np.random.default_rng(9)
    n, k = 300, 3
    cells = rng.integers(0, k, (2, n))
    z = np.zeros((n, 2 * k), np.float32)
    for L in range(2):
        z[np.arange(n), L * k + cells[L]] = 1.0
    z = jnp.asarray(z)

    def block(c):
        D = np.where(c[:, None] == c[None, :], 1.0, 0.0).astype(np.float32)
        np.fill_diagonal(D, 0.0)
        return jnp.asarray(D)

    Ds = [block(cells[0]), block(cells[1])]
    labels = np.arange(2 * k) // k
    right = headstruct.block_modularity(Ds, z, z, labels, 2, k)
    swapped = headstruct.block_modularity(Ds[::-1], z, z, labels, 2, k)
    assert right.shape == (2,)
    assert (right > 0.6).all()              # 1 - 1/k for pure blocks
    assert (np.abs(swapped) < 0.1).all()


def test_soft_modularity_prefers_partition_aligned_with_graph():
    # the graph links samples that fired the same member of planted group
    # 0, so that group's partition is a community structure and a
    # size-matched random regrouping of the latents is not
    F = planted_fires(6)[:512]
    z = jnp.asarray(F)
    cell = F[:, :S].argmax(1)
    D = np.where(cell[:, None] == cell[None, :], 1.0, 0.05).astype(np.float32)
    np.fill_diagonal(D, 0.0)
    D = jnp.asarray(D)
    planted = np.arange(M) // S

    Q = headstruct.soft_modularity(D, z, z, planted, G)
    assert Q[0] > 0.5                       # 1 - 1/S = 0.8 at zero background

    rng = np.random.default_rng(1)
    nulls = [headstruct.soft_modularity(D, z, z, planted[rng.permutation(M)], G)
             for _ in range(5)]
    assert Q[0] > max(np.nanmax(q) for q in nulls)

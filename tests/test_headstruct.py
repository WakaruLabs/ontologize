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

# experiments/ste-arm/coselect.py: the cosine between atoms a row selects
# together, beside the dictionary's and the usage null. The tests plant
# dictionaries where the answer is known: heads that avoid one another's
# directions read below the null, independent heads read at it, and the
# SAE helpers do the same for active sets.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "ste-arm"))
from coselect import (active_sets, center, coselection, draw_sets,  # noqa: E402
                      head_pair_means, pair_groups, sae_coselection,
                      selected_means, set_mean, unit, usage_of)

K = 4


def planted(n: int, rng: np.random.Generator, avoid: bool):
    """Two heads over one orthonormal basis: head 1's entry j is head 0's
    entry j + 1. Selecting the same index in both (`avoid`) always pairs
    orthogonal atoms; independent selections pair equal atoms a quarter
    of the time."""
    E = np.eye(K)
    D = np.stack([E, np.roll(E, -1, axis=0)])            # (2, K, K)
    a = rng.integers(0, K, n)
    b = a if avoid else rng.integers(0, K, n)
    return D, np.stack([a, b])


def test_heads_that_avoid_each_other_read_below_the_null():
    rng = np.random.default_rng(0)
    D, I = planted(4096, rng, avoid=True)
    r = coselection(D, I, np.zeros(2, int), 10, rng, True)["all"]
    assert r["selected"] == 0.0
    assert np.isclose(r["dictionary"], 0.25)
    assert abs(r["usage"] - 0.25) < 0.02
    assert r["selected"] < r["usage"] - 5 * r["usage_sd"]


def test_independent_heads_read_at_the_null():
    rng = np.random.default_rng(1)
    D, I = planted(4096, rng, avoid=False)
    r = coselection(D, I, np.zeros(2, int), 20, rng, True)["all"]
    assert abs(r["selected"] - r["usage"]) < 4 * r["usage_sd"]


def test_usage_null_is_the_expectation_of_a_permutation():
    rng = np.random.default_rng(2)
    s, n = 3, 20000
    D = rng.normal(size=(s, K, 6))
    I = np.stack([rng.choice(K, n, p=[0.7, 0.1, 0.1, 0.1]) for _ in range(s)])
    U = unit(D).reshape(s * K, -1)
    G = np.abs(U @ U.T)
    exact = head_pair_means(G, usage_of(I, K))
    drawn = selected_means(G, np.stack([rng.permutation(r) for r in I]), K)
    off = ~np.eye(s, dtype=bool)
    assert np.allclose(exact[off], drawn[off], atol=0.01)
    # uniform weights give the plain mean over the two heads' atoms
    flat = head_pair_means(G, np.full((s, K), 1 / K))
    assert np.isclose(flat[0, 1], G[:K, K:2 * K].mean())


def test_pair_groups_split_layers():
    g = pair_groups(np.array([0, 0, 1, 1, 1]))
    assert g["layer 0"].sum() == 2 and g["layer 1"].sum() == 6
    assert g["across layers"].sum() == 12 and g["all"].sum() == 20
    assert "across layers" not in pair_groups(np.zeros(3, int))


def test_centering_removes_a_shared_offset():
    rng = np.random.default_rng(3)
    D = rng.normal(size=(2, K, 8)) + 5.0                  # offset dominates
    usage = np.full((2, K), 1 / K)
    raw = unit(D).reshape(2 * K, -1)
    assert np.abs(raw @ raw.T).mean() > 0.8
    C = center(D, usage)
    assert np.allclose(np.einsum("sk,skd->sd", usage, C), 0.0)


def test_sae_active_sets_and_their_null():
    rng = np.random.default_rng(4)
    n, m, a = 2000, 12, 3
    # latents 0-2 share a direction, as do 3-5, ...; a row activates one
    # latent from each of three different groups, so its active set never
    # holds two alike
    W = np.repeat(np.eye(4), 3, axis=0) + 0.01 * rng.normal(size=(m, 4))
    Z = np.zeros((n, m))
    for i in range(n):
        groups = rng.choice(4, a, replace=False)
        Z[i, groups * 3 + rng.integers(0, 3, a)] = 1.0
    S, freq = active_sets(Z)
    assert S.shape == (n, a) and np.isclose(freq.sum(), a)
    assert (np.sort(S, 1) == np.sort(np.argwhere(Z)[:, 1].reshape(n, a), 1)
            ).all()
    r = sae_coselection(W, Z, 10, rng, True)
    assert r["selected"] < 0.05 and r["usage"] > 0.1
    assert r["selected"] < r["usage"] - 5 * r["usage_sd"]
    # the drawn sets are distinct latents, in proportion to firing
    T = draw_sets(freq, 5000, a, rng)
    assert (np.sort(T, 1)[:, 1:] != np.sort(T, 1)[:, :-1]).all()
    never = draw_sets(np.array([0.5, 0.5, 0.0, 0.5]), 200, 2, rng)
    assert 2 not in never
    # set_mean leaves out a latent with itself
    assert np.isclose(set_mean(np.ones((m, m)), S), 1.0)
    assert np.isclose(set_mean(np.eye(m), S), 0.0)

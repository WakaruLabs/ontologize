# pareto.py's pure helpers: the deviation-code truncation that generates
# the Ontologizer's side of the capacity-Pareto curve, the SAE run-name
# parsing and index-bit count that pair each point with its capacity, and
# the Gaussian reference (reverse water-filling) the hard codes are
# compared against.
import math

import jax.numpy as jnp
import numpy as np
import pytest

import pareto

K = 8


def soft_batch(seed=0, shape=(16, 4, K)):
    rng = np.random.default_rng(seed)
    P = rng.exponential(size=shape).astype(np.float32)
    return jnp.asarray(P / P.sum(-1, keepdims=True))


def test_topdev_m0_is_origin():
    P = soft_batch()
    origin = np.full(K, 1.0 / K, np.float32)
    out = pareto.topdev(P, jnp.asarray(origin), 0, K)
    assert np.allclose(out, origin, atol=1e-6)


def test_topdev_full_m_is_identity():
    P = soft_batch()
    origin = np.full(K, 1.0 / K, np.float32)
    assert np.allclose(pareto.topdev(P, jnp.asarray(origin), K, K), P)


def test_topdev_pins_small_deviations_and_renormalizes():
    P = soft_batch(1)
    origin = soft_batch(2, (K,))
    m = 3
    out = np.asarray(pareto.topdev(P, origin, m, K))
    assert np.allclose(out.sum(-1), 1.0, atol=1e-5)

    D = np.abs(np.asarray(P) - np.asarray(origin))
    for idx in np.ndindex(P.shape[:-1]):
        keep = np.argsort(D[idx])[-m:]
        pinned = np.setdiff1d(np.arange(K), keep)
        # pre-renormalization values: kept entries carry p, pinned carry
        # the origin, so the ratio structure must match after one rescale
        raw = np.where(np.isin(np.arange(K), keep),
                       np.asarray(P)[idx], np.asarray(origin))
        assert np.allclose(out[idx], raw / raw.sum(), atol=1e-5), idx
        assert len(pinned) == K - m


def test_tophead_keeps_lowest_entropy_heads():
    P = soft_batch(3)              # (16, 4, K)
    origin = soft_batch(4, (K,))
    m = 2
    out = np.asarray(pareto.tophead(P, origin, m, P.shape[1]))
    assert np.allclose(out.sum(-1), 1.0, atol=1e-5)

    Pn = np.asarray(P)
    H = -(Pn * np.log(Pn + 1e-9)).sum(-1)  # (16, 4)
    for b in range(P.shape[0]):
        keep = np.argsort(H[b])[:m]
        for hd in range(P.shape[1]):
            want = Pn[b, hd] if hd in keep else np.asarray(origin)
            assert np.allclose(out[b, hd], want, atol=1e-6), (b, hd)


def test_tophead_endpoints():
    P = soft_batch(5)
    origin = soft_batch(6, (K,))
    h = P.shape[1]
    assert np.allclose(pareto.tophead(P, origin, 0, h),
                       np.broadcast_to(np.asarray(origin), P.shape))
    assert np.allclose(pareto.tophead(P, origin, h, h), P)


def test_parse_sae_name():
    assert pareto.parse_sae_name("x/m5120_k32/params.npz") == (5120, 32)
    assert pareto.parse_sae_name("m11264_k128/params.npz") == (11264, 128)
    # L1-mode runs have no fixed topk; capacity comes from measured L0
    assert pareto.parse_sae_name("x/m5120_l10.003/params.npz") == (5120, 0)
    with pytest.raises(ValueError):
        pareto.parse_sae_name("x/checkpoint_7/params.npz")


@pytest.mark.parametrize("n,r", [(10, 3), (32, 4), (5120, 32), (11264, 160),
                                 (7, 0), (7, 7)])
def test_log2_choose_is_the_binomial(n, r):
    assert pareto.log2_choose(n, r) == pytest.approx(
        math.log2(math.comb(n, r)), abs=1e-6)


def test_log2_choose_one_is_an_index():
    # one of n is a plain index; more than one is cheaper than r indices,
    # because the order they come in carries no information
    assert pareto.log2_choose(32, 1) == pytest.approx(5.0)
    assert pareto.log2_choose(32, 4) < 4 * math.log2(32)


def test_sae_index_bits():
    # a top-k support is a k-subset of the latents
    assert pareto.sae_index_bits(5120, 32.0) == round(
        math.log2(math.comb(5120, 32)))
    # a dense code names nothing
    assert pareto.sae_index_bits(5120, 5120.0) == 0
    assert pareto.sae_index_bits(5120, 5120.0, 160, "softmax") == 0
    # grouped top-1 with every group firing: one winner of 32 per group
    assert pareto.sae_index_bits(5120, 160.0, 160, "top1") == 800
    # ties can light two latents in a group; it still names one winner
    assert pareto.sae_index_bits(5120, 160.6, 160, "top1") == 800
    # silent groups have to be named too
    assert pareto.sae_index_bits(5120, 150.0, 160, "top1") == round(
        math.log2(math.comb(160, 150)) + 150 * 5)


def test_gaussian_reference_one_component():
    # a scalar Gaussian loses a factor of 4 in distortion per bit
    out = pareto.gaussian_reference(np.array([3.0]), np.array([0.0, 1.0, 2.5]))
    assert np.allclose(out, [1.0, 0.25, 2.0 ** -5], rtol=1e-6)


def test_gaussian_reference_equal_components_share_the_rate():
    d = 8
    out = pareto.gaussian_reference(np.full(d, 0.7), np.array([4.0, 16.0]))
    assert np.allclose(out, 2.0 ** (-2 * np.array([4.0, 16.0]) / d), rtol=1e-6)


def test_gaussian_reference_water_level_between_components():
    # at half a bit the water level 2 sits between the eigenvalues 4 and 1:
    # only the first component is coded, the second stays at its variance
    out = pareto.gaussian_reference(np.array([4.0, 1.0, 0.0]), np.array([0.5]))
    assert np.allclose(out, [(2.0 + 1.0) / 5.0], rtol=1e-6)


def test_gaussian_reference_monotone_and_bounded():
    lam = np.random.default_rng(7).exponential(size=64)
    out = pareto.gaussian_reference(lam, np.geomspace(1, 2000, 50))
    assert np.all(np.diff(out) < 0)
    assert np.all((out > 0) & (out < 1))


def test_whitened_spectrum_matches_the_fvu_base():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(4096, 6)) @ rng.normal(size=(6, 6)) + 3.0
    w = rng.uniform(0.5, 2.0, size=6)
    lam = pareto.whitened_spectrum(X, w)
    assert np.all(np.diff(lam) <= 0) and lam.min() >= 0
    # main() divides by mean(var * w), which is the spectrum's mean
    assert np.isclose(lam.sum() / len(lam), (X.var(0) * w).mean(), rtol=1e-6)

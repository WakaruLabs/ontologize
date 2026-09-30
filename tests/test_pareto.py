# pareto.py's pure helpers: the deviation-code truncation that generates
# the Ontologizer's side of the capacity-Pareto curve, and the SAE
# run-name parsing that pairs each point with its capacity.
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

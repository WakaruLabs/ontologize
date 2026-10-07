# bilinspec.py's closed-form spectra against dense eigendecompositions,
# including BilinearBlock's own `decompose`, plus its layout and step
# helpers and one set of figures from synthetic weights.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bilinspec
from ontologize.layers.nlinear import BilinearBlock


def sym(w, v):
    return (np.outer(w, v) + np.outer(v, w)) / 2


def test_pair_eigs_match_eigvalsh():
    rng = np.random.default_rng(0)
    for _ in range(5):
        w, v = rng.normal(size=(2, 17))
        lam = np.linalg.eigvalsh(sym(w, v))
        lp, lm = bilinspec.pair_eigs(w, v)
        np.testing.assert_allclose([lm, lp], [lam[0], lam[-1]], atol=1e-10)
        np.testing.assert_allclose(lam[1:-1], 0, atol=1e-10)
        a, b = bilinspec.spectrum_stats(lam), \
            bilinspec.spectrum_stats(np.array([lp, lm]))
        for k in a:
            assert a[k] == pytest.approx(b[k], abs=1e-10)
        c = bilinspec.pair_cos(w, v)
        assert b["top"] == pytest.approx((1 + abs(c)) / 2)
        assert b["neg"] == pytest.approx((1 - c) / 2)
        assert b["erank"] == pytest.approx(2 / (1 + c ** 2))


@pytest.mark.parametrize("s", [2.0, -0.5])
def test_parallel_pair_is_rank_one(s):
    w = np.array([1.0, -2.0, 0.5])
    lp, lm = bilinspec.pair_eigs(w, s * w)
    assert abs(bilinspec.pair_cos(w, s * w)) == pytest.approx(1)
    # one eigenvalue vanishes; the other carries the sign of s
    assert min(abs(lp), abs(lm)) == pytest.approx(0, abs=1e-12)
    assert np.sign(lp + lm) == np.sign(s)


def test_orthogonal_pair_is_a_balanced_saddle():
    w, v = np.array([1.0, 0, 0]), np.array([0, 3.0, 0])
    lp, lm = bilinspec.pair_eigs(w, v)
    assert lm == pytest.approx(-lp)
    assert bilinspec.pair_cos(w, v) == 0
    st = bilinspec.spectrum_stats(np.array([lp, lm]))
    assert st["top"] == pytest.approx(0.5) and st["neg"] == pytest.approx(0.5)
    assert st["erank"] == pytest.approx(2)


def test_matches_bilinearblock_decompose():
    blk = BilinearBlock(d_in=6, d_out=4, h=3, biased=False,
                        dtype_str="float32", dtype_p_str="float32")
    params = blk.init(jax.random.PRNGKey(0), jnp.zeros((1, 6)))
    vals, _ = blk.apply(params, method=BilinearBlock.decompose)
    vals = np.asarray(vals, np.float64)  # (h, k, d), ascending
    Wc = np.asarray(params["params"]["weight"], np.float64)  # (2, h, k, d)
    ref = bilinspec.pair_eigs(Wc[0], Wc[1])
    np.testing.assert_allclose(ref[..., 0], vals[..., -1], atol=1e-5)
    np.testing.assert_allclose(ref[..., 1], vals[..., 0], atol=1e-5)
    st = bilinspec.spectrum_stats(vals)
    st2 = bilinspec.spectrum_stats(ref)
    np.testing.assert_allclose(st["top"], st2["top"], atol=1e-5)
    np.testing.assert_allclose(st["neg"], st2["neg"], atol=1e-5)
    # the logit itself is the pair's quadratic form
    x = np.random.default_rng(1).normal(size=6)
    K = np.asarray(blk.apply(params, jnp.asarray(x, jnp.float32)))
    q = np.einsum("hkd,hke,d,e->hk", Wc[0], Wc[1], x, x)
    np.testing.assert_allclose(K, q, rtol=1e-4, atol=1e-5)


def test_odd_share_against_monte_carlo():
    rng = np.random.default_rng(2)
    d = 5
    w, v = rng.normal(size=(2, d))
    a, b = 0.4, -0.7
    U = rng.normal(size=(400000, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    q = (U @ w + a) * (U @ v + b)
    odd = (q - ((-U) @ w + a) * ((-U) @ v + b)) / 2
    mc = (odd ** 2).mean() / q.var()
    got = bilinspec.odd_share(w, v, np.array(a), np.array(b))
    assert got == pytest.approx(mc, rel=0.02)
    assert bilinspec.odd_share(w, v, np.array(0.0), np.array(0.0)) == 0


def test_layer_layout():
    spec = dict(forward="resid", resid_const=True, const0=False, k=4, h=3)
    assert bilinspec.layer_layout(spec, 0, 10) == (10, False)
    assert bilinspec.layer_layout(spec, 1, 11) == (10, True)
    assert bilinspec.layer_layout({**spec, "const0": True}, 0, 11) == (10, True)
    assert bilinspec.layer_layout({**spec, "forward": "resid_labels"},
                                  1, 23) == (10, True)
    assert bilinspec.layer_layout({**spec, "forward": "labels"},
                                  1, 12) == (12, False)


def test_feature_stats_const_split():
    # w = (w_s, a): the semantic cosine ignores the constant coordinate
    Wc = np.zeros((2, 1, 1, 4))
    Wc[0, 0, 0] = [1, 0, 0, 2.0]
    Wc[1, 0, 0] = [-1, 0, 0, 0.5]
    st = bilinspec.feature_stats(Wc, 3, True)
    assert st["cos"][0, 0] == pytest.approx(-1)
    assert st["rho_w"][0, 0] == pytest.approx(2)
    assert st["rho_v"][0, 0] == pytest.approx(0.5)
    assert abs(st["cos_full"][0, 0]) < 1


def test_pick_steps():
    steps = list(range(10000, 1000001, 10000))
    got = bilinspec.pick_steps(steps, 5)
    assert got[0] == 10000 and got[-1] == 1000000 and len(got) <= 5
    assert got == sorted(got)
    assert bilinspec.pick_steps(steps, 0) == steps
    assert bilinspec.pick_steps([5, 7], 8) == [5, 7]


def test_figures(tmp_path):
    rng = np.random.default_rng(3)
    s, l, h, k = 3, 2, 4, 5
    cos = rng.uniform(-1, 1, (s, l, h, k))
    odd = rng.uniform(0, 1, (s, l, h, k))
    odd[:, 0] = 0
    z = dict(cos=cos, odd=odd, steps=np.array([10, 100, 1000]), run="toy",
             has_const=np.array([False, True]), d_sem=np.array([8, 8]))
    bilinspec.draw_all(z, tmp_path)
    for name in ("bilin_ecdf.png", "bilin_heads.png", "bilin_steps.png"):
        assert (tmp_path / name).stat().st_size > 0

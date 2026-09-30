# PWAK noise2self partition score (s_L2pwak / l2pwak_loss / L2_pwak stat):
# the error of predicting each layer's own input from its partition-gated
# neighbourhood, as a fraction of the batch's spread. Unlike KL_pwak it
# scores the CLUSTERING -- the layer input is stop-gradiented, so the only
# gradient path is the partition gate PP^T, and the dictionary is untouched.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.fns.pwak import diffuse, pwak_l2
from ontologize.ontologizer import Ontologizer
from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import update

from conftest import KW, B

T = 0.5
L2P = 9  # stats column: [L1_K, L1_F, H, cossim_b, cossim_h,
         #                cossim_k, KL_m, KL_pwak, L2_pwak]


@pytest.fixture(scope="module")
def clustered():
    """Points in c tight clusters, with a partition that matches them and
    one that deliberately spreads every cluster across all tags."""
    c, n, d, h, k = 4, 16, 8, 3, 4
    rng = np.random.default_rng(0)
    ctr = rng.normal(size=(c, d))
    ctr /= np.linalg.norm(ctr, axis=1, keepdims=True)
    lab = np.repeat(np.arange(c), n)
    E = jnp.asarray(ctr[lab] + 0.05 * rng.normal(size=(c * n, d)), jnp.float32)

    def onehot(idx):
        P = np.zeros((c * n, h, k), np.float32)
        P[np.arange(c * n), :, idx] = 1.0
        return jnp.asarray(P)

    return E, onehot(lab), onehot(np.arange(c * n) % c)


def test_inert_at_zero_depth(clustered):
    # diffusion is the identity at s=0, so the term contributes nothing
    # until the schedule ramps -- diffusing while the partition is still
    # noise would lock the noise in
    E, good, bad = clustered
    assert float(pwak_l2(good, E, 0)) == 0.0
    assert float(pwak_l2(bad, E, 0)) == 0.0


def test_true_partition_beats_scrambled(clustered):
    E, good, bad = clustered
    assert float(pwak_l2(good, E, 1, 1.0)) < 0.1
    assert float(pwak_l2(bad, E, 1, 1.0)) > 0.4


def test_depth_punishes_only_leaky_partitions(clustered):
    # the mechanism the hardening hypothesis rests on: a block-diagonal
    # partition keeps deeper walks inside the block, so depth is free;
    # a leaky one lets the walk escape, so depth costs more and more
    E, good, bad = clustered
    g = [float(pwak_l2(good, E, s, 1.0)) for s in (1, 2, 4)]
    b = [float(pwak_l2(bad, E, s, 1.0)) for s in (1, 2, 4)]
    assert max(g) - min(g) < 0.02                  # flat in s
    assert b[0] < b[1] < b[2]                      # monotone in s
    assert all(bi - gi > hi for bi, gi, hi in zip(b, g, (0.4, 0.6, 0.8)))


def test_scrambled_approaches_the_batch_mean(clustered):
    # 1.0 is the anchor: the graph does no better than predicting every
    # sample by the batch mean
    E, _, bad = clustered
    assert float(pwak_l2(bad, E, 4, 4.0)) == pytest.approx(1.0, abs=0.02)


def test_scale_free(clustered):
    # affinity unit-normalizes its rows, and numerator and denominator are
    # both quadratic in E, so the ratio is invariant to input scale
    E, good, _ = clustered
    assert float(pwak_l2(good, 7.0 * E, 2, 1.0)) == pytest.approx(
        float(pwak_l2(good, E, 2, 1.0)), rel=1e-4)


def test_diffusion_preserves_constants(clustered):
    # wak makes the gate row-stochastic, so diffusion is an averaging
    # operator: a constant field is a fixed point at any depth
    E, good, _ = clustered
    C = jnp.ones((E.shape[0], 3))
    assert jnp.allclose(diffuse(good, C, 3, 1.0), C, atol=1e-5)


def l2p_sum(model, **kw):
    def f(params, X):
        _, stats, _ = model.apply(params, X, temperature=T, pwak_s=2,
                                  method=Ontologizer.withStats, **kw)
        return stats[:, L2P].sum()
    return f


def test_stat_is_live_and_gated(build, X):
    model_on, p_on = build(l2pwak_loss=True)
    model_off, p_off = build(l2pwak_loss=False)
    assert float(l2p_sum(model_on)(p_on, X)) > 0.0
    assert float(l2p_sum(model_off)(p_off, X)) == 0.0

    g = jax.grad(l2p_sum(model_off))(p_off, X)
    assert sum(float(jnp.abs(v).sum())
               for v in jax.tree_util.tree_leaves(g)) == 0.0


def test_gradient_scores_clustering_not_dictionary(build, X):
    # the defining property: the term reaches the classifier (it shapes the
    # partition) but the dictionary gets exactly nothing, because the stat
    # never reads `dicts()` and the layer input is stop-gradiented
    model, params = build(l2pwak_loss=True)
    g = jax.grad(l2p_sum(model))(params, X)

    def total(kind):
        # "dict" alone would also match the `dictencs_i` module names
        return sum(float(jnp.abs(v).sum())
                   for path, v in jax.tree_util.tree_leaves_with_path(g)
                   if f"['{kind}']" in jax.tree_util.keystr(path))

    assert total("classifier") > 0.0
    assert total("dict") == 0.0
    assert total("decoder") == 0.0


def test_composes_with_topk_selection(build, X):
    # no log, so exact-zero tags cost nothing -- the KL form raises here
    with pytest.raises(ValueError):
        build(select="top2", pwak_loss=True)
    model, params = build(select="top2", l2pwak_loss=True)
    v = float(l2p_sum(model)(params, X))
    assert np.isfinite(v) and v > 0.0


def test_constant_coordinate_excluded(build):
    # the constant coordinate would distort the affinity's cosine and pad
    # the target with a trivially predicted dimension
    model, params = build(resid_const=True, l2pwak_loss=True)

    def probe(module, U):
        return module.dictencs[1].pwak_in(U)

    U = jnp.ones((B, KW["d_out"] + 1))
    assert model.apply(params, U, method=probe).shape == (B, KW["d_out"])


def test_training_steps_finite(X):
    d = KW["d_in"]
    hyper = Hyperparams(
        d, d, B, 1, 5e-5, 0.0, 1.0,
        "batchnorm", "normal", "featvar", 0.0, 0.02, 0.1,
        s_Hm=1e-6, s_L2pwak=1e-3, p_drop=0.1, ghost=False,
        pwak_s=2, pwak_tau=1.0)
    model = hyper.ontologizer(
        d, d, KW["e_dec"], KW["k"], KW["h"], KW["l"],
        n=2, gate="none", forward="resid", deepsup=True,
        dtype_str="float32", dtype_p_str="float32")
    assert model.l2pwak_loss is True
    assert model.pwak_loss is False

    state = hyper.init(model, save_each=10)
    rng = jax.random.PRNGKey(2)
    for step in range(10):
        rng, r = jax.random.split(rng)
        state, L, _ = update(state, hyper.loss, r, X, X,
                             temperature=T, p_drop=0.1, sd_K=0.02,
                             sd_in=0.0, sd_F=0.1, grad_clip=1.0,
                             pwak_s=2, pwak_tau=1.0)
        assert jnp.isfinite(L), f"step {step}"
    rows = np.asarray(state.stats[:10])
    assert np.all(np.isfinite(rows))
    assert np.all(rows[:, 12] > 0.0)   # L2_pwak column, live every step

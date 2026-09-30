# Batch mean-entropy bonus (s_Hm / hmean_loss / KL_m stat): the stat is
# KL(E_batch[p] || uniform) in bits, mean over heads -- entropy of the mean
# classification, not the mean of the entropies, so it taxes usage
# imbalance without softening per-sample classifications.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.ontologizer import Ontologizer
from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import update

from conftest import KW, B

T = 0.5


@pytest.fixture(scope="module")
def model_params(build):
    return build(hmean_loss=True)


def classifications(model, params, X):
    """Per-layer soft classifications along the same path withStats takes
    (rng=None: no winner dropout, no noise)."""
    def probe(module, X):
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        Ein = E
        Ps = []
        for i, dictenc in enumerate(module.dictencs):
            # gain-shape split mirrors withStats (identity when
            # gainshape is off), so the probe tracks either default
            U, G = dictenc.gainshape_in(Ein)
            P = dictenc.dict.cluster(dictenc.classifier(U), T)
            Ps.append(P)
            Y = dictenc.dict.combine(dictenc.dict.hfwd(P))
            R = R + dictenc.gained(Y, G)
            if i < module.l - 1:
                Ein = module.nextinput(X, R, None)
        return Ps
    return model.apply(params, X, method=probe)


def klm_sum(model):
    def f(params, X):
        _, stats, _ = model.apply(params, X, temperature=T,
                                  method=Ontologizer.withStats)
        return stats[:, 7].sum()
    return f


def test_stat_matches_manual_kl(model_params, X):
    model, params = model_params
    _, stats, _ = model.apply(params, X, temperature=T,
                              method=Ontologizer.withStats)
    assert stats.shape == (KW["l"], 12)
    for i, P in enumerate(classifications(model, params, X)):
        Pm = np.asarray(P).reshape(-1, KW["h"], KW["k"]).mean(0)
        Pm = Pm / Pm.sum(-1, keepdims=True)
        kl = (np.log2(KW["k"]) + (Pm * np.log2(Pm + 1e-12)).sum(-1)).mean()
        assert abs(float(stats[i, 7]) - kl) < 1e-4, f"layer {i}"


def test_gradient_gated_off(build, X):
    model_off, params = build(hmean_loss=False)
    g = jax.grad(klm_sum(model_off))(params, X)
    total = sum(float(jnp.abs(v).sum())
                for v in jax.tree_util.tree_leaves(g))
    assert total == 0.0


def test_gradient_reaches_classifier_when_on(model_params, X):
    model, params = model_params
    g = jax.grad(klm_sum(model))(params, X)
    cl = sum(float(jnp.abs(v).sum())
             for path, v in jax.tree_util.tree_leaves_with_path(g)
             if "classifier" in jax.tree_util.keystr(path))
    assert cl > 0.0


def test_descent_moves_mean_toward_uniform(model_params, X):
    model, params = model_params
    f = klm_sum(model)
    kl0 = float(f(params, X))
    g = jax.grad(f)(params, X)
    p1 = jax.tree_util.tree_map(lambda w, gw: w - 0.1 * gw, params, g)
    assert float(f(p1, X)) < kl0


def test_update_step_row_layout(X):
    """Full Hyperparams/OntoState step: 12-entry stats row
    [loss, MSE, L2_g, L1_K, L1_F, H, cossim_b, cossim_h, KL_m, KL_pwak,
    L2_pwak], with the loss reconstructing from its weighted terms."""
    s_Hm, s_bcossim = 1e-3, 1e-5
    d = KW["d_in"]
    hyper = Hyperparams(d, d, B, s_Hm=s_Hm, s_bcossim=s_bcossim, ghost=False,
                        noise_K="normal", noise_F="featvar")
    model = hyper.ontologizer(
        d, d, KW["e_dec"], KW["k"], KW["h"], KW["l"], n=2, gate="none",
        forward="resid", deepsup=True,
        dtype_str="float32", dtype_p_str="float32")
    assert model.hmean_loss is True
    assert model.bcossim_loss is True
    assert model.entropy_loss is False

    state = hyper.init(model, save_each=10)
    assert state.stats.shape == (10, 18)
    state, L, _ = update(state, hyper.loss, jax.random.PRNGKey(2), X, X,
                         temperature=T, sd_in=0.0, sd_K=0.01, sd_F=0.01,
                         p_drop=0.1, grad_clip=1.0)
    row = np.asarray(state.stats[0])
    assert np.all(np.isfinite(row)) and np.isfinite(float(L))
    assert row[10] > 0.0  # KL_m
    assert row[11] == 0.0  # KL_pwak with pwak_loss off
    assert row[12] == 0.0  # L2_pwak with l2pwak_loss off
    assert row[2] == 0.0  # L2_g with ghost=False
    assert row[13] == 0.0  # applied s_L1F, uncontrolled and unset here
    assert row[14] == 0.0  # applied s_kcossim, likewise
    assert row[15] >= 0.0  # per-head max cossim_k
    assert abs(row[0] - float(L)) < 1e-6
    assert abs(float(L) - (row[1] + s_Hm * row[10] + s_bcossim * row[6])) < 1e-5

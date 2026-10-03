# experiments/ste-arm/hsic_ste.py: the head-independence penalty rides
# the ghost slot. Its probe must reproduce Ontologizer.withStats exactly
# (the older hsic-bottleneck probe drifted from it), its loss must leave
# the parent's row intact, and the penalty must move the classifier.
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "ste-arm"))
from hsic_ste import HSICSteHyperparams, probe_with_codes   # noqa: E402

from ontologize.ontologizer import Ontologizer
from ontologize.training.ontostate import update

from conftest import KW, B

d = KW["d_in"]
STEP = dict(temperature=0.5, p_drop=0.1, sd_K=0.02, sd_in=0.0, sd_F=0.1,
            grad_clip=1.0)


def make(s_hsic, estimator="unbiased"):
    hyper = HSICSteHyperparams(
        d, d, B, noise_K="normal", noise_F="featvar", s_Hm=1e-4,
        ghost=False, s_hsic_heads=s_hsic, hsic_h=KW["h"],
        hsic_estimator=estimator)
    model = hyper.ontologizer(
        d, d, KW["e_dec"], KW["k"], KW["h"], KW["l"], n=2, gate="none",
        select="ste", forward="resid", deepsup=True, resid_const=True,
        dtype_str="float32", dtype_p_str="float32")
    return hyper, model, hyper.init(model, save_each=10)


def test_probe_matches_withstats(X):
    """Same rng, same kwargs: identical decodes and stats, plus the
    codes."""
    _, model, state = make(1e-3)
    kw = dict(temperature=0.5, sd_K=0.02, sd_F=0.1, p_drop=0.1)
    rng = jax.random.PRNGKey(7)
    Y0, S0, _ = model.apply(state.params, X, 0.0, rng,
                            method=Ontologizer.withStats, **kw)
    Y1, P, S1, _ = model.apply(state.params, X, 0.0, rng,
                               method=probe_with_codes, **kw)
    assert jnp.array_equal(Y0, Y1) and jnp.array_equal(S0, S1)
    assert P.shape == (KW["l"], B, KW["h"] * KW["k"])
    # ste codes are one-hot per head, noise and dropout included
    assert jnp.allclose(P.reshape(KW["l"], B, KW["h"], KW["k"]).sum(-1), 1.0)


def test_row_layout_and_penalty_column(X):
    hyper, _, state = make(1e-3)
    state, L, _ = update(state, hyper.loss, jax.random.PRNGKey(2), X, X,
                         **STEP)
    row = state.stats[0]
    assert row.shape == (18,) and jnp.all(jnp.isfinite(row))
    # raw CKA rides the ghost column. The unbiased estimator is signed --
    # near-independent heads sit around zero from either side -- so the
    # test is that the column is populated and finite, not positive.
    assert row[2] != 0.0 and jnp.isfinite(row[2])
    assert row[10] > 0.0                      # inherited KL_m still there
    assert abs(float(row[0]) - float(L)) < 1e-6
    # with the penalty off the ghost column is the parent's zero
    hyper0, _, state0 = make(0.0)
    state0, _, _ = update(state0, hyper0.loss, jax.random.PRNGKey(2), X, X,
                          **STEP)
    assert state0.stats[0][2] == 0.0


def test_off_equals_parent_loss(X):
    """s_hsic_heads=0 must be the stock objective to the bit."""
    hyper0, model, state = make(0.0)
    from ontologize.training.config import Hyperparams
    parent = Hyperparams(d, d, B, noise_K="normal", noise_F="featvar",
                         s_Hm=1e-4, ghost=False)
    rng = jax.random.PRNGKey(3)
    Y, P, S, _ = model.apply(state.params, X, 0.0, rng,
                             method=probe_with_codes, temperature=0.5)
    L0, r0 = hyper0.loss(Y, X, P, S)
    L1, r1 = parent.loss(Y, X, None, S)
    assert jnp.array_equal(r0, r1) and L0 == L1


def test_constant_head_is_zero_with_finite_gradient():
    """A head constant over the batch has an all-zero centered gram. Its
    CKA with every other head is 0 by definition, and the gradient must
    be finite there rather than sqrt(0)'s infinity times a zero."""
    import ontologize.fns.hsic as hsic
    P = jax.nn.one_hot(
        jax.random.randint(jax.random.PRNGKey(0), (32, 4), 0, 8), 8)
    Pc = P.at[:, 1, :].set(jax.nn.one_hot(3, 8))
    f = lambda P: hsic.pairwise_head_cka(P, 1.0)
    v, g = jax.value_and_grad(f)(Pc)
    assert jnp.isfinite(v) and jnp.all(jnp.isfinite(g))
    # the constant head contributes 0 to its 3 of the 6 pairs, so the
    # mean is the 3-head mean scaled by 3/6
    v3 = f(jnp.concatenate([Pc[:, :1], Pc[:, 2:]], axis=1))
    assert jnp.allclose(v, v3 * 3 / 6, atol=1e-6)


def test_estimator_choice_reaches_the_penalty(X):
    """A field that is stored but never passed through would make the
    two estimators score identically. They must not: the biased one
    carries a floor that independent one-hot heads produce on their
    own, and at a 32-row batch that floor is most of its value."""
    vals = {}
    for est in ("biased", "unbiased"):
        hyper, model, state = make(1e-3, est)
        Y, P, S, _ = model.apply(state.params, X, 0.0,
                                 jax.random.PRNGKey(11),
                                 method=probe_with_codes, temperature=0.5)
        vals[est] = float(hyper.loss(Y, X, P, S)[1][2])
    assert vals["biased"] > vals["unbiased"]


def test_penalty_reaches_the_classifier(X):
    hyper, model, state = make(1e-3)
    rng = jax.random.PRNGKey(4)

    def pen(params):
        _, P, _, _ = model.apply(params, X, 0.0, rng,
                                 method=probe_with_codes, temperature=0.5)
        l, b, hk = P.shape
        import ontologize.fns.hsic as hsic
        return sum(hsic.pairwise_head_cka(
            P[i].reshape(b, KW["h"], KW["k"]), 1.0) for i in range(l))

    g = jax.grad(pen)(state.params)
    cl = sum(float(jnp.abs(v).sum())
             for path, v in jax.tree_util.tree_leaves_with_path(g)
             if "classifier" in jax.tree_util.keystr(path))
    assert cl > 0.0

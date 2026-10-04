# `DictBlock.support_overlap`: the mean cosine between heads' usage-weighted
# coordinate profiles in the dictionary space. Replaces `cossim_h` and
# `cossim_flat` as the measure of disjoint support between heads; both
# keep their stats-row slots, logged as NaN.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.layers.dictblock import DictBlock, ConcatDictBlock
from ontologize.ontologizer import Ontologizer
from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import update

from conftest import (KW, B, DB_KW, N_STATS, RETIRED_LOSS, LAYER_WIDTH,
                      finite_live)

SUPPORT_LAYER = 12     # per-layer row
SUPPORT_COL = 18       # loss.csv


def with_weights(W, cls=DictBlock, **kw):
    db = cls(**DB_KW, **kw)
    K = jax.random.normal(jax.random.PRNGKey(3), (64, DB_KW["h"], DB_KW["k"]))
    params = db.init(jax.random.PRNGKey(4), K)
    params = {"params": {**params["params"], "weights": jnp.asarray(W)}}
    P = db.apply(params, K, 0.7, method=DictBlock.cluster)
    return db, params, P


def explicit(P, W):
    """Pairwise cosines of usage-weighted |W| profiles, looped."""
    P, W = np.asarray(P).reshape(-1, *P.shape[-2:]).mean(0), np.abs(np.asarray(W))
    S = np.einsum("hk,hkd->hd", P, W)
    h = S.shape[0]
    cs = [S[i] @ S[j] / (np.linalg.norm(S[i]) * np.linalg.norm(S[j]))
          for i in range(h) for j in range(h) if i != j]
    return float(np.mean(cs))


def test_matches_the_explicit_profile_cosines():
    W = jax.random.normal(jax.random.PRNGKey(0), (DB_KW["h"], DB_KW["k"], DB_KW["d"]))
    db, params, P = with_weights(W)
    got = float(db.apply(params, P, method=DictBlock.support_overlap))
    assert got == pytest.approx(explicit(P, W), rel=1e-5)


@pytest.mark.parametrize("signed", [False, True])
def test_zero_exactly_when_supports_are_disjoint(signed):
    h, k, d = DB_KW["h"], DB_KW["k"], DB_KW["d"]
    W = jax.random.normal(jax.random.PRNGKey(0), (h, k, d))
    mask = jnp.zeros((h, 1, d)).at[jnp.arange(h), 0,
                                  jnp.arange(d).reshape(h, -1).T].set(1.0)
    db, params, P = with_weights(W * mask, signed=signed)
    assert float(db.apply(params, P, method=DictBlock.support_overlap)) == 0.0
    # one shared coordinate makes it positive
    W2 = (W * mask).at[1, :, 0].set(1.0)
    db, params, P = with_weights(W2, signed=signed)
    assert float(db.apply(params, P, method=DictBlock.support_overlap)) > 0.0


def test_signed_cancellation_does_not_read_as_disjoint():
    """The case `cossim_h` gets wrong: two heads on the same coordinates
    whose signed outputs are orthogonal. Their head-output cosine is 0,
    but they share every coordinate, so support overlap is 1."""
    h, k, d = DB_KW["h"], DB_KW["k"], DB_KW["d"]
    a = jnp.zeros(d).at[0].set(1.0).at[1].set(1.0)
    b = jnp.zeros(d).at[0].set(1.0).at[1].set(-1.0)        # a . b = 0
    W = jnp.zeros((h, k, d)).at[0].set(a).at[1].set(b)
    W = W.at[2:].set(jnp.zeros(d).at[5].set(1.0))
    db, params, P = with_weights(W, signed=True)
    Fs = db.apply(params, P, method=DictBlock.hfwd)
    assert abs(float(jnp.dot(Fs[0, 0], Fs[0, 1]))) < 1e-6  # orthogonal outputs
    S = np.abs(np.asarray(W)).mean(1)
    assert S[0] @ S[1] / (np.linalg.norm(S[0]) * np.linalg.norm(S[1])) == pytest.approx(1.0)
    assert float(db.apply(params, P, method=DictBlock.support_overlap)) > 0.0


def test_unused_entries_do_not_count():
    """Usage-weighted: an entry no sample selects cannot add overlap."""
    h, k, d = DB_KW["h"], DB_KW["k"], DB_KW["d"]
    W = jnp.zeros((h, k, d))
    for i in range(h):
        W = W.at[i, :, i].set(1.0)                          # disjoint
    W = W.at[0, -1, 1].set(5.0)                              # entry k-1 of head 0 invades head 1
    db = DictBlock(**DB_KW, select="ste")
    params = {"params": {"weights": W}}
    P = jax.nn.one_hot(jnp.zeros((64, h), int), k)           # entry 0 always wins
    assert float(db.apply(params, P, method=DictBlock.support_overlap)) == 0.0


def test_concat_is_zero_by_construction():
    W = jax.random.normal(jax.random.PRNGKey(0),
                          (DB_KW["h"], DB_KW["k"], DB_KW["d"] // DB_KW["h"]))
    db, params, P = with_weights(W, cls=ConcatDictBlock)
    assert float(db.apply(params, P, method=ConcatDictBlock.support_overlap)) == 0.0


def test_gradient_gate(build, X):
    """Logged either way; differentiable only when weighted."""
    def summed(model):
        return lambda p: model.apply(
            p, X, temperature=0.5,
            method=Ontologizer.withStats)[1][:, SUPPORT_LAYER].sum()

    def total(g):
        return sum(float(jnp.abs(v).sum()) for v in jax.tree_util.tree_leaves(g))

    off, p_off = build(support_loss=False)
    assert total(jax.grad(summed(off))(p_off)) == 0.0
    on, p_on = build(support_loss=True)
    g = jax.grad(summed(on))(p_on)
    assert total(g) > 0.0
    # usage weighting gives the classifier a path, not just the dictionary
    assert float(jnp.abs(g["params"]["dictencs_0"]["classifier"]["weight"]).sum()) > 0.0


def build_hyper(**hkw):
    d = KW["d_in"]
    hyper = Hyperparams(d, d, B, ghost=False, noise_K="normal",
                        noise_F="featvar", **hkw)
    model = hyper.ontologizer(
        d, d, KW["e_dec"], KW["k"], KW["h"], KW["l"], n=2, gate="none",
        select="ste", forward="resid", deepsup=True, resid_const=True,
        dtype_str="float32", dtype_p_str="float32")
    return hyper, model


def test_lands_in_the_last_column_and_the_weight_multiplies_it(X):
    hyper, model = build_hyper(s_support=1e-3)
    assert model.support_loss is True
    state = hyper.init(model, save_each=10)
    assert state.stats.shape == (10, N_STATS)
    Y, stats, _ = model.apply(state.params, X, 0.5, method=Ontologizer.withStats)
    assert stats.shape[1] == LAYER_WIDTH
    L1, row1 = hyper.loss(Y, X, None, stats)
    L0, row0 = build_hyper()[0].loss(Y, X, None, stats)
    assert float(row1[SUPPORT_COL]) == pytest.approx(
        float(np.asarray(stats)[:, SUPPORT_LAYER].sum()), rel=1e-5)
    assert float(L1 - L0) == pytest.approx(1e-3 * float(row1[SUPPORT_COL]), rel=1e-4)


def test_retired_columns_are_nan_and_the_loss_is_finite(X):
    hyper, model = build_hyper(s_support=1e-3, s_bcossim=1e-5)
    state = hyper.init(model, save_each=10)
    state, L, _ = update(state, hyper.loss, jax.random.PRNGKey(2), X, X,
                         temperature=0.5, sd_in=0.0, sd_K=0.01, sd_F=0.01,
                         p_drop=0.1, grad_clip=1.0)
    assert jnp.isfinite(L)
    assert finite_live(state.stats[0], RETIRED_LOSS)
    g = jax.grad(lambda p: hyper.loss(
        *[model.apply(p, X, 0.5, method=Ontologizer.withStats)[i] for i in (0,)],
        X, None, model.apply(p, X, 0.5, method=Ontologizer.withStats)[1])[0])(state.params)
    assert all(bool(jnp.isfinite(v).all()) for v in jax.tree_util.tree_leaves(g))


@pytest.mark.parametrize("field", ["s_hcossim", "s_flatcos"])
def test_retired_weights_are_refused(field):
    with pytest.raises(ValueError, match="s_support"):
        Hyperparams(KW["d_in"], KW["d_in"], B, **{field: 1e-6})

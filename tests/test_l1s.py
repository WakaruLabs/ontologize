# `L1_S` / `s_L1S`: an L1 on the router alone.
#
# `L1_F` cannot stand in for it. `DictBlock.withL1` reduces `hfwd(P, S)`,
# so `L1_F` prices the product `S * |W| * c` and either factor can pay it
# down -- shrink the dictionary and the router is untouched, and the other
# way about. Pricing the router alone leaves the dictionary's magnitude to
# reconstruction, which is the separation a gated architecture rests on.
#
# The statistics row is append-only and pairs against
# `Hyperparams.s_loss`'s weight vector POSITIONALLY, so the alignment is
# what these tests are mostly for: a stat inserted anywhere but the end
# makes every weight past it multiply the wrong quantity, silently.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.ontologizer import Ontologizer
from ontologize.training.config import Hyperparams

from conftest import KW, B, LAYER_WIDTH, N_STATS

L1_S_COL = 17          # csv column; 0..16 are the historical layout
L1_S_LAYER = 11        # its position in the per-layer row


def build(s_L1S=0.0, scaled=True, **over):
    hp = Hyperparams(KW["d_in"], KW["d_out"], B, s_L1S=s_L1S)
    model = hp.ontologizer(**{**KW, "scaled": scaled, "select": "ste", **over})
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((2, KW["d_in"])))
    return hp, model, params


def run(hp, model, params, X):
    out = model.apply(params, X, 1.0, rng=jax.random.PRNGKey(2),
                      method=Ontologizer.withStats)
    Y, stats = out[0], out[1]
    Y = Y[-1] if Y.ndim == 3 else Y
    return hp.loss(Y, X, None, jnp.asarray(stats)), np.asarray(stats)


def test_weight_vector_and_isloss_keep_the_position_it_was_appended_at():
    # appended after cossim_flat's slot when it was added; later stats
    # (support) go after it, never before
    hp = Hyperparams(KW["d_in"], KW["d_out"], B, s_L1S=1e-3)
    s, isloss = hp.s_loss()
    assert float(s[L1_S_LAYER + 1]) == pytest.approx(1e-3)
    assert isloss[L1_S_LAYER] is True              # isloss has no kmax entry
    assert sum(isloss) == 1


def test_the_stat_lands_in_the_new_last_column(X):
    hp, model, params = build(s_L1S=0.0)
    (_, row), stats = run(hp, model, params, X)
    assert stats.shape[1] == LAYER_WIDTH
    assert np.asarray(row).shape == (N_STATS,)
    # the column is the layer-summed statistic
    assert float(row[L1_S_COL]) == pytest.approx(
        stats[:, L1_S_LAYER].sum(), rel=1e-5)
    assert float(row[L1_S_COL]) > 0.0


def test_the_weight_multiplies_exactly_that_column(X):
    """The positional-alignment check. If `s_L1S` were paired against any
    other column this difference would not come out as the product."""
    hp0, model, params = build(s_L1S=0.0)
    (L0, row0), _ = run(hp0, model, params, X)
    hp1, model1, params1 = build(s_L1S=1e-3)
    (L1, row1), _ = run(hp1, model1, params1, X)
    assert float(row1[L1_S_COL]) == pytest.approx(float(row0[L1_S_COL]), rel=1e-5)
    assert float(L1) - float(L0) == pytest.approx(
        1e-3 * float(row0[L1_S_COL]), rel=1e-4)


def test_sparse_S_is_set_from_the_weight_and_gates_the_gradient(X):
    """`Sparse.l1` stop-gradients unless `sparse` is set, so the stat is
    inert until weighted -- the same wiring `sparse_K`/`sparse_F` use."""
    def routed(mod, _):
        return mod.dictencs[0].sparse_S

    grads = {}
    for s_l1s in (0.0, 1e-3):
        hp, model, params = build(s_L1S=s_l1s)
        assert bool(model.apply(params, X, method=routed)) is (s_l1s != 0.0)
        g = jax.grad(lambda p: float_loss(hp, model, p, X))(params)
        grads[s_l1s] = float(
            jnp.abs(g["params"]["dictencs_0"]["router"]["weight"]).max())
    # reconstruction already reaches the router; the penalty adds to it
    assert grads[1e-3] > grads[0.0]


def float_loss(hp, model, params, X):
    out = model.apply(params, X, 1.0, rng=jax.random.PRNGKey(2),
                      method=Ontologizer.withStats)
    Y = out[0][-1] if out[0].ndim == 3 else out[0]
    return hp.loss(Y, X, None, jnp.asarray(out[1]))[0]


def test_zero_without_a_router(X):
    """`scaled` off means there is no router to read, and the column has to
    be a clean zero rather than absent -- the row width is fixed."""
    hp, model, params = build(s_L1S=1e-3, scaled=False)
    (L, row), stats = run(hp, model, params, X)
    assert np.asarray(row).shape == (N_STATS,)
    assert float(row[L1_S_COL]) == 0.0
    assert "router" not in params["params"]["dictencs_0"]


def test_l1f_prices_the_product_and_l1s_does_not(X):
    """The motivation, stated as a measurement. Scaling the router changes
    `L1_F`, because it reduces `hfwd(P, S)`; so `L1_F` cannot isolate the
    dictionary's magnitude and `L1_S` is not redundant with it."""
    hp, model, params = build(s_L1S=0.0)
    (_, row0), _ = run(hp, model, params, X)
    w = params["params"]["dictencs_0"]["router"]["weight"]
    louder = jax.tree_util.tree_map(lambda v: v, params)
    louder["params"]["dictencs_0"]["router"]["weight"] = w * 3.0
    (_, row1), _ = run(hp, model, louder, X)
    L1F, L1S = 4, L1_S_COL
    assert float(row1[L1S]) > 1.5 * float(row0[L1S])       # router L1 moved
    assert float(row1[L1F]) != pytest.approx(float(row0[L1F]), rel=1e-3)


def test_the_training_and_inference_paths_use_the_same_router_gain(X):
    """`DictEnc.scale` (the withClusts/withGhost path) returns
    `abs(router(E))`; `withStats` has to match it. It did not, and at init
    most of the router is negative, so the trained forward differed from
    the evaluated one -- and a negative gain subtracts a head's
    contribution, which the non-negative dictionary exists to rule out."""
    _, model, params = build(s_L1S=0.0)

    def probe(mod, x):
        de = mod.dictencs[0]
        E, _ = mod.encode(x, 0.0, None)
        U, _ = de.gainshape_in(mod.constinput(E))
        return de.router.withL1(U)[0], de.scale(U)

    raw, scaled = (np.asarray(v) for v in model.apply(params, X, method=probe))
    assert (raw < 0).any(), "fixture has no negative router output to test"
    assert np.allclose(np.abs(raw), scaled, atol=1e-6)

    # and the statistic the penalty prices stays non-negative in the row
    (_, row), stats = run(*build(s_L1S=0.0), X)
    assert float(row[L1_S_COL]) >= 0.0


def test_a_sigmoid_gate_router_keeps_every_head_reachable(X):
    """sigmoid > 0 strictly, so `L1_S` can close a head arbitrarily far
    without shutting it: unlike a relu gate, the router keeps receiving
    gradient and the head can come back. That is why this is the
    configuration `s_L1S` is usable in."""
    hp = Hyperparams(KW["d_in"], KW["d_out"], B, s_L1S=1e-2)
    model = hp.ontologizer(**{**KW, "scaled": True, "select": "ste",
                              "gate_router": "sigmoid"})
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((2, KW["d_in"])))

    def gate(mod, x):
        de = mod.dictencs[0]
        E, _ = mod.encode(x, 0.0, None)
        U, _ = de.gainshape_in(mod.constinput(E))
        return de.router.fn_gate(de.router.nfwd(U)[0])

    g = np.asarray(model.apply(params, X, method=gate))
    assert (g > 0.0).all() and (g < 1.0).all()
    # the penalty reaches the router, so it can act on that gate
    grad = jax.grad(lambda p: float_loss(hp, model, p, X))(params)
    assert float(jnp.abs(
        grad["params"]["dictencs_0"]["router"]["weight"]).max()) > 0.0

# PWAK consensus KL (s_pwak / pwak_loss / KL_pwak stat): each head's
# classification is pulled toward a stop-gradient consensus of its
# partition-gated neighbours' classifications, `diffuse_kl`. Also covers the
# kernel pieces it shares with L2_pwak (test_pwak_l2.py): the row-stochastic
# `wak` and the self-edge-free `affinity`.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.fns.pwak import affinity, diffuse_kl, pwak_kl, wak
from ontologize.ontologizer import Ontologizer
from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import update
from ontologize.visualize.loss import COLUMNS

from conftest import RETIRED_LOSS, finite_live

B, H, K, D = 24, 3, 4, 8
T = 0.5
KL_COL = COLUMNS.index("KL_pwak")   # loss.csv column
KL_ROW = KL_COL - 3                 # per-layer row: loss.csv columns 3-12 are
                                    # its first ten entries, summed over layers


def _P(seed=1):
    return jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(seed), (B, H, K)), -1)


def _model(s_pwak):
    return Hyperparams(D, D, B, s_pwak=s_pwak).ontologizer(
        D, D, 16, K, H, 2, n=2, gate="none", forward="resid", deepsup=True,
        resid_norm=True, resid_const=True,
        dtype_str="float32", dtype_p_str="float32")


def test_wak_rows_are_stochastic():
    G = wak(jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (B, B))))
    assert jnp.allclose(G.sum(-1), 1.0, atol=1e-5)
    assert jnp.allclose(wak(jnp.zeros((3, 3))), 0.0)  # empty rows, not NaN


def test_affinity_drops_the_self_edge():
    # noise2self: no estimate may use the value of the point being estimated
    A = affinity(jax.random.normal(jax.random.PRNGKey(0), (B, D)), 0.2)
    assert jnp.allclose(jnp.diag(A), 0.0)
    assert jnp.allclose(A, A.T, atol=1e-5)
    assert jnp.all(A >= 0)


def test_diffusion_preserves_the_simplex():
    E = jax.random.normal(jax.random.PRNGKey(0), (B, D))
    for s in (0, 1, 3):
        Q = diffuse_kl(_P(), E, s)
        assert jnp.allclose(Q.sum(-1), 1.0, atol=1e-4), s
        assert jnp.all(Q >= 0), s


def test_s0_is_exactly_inert():
    # the term must cost nothing before the schedule ramps
    E = jax.random.normal(jax.random.PRNGKey(0), (B, D))
    assert jnp.allclose(diffuse_kl(_P(), E, 0), _P())
    assert float(pwak_kl(_P(), E, 0)) == pytest.approx(0.0, abs=1e-6)


def test_confidence_flows_from_neighbours_to_the_uncertain():
    # the mechanism: a confident neighbourhood hardens an uncertain sample,
    # while the confident samples themselves barely move
    E = jnp.ones((B, D)) + 0.01 * jax.random.normal(jax.random.PRNGKey(3), (B, D))
    P = jnp.tile(jnp.array([0.85, 0.05, 0.05, 0.05]), (B, 1))
    P = P.at[0].set(jnp.full(K, 1.0 / K))[:, None, :]
    Q = diffuse_kl(P, E, 1)
    assert Q[0, 0, 0] > 0.6                          # pulled to the consensus
    assert abs(float(Q[5, 0, 0]) - 0.85) < 0.05      # stays put


def test_target_is_stop_gradiented_but_p_still_learns():
    E = jax.random.normal(jax.random.PRNGKey(0), (B, D))
    L = jax.random.normal(jax.random.PRNGKey(1), (B, H, K))
    g = jax.grad(lambda L: pwak_kl(jax.nn.softmax(L, -1), E, 3, 0.2, True))(L)
    assert jnp.all(jnp.isfinite(g)) and float(jnp.abs(g).sum()) > 0
    # with isloss=False the whole term is detached
    g0 = jax.grad(lambda L: pwak_kl(jax.nn.softmax(L, -1), E, 3, 0.2, False))(L)
    assert float(jnp.abs(g0).sum()) == 0.0


def test_traced_depth_does_not_break_jit():
    E = jax.random.normal(jax.random.PRNGKey(0), (B, D))
    f = jax.jit(lambda P, E, s: pwak_kl(P, E, s))
    assert float(f(_P(), E, 0)) == pytest.approx(0.0, abs=1e-6)
    assert float(f(_P(), E, 2)) > 0.0


def test_s_pwak_gates_the_layer_and_weights_its_slot():
    # the weight vector pairs against [MSE_ghost] + the per-layer row by
    # position, so s_pwak has to land on KL_pwak's slot
    s, _ = Hyperparams(D, D, B, s_pwak=1e-3).s_loss()
    assert float(s[1 + KL_ROW]) == pytest.approx(1e-3)
    assert _model(1e-3).pwak_loss is True
    assert _model(0.0).pwak_loss is False


@pytest.mark.parametrize("s_pwak", [0.0, 1e-3])
def test_training_steps_with_pwak(s_pwak):
    # ten steps with the term wired in: finite loss and live stats, KL_pwak
    # 0 at depth 0 and positive once diffusion starts (only when the
    # pwak_loss gate is on)
    hyper = Hyperparams(D, D, B, lr=1e-3, s_bcossim=1e-5, s_Hm=1e-6,
                        s_pwak=s_pwak, ghost=False)
    model = hyper.ontologizer(D, D, 16, K, H, 2, n=2, gate="none",
                              forward="resid", deepsup=True, resid_norm=True,
                              resid_const=True, dtype_str="float32",
                              dtype_p_str="float32")
    state = hyper.init(model, save_each=10)
    X = jax.random.normal(jax.random.PRNGKey(0), (B, D))
    rng = jax.random.PRNGKey(2)
    for step in range(10):
        rng, r = jax.random.split(rng)
        state, L, _ = update(state, hyper.loss, r, X, X, temperature=T,
                             p_drop=0.0, sd_K=0.0, sd_in=0.0, sd_F=0.0,
                             pwak_s=(step // 5) * 2, pwak_tau=0.2,
                             grad_clip=1.0)
        assert jnp.isfinite(L), step
    rows = np.asarray(state.stats[:10])
    assert rows.shape == (10, len(COLUMNS))
    assert finite_live(rows, RETIRED_LOSS)
    kl = rows[:, KL_COL]
    assert np.allclose(kl[:5], 0.0, atol=1e-6)    # depth 0: inert
    if s_pwak:
        assert kl[5:].max() > 0.0                 # depth 2: live
    else:
        assert np.allclose(kl, 0.0)               # gate off: no term


def test_graph_is_gated_on_pwak_loss():
    # pwak-off models must never build the b x b affinity graph. The
    # schedule passes `pwak_s` traced, so the diffusion lowers to a `while`
    # that XLA cannot prune; the only real gate is the static `pwak_loss`
    X = jax.random.normal(jax.random.PRNGKey(0), (B, D))

    def jaxpr(s_pwak):
        model = _model(s_pwak)
        params = model.init(jax.random.PRNGKey(1), X)
        f = lambda p, x, s: model.apply(p, x, temperature=T, pwak_s=s,
                                        method=Ontologizer.withStats)
        return str(jax.make_jaxpr(f)(params, X, 2))

    assert "while" in jaxpr(1e-3)      # on: traced-depth diffusion
    assert "while" not in jaxpr(0.0)   # off: no graph at all

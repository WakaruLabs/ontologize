# Sparsemax selection (select="sparsemax"): cluster projects the
# temperature-scaled logits onto the probability simplex (Martins &
# Astudillo 2016). The tests check the projection against its optimality
# conditions, its support against the temperature, its Jacobian against
# the closed form, a dropped winner, and that a model built and trained
# under it stays finite.
import jax
import jax.numpy as jnp
import numpy as np

from ontologize.fns.classify import sparsemax
from ontologize.fns.keys import get_activation
from ontologize.layers.dictblock import DictBlock
from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import update

from conftest import KW, B, DB_KW, RETIRED_LOSS, finite_live


def test_get_activation_parses_sparsemax():
    assert get_activation("sparsemax") is sparsemax


def test_is_the_projection_onto_the_simplex():
    x = np.asarray(jax.random.normal(jax.random.PRNGKey(0), (64, 8)) * 2.0,
                   np.float64)
    P = np.asarray(sparsemax(jnp.asarray(x)), np.float64)
    assert (P >= 0).all() and np.allclose(P.sum(-1), 1.0, atol=1e-5)
    # the optimality conditions: x - p is one threshold tau on the
    # support, and every logit off the support is at most tau
    for xi, pi in zip(x, P):
        s = pi > 0
        tau = (xi[s] - pi[s]).mean()
        assert np.allclose(xi[s] - pi[s], tau, atol=1e-5)
        assert (xi[~s] <= tau + 1e-5).all()


def test_a_point_on_the_simplex_is_its_own_projection():
    p = jnp.array([0.5, 0.3, 0.2, 0.0])
    assert jnp.allclose(sparsemax(p), p, atol=1e-6)


def test_temperature_sets_the_support():
    x = jax.random.normal(jax.random.PRNGKey(1), (256, 8))
    sizes = [float((sparsemax(x / T) > 0).sum(-1).mean())
             for T in (100.0, 1.0, 0.1, 1e-3)]
    # a lower temperature never adds an entry to a row's support
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] == 8.0
    P = sparsemax(x / 1e-3)
    assert jnp.all((P > 0).sum(-1) == 1)
    assert jnp.all(jnp.argmax(P, -1) == jnp.argmax(x, -1))


def test_jacobian_is_the_projection_on_the_support():
    x = jnp.array([1.0, 0.8, 0.1, -1.0, 0.75])
    J = np.asarray(jax.jacobian(sparsemax)(x))
    s = np.asarray(sparsemax(x) > 0, np.float64)
    assert s.sum() == 3
    assert np.allclose(J, np.diag(s) - np.outer(s, s) / s.sum(), atol=1e-6)
    # a row whose support is one entry passes no gradient
    J1 = jax.jacobian(sparsemax)(jnp.array([5.0, 0.0, -1.0]))
    assert jnp.all(J1 == 0)


def test_a_dropped_winner_is_left_out():
    # winner dropout masks the argmax to -inf before cluster
    x = jnp.array([2.0, 1.0, 0.5, -0.5])
    drop = jnp.arange(4) == 0
    P = sparsemax(jnp.where(drop, -jnp.inf, x))
    assert P[0] == 0 and jnp.isclose(P.sum(), 1.0)
    g = jax.grad(lambda x: (sparsemax(jnp.where(drop, -jnp.inf, x))
                            * jnp.arange(4.0)).sum())(x)
    assert jnp.all(jnp.isfinite(g)) and g[0] == 0


def test_dictblock_clusters_by_sparsemax_and_counts_it_dense():
    db = DictBlock(**dict(DB_KW, select="sparsemax"))
    K = jax.random.normal(jax.random.PRNGKey(3), (16, DB_KW["h"], DB_KW["k"]))
    params = db.init(jax.random.PRNGKey(4), K)
    P = db.apply(params, K, 0.5, method=DictBlock.cluster)
    assert jnp.allclose(P, sparsemax(K / 0.5), atol=1e-6)
    assert db.apply(params, method=lambda m: m.k_sel) == DB_KW["k"]


def test_training_steps_finite(X):
    # regularizer mix as in test_topk_select, with sparsemax selection
    d = KW["d_in"]
    hyper = Hyperparams(
        d, d, B, 1, 5e-5, 0.0, 1.0,
        "batchnorm", "normal", "featvar", 0.0, 0.02, 0.1,
        s_Hm=1e-6, p_drop=0.1, ghost=False)
    model = hyper.ontologizer(
        d, d, KW["e_dec"], KW["k"], KW["h"], KW["l"],
        n=2, gate="none", forward="resid", deepsup=True,
        select="sparsemax", dtype_str="float32", dtype_p_str="float32")
    state = hyper.init(model, save_each=10)
    rng = jax.random.PRNGKey(2)
    for step in range(10):
        rng, r = jax.random.split(rng)
        state, L, _ = update(state, hyper.loss, r, X, X,
                             temperature=0.5, p_drop=0.1, sd_K=0.02,
                             sd_in=0.0, sd_F=0.1, grad_clip=1.0)
        assert jnp.isfinite(L), f"step {step}"
    assert finite_live(state.stats[:10], RETIRED_LOSS)

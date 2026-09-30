# Top-k tag selection (select="top<k>"): cluster softmaxes the top-k
# logits per head and masks the rest to -inf -- exact-zero probability
# outside the support, no gradient to or from masked entries. The support
# is chosen on the logits cluster receives (after sd_K noise and winner
# dropout), so both mechanisms still explore membership.
import functools

import jax
import jax.numpy as jnp
import pytest

from ontologize.fns.classify import topk_cl
from ontologize.fns.keys import get_activation
from ontologize.layers.dictblock import DictBlock
from ontologize.ontologizer import Ontologizer
from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import update

from conftest import KW, B, DB_KW

M = 3


@pytest.fixture(scope="module")
def topk_model(build):
    return build(select=f"top{M}")


def test_get_activation_parses_topk_keys():
    fn = get_activation("top1")
    assert isinstance(fn, functools.partial)
    assert fn.func is topk_cl and fn.keywords == {"k": 1}


def test_support_and_simplex():
    x = jax.random.normal(jax.random.PRNGKey(0), (16, 4, 8))
    P = topk_cl(x, M)
    assert jnp.all((P > 0).sum(-1) == M)  # random logits: no ties
    assert jnp.allclose(P.sum(-1), 1.0, atol=1e-6)
    # the survivors are exactly the top-M logits
    top = jax.lax.top_k(x, M)[1]
    assert jnp.all(jnp.take_along_axis(P, top, -1) > 0)


def test_support_temperature_invariant():
    # cluster masks after dividing by temperature; positive scaling
    # preserves logit order, so T only sharpens within the survivor set
    x = jax.random.normal(jax.random.PRNGKey(1), (16, 8))
    assert jnp.array_equal(topk_cl(x / 1.0, M) > 0,
                           topk_cl(x / 0.03, M) > 0)


def test_masked_entries_carry_no_gradient():
    x = jax.random.normal(jax.random.PRNGKey(2), (8,))
    J = jax.jacobian(lambda x: topk_cl(x, M))(x)
    assert jnp.all(jnp.isfinite(J))
    mask = topk_cl(x, M) == 0
    assert jnp.all(J[mask] == 0)     # masked probabilities are constant 0
    assert jnp.all(J[:, mask] == 0)  # masked logits receive no gradient


def test_winner_dropout_promotes_next_tag():
    # drop_winners masks the argmax before cluster, so top-k promotes the
    # (k+1)-th logit into the support instead of shrinking it -- the
    # property that would be lost with selection at the classifier
    db = DictBlock(**dict(DB_KW, select=f"top{M}"))
    K = jax.random.normal(jax.random.PRNGKey(3), (16, DB_KW["h"], DB_KW["k"]))
    params = db.init(jax.random.PRNGKey(4), K)
    win = jnp.argmax(K, -1)
    drop = jax.nn.one_hot(win, DB_KW["k"], dtype=bool)
    P_drop = db.apply(params, jnp.where(drop, -jnp.inf, K),
                      method=DictBlock.cluster)
    assert jnp.all(jnp.take_along_axis(P_drop, win[..., None], -1) == 0)
    assert jnp.all((P_drop > 0).sum(-1) == M)


def test_bounds_and_pwak_guards(build):
    with pytest.raises(ValueError):
        build(select="top0")
    with pytest.raises(ValueError):
        build(select=f"top{KW['k'] + 1}")
    with pytest.raises(ValueError):
        build(select=f"top{M}", pwak_loss=True)


def test_realized_code_is_topk_sparse(topk_model, X):
    model, params = topk_model

    def probe(module, X):
        E, _ = module.encode(X, 0.0, None)
        _, Ps = module.classify(E, temperature=0.5, X_ref=X)
        return Ps

    Ps = model.apply(params, X, method=probe)
    P = Ps.reshape(KW["l"], B, KW["h"], KW["k"])
    assert jnp.all((P > 0).sum(-1) == M)


def test_withstats_finite(topk_model, X):
    model, params = topk_model
    Y, stats, _ = model.apply(params, X, temperature=0.5,
                              method=Ontologizer.withStats)
    assert jnp.all(jnp.isfinite(Y))
    assert stats.shape == (KW["l"], 12)
    assert jnp.all(jnp.isfinite(stats))


def test_training_steps_finite(X):
    # regularizer mix as in test_training_step, with top-k selection
    d = KW["d_in"]
    hyper = Hyperparams(
        d, d, B, 1, 5e-5, 0.0, 1.0,
        "batchnorm", "normal", "featvar", 0.0, 0.02, 0.1,
        s_Hm=1e-6, p_drop=0.1, ghost=False)
    model = hyper.ontologizer(
        d, d, KW["e_dec"], KW["k"], KW["h"], KW["l"],
        n=2, gate="none", forward="resid", deepsup=True,
        select=f"top{M}",
        dtype_str="float32", dtype_p_str="float32")
    state = hyper.init(model, save_each=10)

    rng = jax.random.PRNGKey(2)
    for step in range(10):
        rng, r = jax.random.split(rng)
        state, L, _ = update(state, hyper.loss, r, X, X,
                             temperature=0.5, p_drop=0.1, sd_K=0.02,
                             sd_in=0.0, sd_F=0.1, grad_clip=1.0)
        assert jnp.isfinite(L), f"step {step}"
    assert jnp.all(jnp.isfinite(state.stats[:10]))

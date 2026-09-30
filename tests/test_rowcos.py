"""`DictBlock.rowcos` / `s_kcossim`: within-head dictionary row collinearity.

The stat exists because `hmean_kl` measures usage balance, not head
liveness, and the two come apart: a head whose entries all decode to one
direction produces an output independent of which entry wins, so its
classifier can spread usage perfectly and score `KL_m = 0` while the head
carries no information. These tests pin that gap, the gradient gate, and
the two properties that let a first version of this stat be defeated:
scale (the normalization must not deflate as the rows shrink) and
aggregation (it returns per-head values, because a mean hides a collapsed
head among healthy ones).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.layers.dictblock import DictBlock
from ontologize.training.config import Hyperparams

H, K, D, B = 4, 8, 32, 16


def block(**kw):
    m = DictBlock(k=K, d=D, h=H, dtype_str="float32", **kw)
    p = m.init(jax.random.key(0), jnp.zeros((B, H, K)))
    return m, p


def with_weights(W, **kw):
    m = DictBlock(k=K, d=D, h=H, dtype_str="float32", **kw)
    return m, {"params": {"weights": W}}


def test_collapsed_head_scores_one():
    """All entries the same direction -- the pathology."""
    W = jnp.broadcast_to(jnp.ones((1, 1, D)), (H, K, D))
    m, p = with_weights(W)
    assert float(m.apply(p, method=m.rowcos).mean()) == pytest.approx(1.0, abs=1e-5)


def test_disjoint_supports_score_zero():
    """Entries on distinct coordinates: orthogonal, so nothing to penalize.
    Reachable here because the rows are non-negative and k <= d."""
    W = jnp.zeros((H, K, D)).at[:, jnp.arange(K), jnp.arange(K)].set(1.0)
    m, p = with_weights(W)
    assert float(m.apply(p, method=m.rowcos).mean()) == pytest.approx(0.0, abs=1e-5)


def test_bounded_and_non_negative():
    """`dicts()` is abs()'d, so every cosine lies in [0, 1] and so does the
    mean -- the stat cannot be gamed negative."""
    m, p = block()
    v = np.asarray(m.apply(p, method=m.rowcos))
    assert v.shape == (H,)
    assert (v >= 0.0).all() and (v <= 1.0).all()


def test_scale_invariant():
    """A property of directions, not magnitudes: rescaling rows must not
    move it (so it does not double as a norm penalty like s_L1F)."""
    W = jax.random.normal(jax.random.key(3), (H, K, D))
    m, p = with_weights(W)
    scales = jax.random.uniform(jax.random.key(4), (H, K, 1), minval=0.1,
                                maxval=10.0)
    m2, p2 = with_weights(W * scales)
    assert float(m.apply(p, method=m.rowcos).mean()) == pytest.approx(
        float(m2.apply(p2, method=m2.rowcos).mean()), rel=1e-4)


def test_independent_of_the_batch():
    """Reads the weights alone, so it is identical for any classification --
    which is what makes it cost one tagGram and no batch statistics."""
    m, p = block()
    v = float(m.apply(p, method=m.rowcos).mean())
    for seed in (1, 2, 3):
        assert float(m.apply(p, method=m.rowcos).mean()) == pytest.approx(v)


def test_gradient_gated_by_kcossim_loss():
    def total(mod, params):
        g = jax.grad(lambda q: mod.apply({"params": q},
                                         method=mod.rowcos).mean())(
            params["params"])
        return sum(float(jnp.abs(v).sum())
                   for v in jax.tree_util.tree_leaves(g))
    off, p_off = block(kcossim_loss=False)
    on, p_on = block(kcossim_loss=True)
    assert total(off, p_off) == 0.0
    assert total(on, p_on) > 0.0


def test_descent_separates_a_partly_collapsed_head():
    """The gradient has to actually undo the overlap, not merely be
    nonzero: a step downhill must lower the stat."""
    W = jnp.ones((H, K, D)) + 0.5 * jax.random.normal(
        jax.random.key(5), (H, K, D))
    m, p = with_weights(W, kcossim_loss=True)
    f = lambda q: m.apply({"params": q}, method=m.rowcos).mean()
    v0 = float(f(p["params"]))
    g = jax.grad(f)(p["params"])
    p1 = jax.tree_util.tree_map(lambda w, gw: w - 0.1 * gw, p["params"], g)
    assert float(f(p1)) < v0


def test_gradient_vanishes_as_a_head_approaches_collapse():
    """Cosine is maximal at collinearity, so its gradient goes to zero
    there: the penalty prevents a collapse far better than it repairs one.
    Escape from an already-collapsed head is asymptotically slow, which is
    why the term wants to be live from the start rather than switched on
    after a collapse is observed."""
    def gnorm(noise):
        W = jnp.ones((H, K, D)) + noise * jax.random.normal(
            jax.random.key(5), (H, K, D))
        m, p = with_weights(W, kcossim_loss=True)
        g = jax.grad(lambda q: m.apply({"params": q},
                                       method=m.rowcos).mean())(p["params"])
        return float(jnp.sqrt(sum(jnp.sum(x ** 2)
                                  for x in jax.tree_util.tree_leaves(g))))
    near, mid, far = gnorm(0.001), gnorm(0.01), gnorm(0.1)
    assert near < mid < far          # weaker the closer to collapse
    assert near < far / 50


def test_klm_cannot_see_what_rowcos_sees():
    """The gap the stat exists for: a fully collapsed dictionary under a
    uniform classification is perfect by `hmean_kl` and maximal by
    `rowcos`."""
    W = jnp.broadcast_to(jnp.ones((1, 1, D)), (H, K, D))
    m, p = with_weights(W)
    P = jnp.full((B, H, K), 1.0 / K)
    assert float(m.apply(p, P, method=m.hmean_kl)) == pytest.approx(0.0,
                                                                    abs=1e-5)
    assert float(m.apply(p, method=m.rowcos).mean()) == pytest.approx(1.0, abs=1e-5)


def test_weight_enables_the_gradient_gate():
    d = 6
    off = Hyperparams(d, d, B, s_kcossim=0.0, ghost=False)
    on = Hyperparams(d, d, B, s_kcossim=1e-4, ghost=False)
    mk = lambda h: h.ontologizer(d, d, 10, K, H, 2, n=2, forward="resid",
                                 dtype_str="float32", dtype_p_str="float32")
    assert mk(off).kcossim_loss is False
    assert mk(on).kcossim_loss is True

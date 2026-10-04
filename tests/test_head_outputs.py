# `DictEnc.head_outputs` is what analysis scripts use to replay a layer
# under a held or forced classification; it must write what the layer's own
# forward writes, router gain and fibers included.
import jax
import jax.numpy as jnp
import pytest

from conftest import KW, X  # noqa: F401  (X is a fixture)
from ontologize.layers.dictenc import DictEnc
from ontologize.ontologizer import Ontologizer


def replay(module, X, T):
    """Each layer's output as the forward computes it, and as
    `head_outputs` rebuilds it from the same classification."""
    E, _ = module.encode(X, 0.0, None)
    R = module.resid(E)
    Ein = module.constinput(E)
    pairs = []
    for i, de in enumerate(module.dictencs):
        U, G = de.gainshape_in(Ein)
        P = de.dict.cluster(de.classifier(U), T)
        R_fwd, _ = de.withClusts(R, Ein, temperature=T)
        R_rep = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
        pairs.append((R_fwd, R_rep))
        R = R_fwd
        if i < module.l - 1:
            Ein = module.nextinput(X, R, P.reshape(X.shape[0], -1))
    return pairs


@pytest.mark.parametrize("over", [
    dict(),
    dict(scaled=True),
    dict(scaled=True, gate_router="sigmoid"),
    dict(fiber_rank=2),
    dict(scaled=True, concat=True),
])
def test_head_outputs_replays_the_forward(over, X):
    model = Ontologizer(**{**KW, "select": "ste", "resid_gain": True, **over})
    params = model.init(jax.random.PRNGKey(0), X)
    for R_fwd, R_rep in model.apply(params, X, 0.5, method=replay):
        assert jnp.allclose(R_fwd, R_rep, rtol=1e-5, atol=1e-6)


def test_the_router_changes_what_a_layer_writes(X):
    # the bug `head_outputs` exists to prevent: `dict.hfwd(P)` alone drops
    # the router, which this pins as a real difference, not a no-op
    model = Ontologizer(**{**KW, "select": "ste", "scaled": True})
    params = model.init(jax.random.PRNGKey(0), X)

    def both(module, X):
        de = module.dictencs[0]
        U, _ = de.gainshape_in(module.constinput(module.encode(X, 0.0, None)[0]))
        P = de.dict.cluster(de.classifier(U), 0.5)
        return de.head_outputs(U, P), de.dict.hfwd(P)

    a, b = model.apply(params, X, method=both)
    assert not jnp.allclose(a, b)

# steerfid.py's steering machinery: direction normalization, feature
# selection, effect scoring, and the Ontologizer set-intervention delta
# (nonzero for a real intervention, zero direction never produced).
import jax.numpy as jnp
import numpy as np

import steerfid


def test_unit_normalizes_rows():
    v = np.random.default_rng(0).normal(size=(5, 16)).astype(np.float32)
    u = steerfid.unit(v)
    assert np.allclose(np.linalg.norm(u, axis=-1), 1.0, atol=1e-5)


def test_pick_features_live_and_deterministic():
    rate = np.zeros(100)
    rate[10:30] = 0.05
    a = steerfid.pick_features(rate, 8, seed=1)
    b = steerfid.pick_features(rate, 8, seed=1)
    assert np.array_equal(a, b)
    assert ((a >= 10) & (a < 30)).all()  # only live features
    assert len(np.unique(a)) == 8


def test_effect_scores():
    a_base = np.array([0.1, 0.1])
    a_steer = np.array([0.6, 0.05])
    eff, hit = steerfid.effect_scores(a_base, a_steer, q_hi=0.5, q_hit=0.4)
    assert np.allclose(eff, [1.0, -0.1])
    assert np.array_equal(hit, [1.0, 0.0])


def test_firing_quantiles_condition_on_firing():
    # a sparse column: unconditional q99 would be 0 and blow up the
    # normalization; conditioned on firing it reflects firing strength
    sub = np.zeros((1000, 2), np.float32)
    sub[:5, 0] = [1.0, 2.0, 3.0, 4.0, 5.0]  # fires 0.5% of the time
    sub[:, 1] = 0.0                          # never fires: fallback 1.0
    hi, hit = steerfid.firing_quantiles(sub, thr=0.0)
    assert 4.0 <= hi[0] <= 5.0
    assert 2.0 <= hit[0] <= 4.0
    assert hi[1] == 1.0 and hit[1] == 1.0


def test_onto_set_intervention_delta(build, X):
    """withArgs set-intervention semantics as steerfid uses them: a forced
    tag moves the output, a noop arglist doesn't."""
    from ontologize.ontologizer import Ontologizer, DictIntervention
    model, params = build()
    l, h, k = model.l, model.h, model.k

    def run(arglist):
        Y, _, _, _ = model.apply(params, X, arglist, temperature=0.5,
                                 method=Ontologizer.withArgs)
        return np.asarray(Y)

    noop = [DictIntervention() for _ in range(l)]
    assert np.allclose(run(noop), run(noop))

    arglist = [DictIntervention() for _ in range(l)]
    arglist[1] = DictIntervention(h_set=jnp.array([2]),
                                  k_set=jnp.array([5]))
    delta = run(arglist) - run(noop)
    assert np.abs(delta).max() > 1e-5
    u = steerfid.unit(delta)
    assert np.allclose(np.linalg.norm(u, axis=-1), 1.0, atol=1e-5)

# experiments/effective-information/effinfo.py: the EI / MI estimators, and
# the intervention probe, which must be `Ontologizer.withArgs` with every
# layer's classification kept -- EI is only an estimate of the formal
# reading's do-operator if the probe is that operator.
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "effective-information"))
import effinfo  # noqa: E402

from ontologize.ontologizer import Ontologizer, DictIntervention  # noqa: E402
from conftest import KW  # noqa: E402


def test_ei_identity_channel_is_log_k():
    ei, det, deg = effinfo.effective_information(np.eye(8))
    assert np.isclose(ei, 3.0) and np.isclose(det, 3.0)
    assert np.isclose(deg, 0.0)


def test_ei_constant_channel_is_zero():
    # every entry leads to the same effect: all degeneracy, no information
    row = np.array([0.5, 0.25, 0.25, 0.0])
    ei, det, deg = effinfo.effective_information(np.tile(row, (4, 1)))
    assert np.isclose(ei, 0.0)
    assert np.isclose(det, 0.5) and np.isclose(deg, 0.5)


def test_ei_is_mi_under_uniform_intervention():
    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(6), size=5)     # 5 source entries, 6 effects
    ei = effinfo.effective_information(W)[0]
    mi = effinfo.mutual_information(W / 5 * 1e6, miller_madow=False)
    assert np.isclose(ei, mi)


def test_run_temperature_reads_end_or_constant(tmp_path):
    import json
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"hyper": {"temperature": 1.0,
                                         "temperature_end": 0.03}}) + "\n")
    assert effinfo.run_temperature(tmp_path) == 0.03
    log.write_text(json.dumps({"hyper": {"temperature": 1.5e-4,
                                         "temperature_end": None}}) + "\n")
    assert effinfo.run_temperature(tmp_path) == 1.5e-4
    assert effinfo.run_temperature(tmp_path / "missing") is None


def test_miller_madow_shrinks_independence_bias():
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 8, 2000), rng.integers(0, 8, 2000)
    C = np.zeros((8, 8))
    np.add.at(C, (a, b), 1)
    plug = effinfo.mutual_information(C, miller_madow=False)
    assert plug > 0
    assert abs(effinfo.mutual_information(C)) < plug


@pytest.fixture(scope="module")
def conditioned(build):
    return build(resid_norm=True, resid_const=True)


def noop():
    return [DictIntervention() for _ in range(KW["l"])]


def probe(model, params, X, layer=-1, head=0, entry=0):
    return model.apply(params, X, layer, jnp.int32(head), jnp.int32(entry),
                       0.5, method=effinfo.assignments)


def test_probe_plain_forward_matches_model(conditioned, X):
    model, params = conditioned
    Ps, Y = probe(model, params, X)
    assert jnp.allclose(Y, model.apply(params, X, temperature=0.5), atol=1e-6)
    _, K_last, _, _ = model.apply(params, X, noop(), temperature=0.5,
                                  method=Ontologizer.withArgs)
    assert jnp.allclose(Ps[:, -1].reshape(X.shape[0], -1), K_last, atol=1e-6)


def test_probe_intervention_is_withargs(conditioned, X):
    model, params = conditioned
    arglist = noop()
    arglist[1] = DictIntervention(h_set=jnp.array([2]), k_set=jnp.array([5]))
    Y_ref, K_ref, _, _ = model.apply(params, X, arglist, temperature=0.5,
                                     method=Ontologizer.withArgs)
    Ps, Y = probe(model, params, X, layer=1, head=2, entry=5)
    P0, _ = probe(model, params, X)
    assert jnp.allclose(Y, Y_ref, atol=1e-6)
    assert jnp.allclose(Ps[:, -1].reshape(X.shape[0], -1), K_ref, atol=1e-6)
    # layers up to and including the source keep their own classification;
    # the layer below it reclassifies the intervened residual
    assert jnp.array_equal(Ps[:, :2], P0[:, :2])
    assert not jnp.allclose(Ps[:, 2], P0[:, 2])


def test_effect_counts_match_direct_interventions(conditioned, X):
    model, params = conditioned
    counts = effinfo.effect_counts_fn(model, params, 0, 0.5, soft=False)(
        X, jnp.int32(1))
    assert counts.shape == (KW["k"], KW["l"] - 1, KW["h"], KW["k"])
    assert jnp.allclose(counts.sum(-1), X.shape[0])   # one entry per context
    Ps, _ = probe(model, params, X, layer=0, head=1, entry=3)
    direct = jax.nn.one_hot(Ps[:, 1:].argmax(-1), KW["k"]).sum(0)
    assert jnp.allclose(counts[3], direct)

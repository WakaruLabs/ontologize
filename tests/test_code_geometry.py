# Geometric invariants of the code the interpretability analyses rely on.
# With resid() zeros and an unbiased linear decoder, the output is jointly
# linear in the per-layer classifications -- so the batch-mean code decodes
# to the batch-mean output (the "generic origin" fact: decode(E[p]) equals
# E[Y], which lands on the corpus mean). Also pins the decode_tags.py
# dictionary-decoding shapes for resid mode.
import jax
import jax.numpy as jnp
import pytest

from ontologize.ontologizer import Ontologizer

from conftest import KW

T = 0.5


@pytest.fixture(scope="module")
def conditioned(build):
    return build(resid_norm=True, resid_const=True)


def soft_forward(model, params, X):
    """Final decode plus the per-layer classifications that produced it."""
    def probe(module, X):
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        E_in = module.constinput(E)
        Ps = []
        for i, dictenc in enumerate(module.dictencs):
            P = dictenc.dict.cluster(dictenc.classifier(E_in), T)
            Ps.append(P)
            R = R + dictenc.dict.combine(dictenc.dict.hfwd(P))
            if i < module.l - 1:
                E_in = module.nextinput(X, R, None)
        return module.decode(R), Ps
    return model.apply(params, X, method=probe)


def const_forward(model, params, Ps):
    """Forward with every layer's classification pinned to a constant."""
    def probe(module, Ps):
        R = jnp.zeros((1, module.e_dec))
        for dictenc, P in zip(module.dictencs, Ps):
            R = R + dictenc.dict.combine(dictenc.dict.hfwd(P[None]))
        return module.decode(R)
    return model.apply(params, Ps, method=probe)


def test_mean_code_decodes_to_mean_output(conditioned, X):
    model, params = conditioned
    Y, Ps = soft_forward(model, params, X)
    Y_mean_code = const_forward(model, params, [P.mean(0) for P in Ps])
    assert jnp.allclose(Y_mean_code[0], Y.mean(0), atol=1e-5)


def test_code_is_affine_in_classifications(conditioned, X):
    # decode(origin + (p - origin)) decomposes: moving one layer's code by
    # a delta moves the output by the delta's decode, independent of the
    # other layers' codes -- the property that makes tag directions
    # (W_h^T (e_j - u)) well-defined intervention units
    model, params = conditioned
    _, Ps = soft_forward(model, params, X)
    origins = [P.mean(0) for P in Ps]
    delta = jnp.zeros_like(origins[1]).at[2, 5].add(0.5).at[2, 3].add(-0.5)

    moved = list(origins)
    moved[1] = moved[1] + delta
    d_out = (const_forward(model, params, moved)
             - const_forward(model, params, origins))

    # same delta from a different base point
    base2 = [P[0] for P in Ps]
    moved2 = list(base2)
    moved2[1] = moved2[1] + delta
    d_out2 = (const_forward(model, params, moved2)
              - const_forward(model, params, base2))
    assert jnp.allclose(d_out, d_out2, atol=1e-5)


def test_decode_entries_shapes(conditioned):
    model, params = conditioned
    k, h, l = KW["k"], KW["h"], KW["l"]
    Rs, Ps = model.apply(params, method=Ontologizer.decodeEntries)
    assert Rs.shape == (l * h * k, KW["d_out"])
    assert Ps.shape == (l, h * k, h * k)
    U = model.apply(params, method=Ontologizer.decodeUniform)
    assert U.shape == (l * (1 + h * k), KW["d_out"])

# resid_first, gain_clip and logit_norm, ported from the headline branch.
# logit_norm standardizes the classifier's own output, so every caller sees
# the same logits and the temperature alone sets the spread; resid_first
# hands layer 0 the residual X - decode(0), which is the plain input when
# there is no decoder; gain_clip caps the residual's norm at the row's,
# keeping its direction; and the combinations either branch would compute
# differently are refused.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.ontologizer import Ontologizer

from conftest import KW, B

GAIN = dict(resid_gain=True, resid_norm=True, resid_const=True)
DIRECT = dict(direct=True, signed=True, e_dec=KW["d_out"])


def built(X, seed=0, **kw):
    model = Ontologizer(**{**KW, **kw})
    return model, model.init(jax.random.PRNGKey(seed), X)


def logits0(model, params, X):
    def probe(m, X):
        de = m.dictencs[0]
        U, _ = de.gainshape_in(m.constinput(m.encode(X, 0.0, None)[0]))
        return de.classifier(U)
    return np.asarray(model.apply(params, X, method=probe))


def test_logit_norm_standardizes_each_head_and_keeps_the_order(X):
    raw, params = built(X)
    std = Ontologizer(**{**KW, "logit_norm": True})
    K, Kn = logits0(raw, params, X), logits0(std, params, X)
    assert np.allclose(Kn.mean(-1), 0.0, atol=1e-5)
    assert np.allclose(Kn.std(-1), 1.0, atol=1e-4)
    assert (Kn.argmax(-1) == K.argmax(-1)).all()


def selection0(model, params, X, T=0.5):
    def probe(m, X):
        de = m.dictencs[0]
        U, _ = de.gainshape_in(m.constinput(m.encode(X, 0.0, None)[0]))
        return de.dict.cluster(de.classifier(U), T)
    return np.asarray(model.apply(params, X, method=probe))


def scaled_classifier(params, c):
    params = jax.tree_util.tree_map(lambda a: a, params)
    w = params["params"]["dictencs_0"]["classifier"]["weight"]
    params["params"]["dictencs_0"]["classifier"]["weight"] = w * c
    return params


def test_logit_norm_makes_the_selection_blind_to_the_logits_scale(X):
    std, params = built(X, logit_norm=True, select="softmax")
    big = scaled_classifier(params, 10.0)
    assert np.allclose(selection0(std, params, X), selection0(std, big, X),
                       atol=1e-5)
    # without it, the same rescaling sharpens the code
    plain = Ontologizer(**{**KW, "select": "softmax"})
    assert not np.allclose(selection0(plain, params, X),
                           selection0(plain, big, X), atol=1e-3)


def test_resid_first_is_the_plain_input_without_a_decoder(X):
    a, params = built(X, **GAIN, **DIRECT)
    b = Ontologizer(**{**KW, **GAIN, **DIRECT, "resid_first": True})
    Ya = a.apply(params, X, temperature=0.5)
    Yb = b.apply(params, X, temperature=0.5)
    assert np.allclose(np.asarray(Ya), np.asarray(Yb), atol=1e-6)


def test_resid_first_reads_the_row_about_decode_of_zero(X):
    model, params = built(X, **GAIN, resid_first=True, biased_dec=True)
    params = jax.tree_util.tree_map(lambda a: a, params)
    bias = params["params"]["decoder"]["bias"]
    params["params"]["decoder"]["bias"] = 0.1 * jax.random.normal(
        jax.random.PRNGKey(3), bias.shape)

    def probe(m, X):
        E, _ = m.encode(X, 0.0, None)
        return m.constinput(E), m.decode(jnp.zeros((1, m.e_lat), X.dtype))

    E0, b = model.apply(params, X, method=probe)
    E0, b = np.asarray(E0), np.asarray(b)
    assert np.abs(b).max() > 0.05                    # decode(0) is the bias
    assert np.allclose(E0[:, :-1], np.asarray(X) - b, atol=1e-6)
    assert np.allclose(E0[:, -1], 1.0)
    # the layer's gain is that residual's norm
    def gain(m, X):
        E, _ = m.encode(X, 0.0, None)
        return m.dictencs[0].gainshape_in(m.constinput(E))[1]
    G = np.asarray(model.apply(params, X, method=gain))[:, 0]
    assert np.allclose(G, np.linalg.norm(np.asarray(X) - b, axis=-1),
                       rtol=1e-5)


def test_gain_clip_caps_the_residual_norm_and_keeps_its_direction(X):
    clip, params = built(X, **GAIN, gain_clip=True)
    free = Ontologizer(**{**KW, **GAIN})
    # a residual state that leaves some rows' residuals larger than the
    # rows and others smaller
    R = 0.5 * jax.random.normal(jax.random.PRNGKey(5), (B, KW["e_dec"]))

    def nxt(m, X, R):
        return m.nextinput(X, R, None)

    Dc = np.asarray(clip.apply(params, X, R, method=nxt))[:, :-1]
    Df = np.asarray(free.apply(params, X, R, method=nxt))[:, :-1]
    nx = np.linalg.norm(np.asarray(X), axis=-1)
    nf = np.linalg.norm(Df, axis=-1)
    assert (nf > nx).any()
    assert np.allclose(np.linalg.norm(Dc, axis=-1), np.minimum(nf, nx),
                       rtol=1e-5)
    cos = (Dc * Df).sum(-1) / (np.linalg.norm(Dc, axis=-1) * nf)
    assert np.allclose(cos, 1.0, atol=1e-5)


@pytest.mark.parametrize("kw,match", [
    (dict(resid_first=True, forward="labels", resid_gain=False), "forward"),
    (dict(resid_first=True, encoded=True, e_enc=32, resid_gain=False),
     "encoder"),
    (dict(resid_first=True, resid_const=True, const0=False), "const0"),
    (dict(gain_clip=True, resid_gain=False), "resid_gain"),
])
def test_contradictory_settings_refuse(X, kw, match):
    with pytest.raises(ValueError, match=match):
        Ontologizer(**{**KW, **kw}).init(jax.random.PRNGKey(0), X)


@pytest.mark.parametrize("kw", [dict(resid_first=True, **GAIN),
                                dict(logit_norm=True)])
def test_the_ghost_path_refuses(X, kw):
    model, params = built(X, **kw)
    with pytest.raises(NotImplementedError):
        model.apply(params, X, 0.0, None, temperature=0.5,
                    method=Ontologizer.withGhost)

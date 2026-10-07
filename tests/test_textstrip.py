# textstrip.py: its probes must reproduce the model's own forward and its
# intervention path (withArgs), its contributions must sum to the
# reconstruction, and the pure helpers (ablation masks, head choice, the
# null step, the paired summary, HTML escaping) must do what they say.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import KW, X  # noqa: F401  (X is a fixture)
from ontologize.ontologizer import DictIntervention, Ontologizer
from textstrip import (NONE, UNIF, ZERO, ablate, choose_heads, contrib_probe,
                       forward_probe, null_steps, parse_heads, render_html,
                       summarize, top_heads, whitened_norm)

T = 0.5


def test_ablate_hits_only_its_row_layer_and_head():
    P = jax.random.uniform(jax.random.PRNGKey(0), (4, 3, 5))
    al = jnp.array([0, 1, 0, 0])
    ah = jnp.array([2, 0, 1, 1])
    am = jnp.array([UNIF, UNIF, ZERO, NONE])
    Q = np.asarray(ablate(P, 0, al, ah, am))
    P = np.asarray(P)
    assert np.allclose(Q[0, 2], 1 / 5) and np.allclose(Q[2, 1], 0)
    keep = np.ones(Q.shape[:2], bool)
    keep[0, 2] = keep[2, 1] = False
    assert np.array_equal(Q[keep], P[keep])   # row 1 targets layer 1
    assert np.array_equal(np.asarray(ablate(P, 1, al, ah, am))[0], P[0])


def test_top_heads_and_parse_heads():
    norms = np.array([[0.1, 0.9, 0.3], [0.9, 0.2, 0.5]])
    assert top_heads(norms, 3) == [(0, 1), (1, 0), (1, 2)]
    assert choose_heads(norms, 2) == [(0, 1), (0, 2), (1, 0), (1, 2)]
    assert choose_heads(norms, 2, per_row=1) == [(0, 1)]
    assert parse_heads(["1:2", "0:0"], 2, 3) == [(1, 2), (0, 0)]
    with pytest.raises(AssertionError):
        parse_heads(["2:0"], 2, 3)


def test_null_steps_match_length_and_follow_the_data_difference():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(2, 6, 8))
    w = rng.uniform(0.5, 2.0, 8)
    norms = rng.uniform(0.1, 3.0, 6)
    S = null_steps(A, B, norms, w)
    assert np.allclose(whitened_norm(S, w), norms, rtol=1e-5)
    D = A - B
    cos = (S * D).sum(-1) / np.linalg.norm(S, axis=-1) / np.linalg.norm(D, axis=-1)
    assert np.allclose(cos, 1.0, atol=1e-6)


def cells(row, rows):
    """Cells with chrF v and dNLL 1 - v, so damage agrees across metrics."""
    return [{"row": row, "cond": c, "layer": L, "head": i, "chrf": v,
             "dnll": 1.0 - v, "text": c, "dnorm": 0.1}
            for c, L, i, v in rows]


def test_summarize_pairs_cells_by_row_and_head():
    recs = [{"row": 0, "ref": "r", "lang": None, "cells": cells(0, [
                ("unif_frozen", 0, 1, 0.4), ("null", 0, 1, 0.9),
                ("unif_frozen", 1, 2, 0.8), ("null", 1, 2, 0.8),
                ("unif", 0, 1, 0.3)])},
            {"row": 1, "ref": "r", "lang": None, "cells": cells(1, [
                ("unif_frozen", 0, 1, 0.6), ("null", 0, 1, 0.5)])}]
    s = summarize(recs)
    p = s["unif_frozen_vs_null"]
    for m, sign in (("chrf", 1), ("dnll", -1)):
        assert p[m]["n"] == 3
        assert np.isclose(p[m]["mean_diff"], sign * (-0.5 + 0.0 + 0.1) / 3)
        # (row 0, L0.h1) is the one cell where the ablation is worse
        assert np.isclose(p[m]["frac_worse"], 1 / 3)
        assert np.isclose(p[m]["frac_equal"], 1 / 3)
    assert s["unif_vs_unif_frozen"]["chrf"]["n"] == 1   # only (row 0, L0.h1)
    assert s["unif_frozen_by_layer"]["0"]["dnll"]["n"] == 2
    assert s["only_vs_only_mean"] is None
    assert np.isclose(s["conditions"]["null"]["dnll"]["mean"],
                      np.mean([0.1, 0.2, 0.5]))
    assert s["delta_over_xhat_median"] is None     # no xnorm recorded
    recs[0]["xnorm"] = recs[1]["xnorm"] = 0.5
    assert np.isclose(summarize(recs)["delta_over_xhat_median"], 0.2)


def test_render_html_escapes_decodes():
    recs = [{"row": 7, "ref": "<script>x</script>", "lang": "eng_Latn",
             "cells": cells(7, [("recon", 2, None, 0.5),
                                ("unif_frozen", 0, 1, 0.2)])}]
    for c in recs[0]["cells"]:
        c["text"] = "a < b & c"
    out = render_html(recs, "t", summarize(recs))
    assert "<script>x" not in out and "&lt;script&gt;" in out
    assert "a &lt; b &amp; c" in out


@pytest.mark.parametrize("over", [
    dict(),
    dict(select="ste", resid_gain=True),
    dict(select="ste", resid_gain=True, scaled=True, concat=True),
])
def test_probes_reproduce_the_model(over, X):
    model = Ontologizer(**{**KW, **over})
    params = model.init(jax.random.PRNGKey(0), X)
    l, h = model.l, model.h
    b = X.shape[0]
    none = jnp.zeros(b, jnp.int32)

    Ys = model.apply(params, X, none, none, none, T, method=forward_probe)
    own = model.apply(params, X, temperature=T)
    assert jnp.allclose(Ys[-1], own, rtol=1e-4, atol=1e-5)

    phi, phi_u, zero = model.apply(params, X, T, method=contrib_probe)
    assert phi.shape == (b, l, h, own.shape[-1])
    assert jnp.allclose(zero + phi.sum((1, 2)), own, rtol=1e-4, atol=1e-4)

    # a live uniform ablation is withArgs' h_unif at that layer
    L, i = 1, 2
    Yl = model.apply(params, X, jnp.full(b, L), jnp.full(b, i),
                     jnp.full(b, UNIF), T, method=forward_probe)[-1]
    args = [DictIntervention() for _ in range(l)]
    args[L] = DictIntervention(h_unif=jnp.array([i]))
    Ya, _, _, _ = model.apply(params, X, args, temperature=T,
                              method=Ontologizer.withArgs)
    assert jnp.allclose(Yl, Ya, rtol=1e-4, atol=1e-5)
    assert not jnp.allclose(Yl, own)

    # in the last layer nothing downstream re-reads it, so the frozen
    # ablation equals the live one
    Ll = l - 1
    Ylast = model.apply(params, X, jnp.full(b, Ll), jnp.full(b, i),
                        jnp.full(b, UNIF), T, method=forward_probe)[-1]
    frozen = own + phi_u[:, Ll, i] - phi[:, Ll, i]
    assert jnp.allclose(Ylast, frozen, rtol=1e-4, atol=1e-4)

# experiments/nonlinear-steering: the equal-length steps rescale.py compares,
# the next-token metrics, the hook that leaves BOS alone, the task windows,
# and the pilot's objective and pass criteria.
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "nonlinear-steering"))
import common  # noqa: E402
import pilot  # noqa: E402
import rescale  # noqa: E402
import tasks  # noqa: E402


def test_orth_step_has_the_rescale_length_and_is_orthogonal():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(5, 7, 16, generator=g) * 3
    raw = torch.randn(5, 7, 16, generator=g)
    step = common.orth_step(h, 0.25, raw)
    torch.testing.assert_close(step.norm(dim=-1), 0.25 * h.norm(dim=-1))
    assert (step * h).sum(-1).abs().max() < 1e-4
    # the same length a rescale by 1.25 moves h
    torch.testing.assert_close(step.norm(dim=-1), (1.25 * h - h).norm(dim=-1))


def test_next_token_stats_are_zero_for_an_unchanged_model():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 6, 11, generator=g)
    tokens = torch.randint(11, (2, 6), generator=g)
    kl, ce_r, ce_e, same = common.next_token_stats(logits, logits, tokens)
    assert kl.shape == (2, 4)                     # predictions at positions 1..T-2
    assert kl.abs().max() < 1e-6 and torch.equal(ce_r, ce_e) and bool(same.all())
    kl2, _, _, _ = common.next_token_stats(logits, logits + torch.randn(2, 6, 11, generator=g), tokens)
    assert bool((kl2 >= -1e-6).all()) and float(kl2.mean()) > 0
    # the cross-entropy is that of the actual next token
    lp = torch.log_softmax(logits[:, 1:-1], -1)
    torch.testing.assert_close(ce_r, -lp.gather(-1, tokens[:, 2:, None])[..., 0])


def test_site_edit_leaves_position_0_alone():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=common.LAYER + 1, n_embd=16, n_head=2,
                                       n_positions=32, vocab_size=64)).eval()
    tk = torch.randint(64, (2, 10))
    with torch.no_grad():
        clean = model(tk).logits
        with common.site_edit(model, lambda h: 3.0 * h):
            edited = model(tk).logits
    torch.testing.assert_close(edited[:, 0], clean[:, 0])
    assert not torch.allclose(edited[:, 1:], clean[:, 1:])
    H = common.capture(model, tk)
    assert H.shape == (2, 9, 16)                   # positions >= 1 only


def test_alt_edits_absolute_positions_1_3_5():
    h = torch.ones(1, 6, 4)
    g = torch.randn(1, 6, 4)
    e = rescale.edited(h, "rescale", 2.0, g, "alt")
    # local index 0 is absolute position 1
    assert e[0, :, 0].tolist() == [2.0, 1.0, 2.0, 1.0, 2.0, 1.0]
    assert torch.equal(rescale.edited(h, "rescale", 2.0, g, "all"), 2.0 * h)


def test_windows_prepend_bos_and_skip_short_documents():
    rng = np.random.default_rng(0)
    docs = [list(range(100, 110)), list(range(5)), list(range(200, 230))]
    W = tasks.windows(docs, T=8, per_doc=3, rng=rng)
    assert W.shape == (6, 9)                       # the 5-token document is skipped
    assert (W[:, 0] == tasks.BOS).all()
    for row in W:                                  # each window is a contiguous slice
        assert np.all(np.diff(row[1:]) == 1)


def test_topic_needs_three_hits_and_at_most_one_of_the_other_set():
    text = "Preheat the oven, whisk the flour, then bake. The season starts."
    assert tasks.hits(text, tasks.RECIPES) >= 3
    assert tasks.topic(text, tasks.RECIPES, tasks.SPORTS)
    assert not tasks.topic(text + " The coach and the league.", tasks.RECIPES, tasks.SPORTS)
    assert not tasks.topic(text, tasks.SPORTS, tasks.RECIPES)


def test_softplus_inverse_round_trips():
    for y in (0.1, 1.0, 3.0, 7.0):
        assert math.log1p(math.exp(pilot.softplus_inv(y))) == pytest.approx(y)


def test_objective_scores_each_steer_once_on_its_own_text():
    terms = pilot.objective_terms()
    assert len(terms) == 16 + 16 + 9
    own = {}
    for cond, text, kind in terms:
        if kind == "nll":
            assert cond not in own
            own[cond] = text
    assert own["v1+v2"] == "code" and own["v1"] == "french" and own["shared:sports"] == "sports"
    # KL terms never land on the steer's own text
    assert all(text != own[cond] for cond, text, kind in terms if kind == "kl")
    groups = pilot.by_text(terms)
    assert sum(len(v) for v in groups.values()) == len(terms)
    assert set(groups) == set(tasks.TASKS)

    steers = pilot.Steers(8, [1.0, 3.0, 5.0, 7.0])
    with torch.no_grad():
        steers.v.copy_(torch.arange(8.0))
    conds = steers.conditions()
    torch.testing.assert_close(conds["shared:code"], 3.0 * steers.v, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(conds["v1+v2"], steers.v1 + steers.v2)


def test_verdict_applies_the_fixed_criteria():
    base = {u: {"nll": 3.0, "kl": 0.0} for u in pilot.TEXTS}

    def cond(own, eff, kl):
        return {u: {"nll": 3.0 - (eff if u == own else 0.0), "kl": 0.0 if u == own else kl}
                for u in pilot.TEXTS}

    m = {"base": base}
    for t in tasks.TASKS:
        m[f"single:{t}"] = cond(t, 0.4, 0.01)
        m[f"shared:{t}"] = cond(t, 0.3, 0.05)       # 75% of control, crosstalk 1/6 of efficacy
    m["v1"], m["v2"] = cond("french", 0.3, 0.05), cond("recipes", 0.3, 0.05)
    m["v1+v2"] = cond("code", 0.1, 0.05)            # a quarter of its control
    v = pilot.verdict(m)
    assert v["intensity_pass"] and not v["composition_pass"]
    assert not v["rows"]["v1+v2"]["pass"] and v["rows"]["v1"]["pass"]
    m["shared:sports"] = cond("sports", 0.3, 0.2)   # crosstalk too large
    assert not pilot.verdict(m)["intensity_pass"]

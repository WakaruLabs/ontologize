"""autointerp.py pure helpers: contrastive background selection must never
leak what the score stage grades on, the null control must never grade a
feature against its own description, and both must be deterministic. The
language stage's statistics must reproduce from precision and recall, and
language-matched rows must match, stay below the feature's median, and
never show the describer a row the judge grades."""
import numpy as np
import pytest

from conftest import KW, X  # noqa: F401  (X is a fixture)
from autointerp import (CAL_PROMPT_CHARS, CONTRAST_PROMPT, SUMMARIZE_PROMPT,
                        call_weight, campaign_label, campaign_languages,
                        contrast_rows, detection_accuracy, detection_items,
                        cap_pairing, load_lang_rows, matched_rows,
                        modal_language, needed_rows, next_slot, null_pairing,
                        ols_r2, prf, random_span, read_scores, snippet,
                        span_bins, spearman, stratified_features,
                        trend_residuals)


def test_next_slot_spacing_and_idle_catchup():
    gap = 36.0
    # back-to-back requests space out by exactly gap
    t0, prev = next_slot(0.0, 100.0, gap)
    t1, prev = next_slot(prev, 100.0, gap)
    t2, prev = next_slot(prev, 100.0, gap)
    assert t0 == 100.0 and t1 == 136.0 and t2 == 172.0
    # a long idle period doesn't bank a burst: next call is immediate, once
    t3, prev = next_slot(prev, 10000.0, gap)
    t4, _ = next_slot(prev, 10000.0, gap)
    assert t3 == 10000.0 and t4 == 10036.0


def test_call_weight_tracks_prompt_size():
    # cost-proportional: doubling the prompt doubles the bucket charge
    assert call_weight("x" * CAL_PROMPT_CHARS) == 1.0
    assert call_weight("x" * (2 * CAL_PROMPT_CHARS)) == 2.0
    assert call_weight("hi") == 0.25  # floor: tiny prompts can't burst

    # the calibration unit matches reality: a full plain describe prompt
    # (10 max-length snippets) weighs ~1, its cacts counterpart ~2
    row = "- " + snippet("x" * 999) + "\n"
    plain = SUMMARIZE_PROMPT.format(samples=10 * row)
    cacts = CONTRAST_PROMPT.format(pos=10 * row, neg=10 * row)
    assert 0.8 <= call_weight(plain) <= 1.3
    assert 1.7 <= call_weight(cacts) <= 2.5


def test_stratified_features_degenerate_freq():
    # a dense softmax code makes every latent's firing rate identical (or
    # identically zero); selection must still return n features
    for freq in (np.zeros(500), np.full(500, 0.3)):
        sel = stratified_features(freq, 64, 42)
        assert len(sel) == 64
        assert len(np.unique(sel)) == 64


def _feature(seed=0, n_top=20, n_neg=8, n_rows=4000):
    rng = np.random.default_rng(seed)
    rows = rng.choice(n_rows, n_top + n_neg, replace=False)
    return rows[:n_top], rows[n_top:]


def test_contrast_rows_disjoint_from_scored_items():
    n_desc, n_test, seed = 10, 5, 42
    top_i, neg_i = _feature()
    pool = np.unique(np.concatenate(
        [_feature(s)[1] for s in range(1, 40)] + [neg_i]))
    bg = contrast_rows(7, top_i, neg_i, pool, n_desc, n_test, seed)

    assert len(bg) == n_desc
    assert len(np.unique(bg)) == n_desc
    # never any top row (description positives AND held-out score positives)
    assert not set(bg) & set(top_i.tolist())
    # never a held-out scoring negative
    assert not set(bg) & set(neg_i[:n_test].tolist())
    # hence disjoint from everything the detection prompt will contain
    scored, _ = detection_items(7, top_i, neg_i, n_desc, n_test, seed)
    assert not set(bg) & set(scored.tolist())


def test_contrast_rows_deterministic_per_feature():
    top_i, neg_i = _feature()
    pool = np.unique(np.concatenate([_feature(s)[1] for s in range(1, 40)]))
    a = contrast_rows(3, top_i, neg_i, pool, 10, 5, 42)
    b = contrast_rows(3, top_i, neg_i, pool, 10, 5, 42)
    c = contrast_rows(4, top_i, neg_i, pool, 10, 5, 42)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_contrast_rows_small_pool():
    top_i, neg_i = _feature()
    pool = np.unique(neg_i)  # own negatives only: 3 usable after exclusion
    bg = contrast_rows(7, top_i, neg_i, pool, 10, 5, 42)
    assert len(bg) == 3
    assert not set(bg) & set(neg_i[:5].tolist())


# ---------- null control ----------

def test_null_pairing_is_a_derangement():
    feats = [3, 9, 14, 27, 100, 512]
    pair = null_pairing(feats, 42)
    assert sorted(pair) == feats               # every feature gets a control
    assert sorted(pair.values()) == feats      # every description used once
    assert all(f != donor for f, donor in pair.items())  # never self-graded


def test_null_pairing_deterministic_and_degenerate():
    assert null_pairing([1, 2, 3], 7) == null_pairing([1, 2, 3], 7)
    assert null_pairing([1, 2, 3], 7) != null_pairing([1, 2, 3], 8)
    assert null_pairing([5], 42) == {}         # nothing to swap with
    assert null_pairing([], 42) == {}
    assert null_pairing([1, 2], 42) == {1: 2, 2: 1}


def test_null_pairing_derangement_under_adversarial_seeds():
    # the fixed-point repair must hold for every seed, including ones whose
    # raw permutation is the identity or fixes the last index
    for seed in range(200):
        for n in (2, 3, 5, 8):
            feats = list(range(n))
            pair = null_pairing(feats, seed)
            assert all(f != d for f, d in pair.items()), (seed, n)
            assert sorted(pair.values()) == feats, (seed, n)


def test_null_pairing_donor_text_must_differ():
    # descriptions repeat verbatim across features; a duplicate donor is a
    # self-grade in disguise, so it must be swapped away or dropped
    feats = [1, 2, 3, 4]
    texts = {1: "a", 2: "a", 3: "b", 4: "c"}
    pair = null_pairing(feats, 42, texts=texts)
    assert all(texts[d] != texts[f] for f, d in pair.items())
    assert set(pair) == set(feats)          # all four still have a donor

    # every description identical -> no valid null for anyone
    assert null_pairing(feats, 42, texts={f: "same" for f in feats}) == {}

    # one odd feature out: it can be graded, the duplicates cannot
    pair = null_pairing(feats, 42, texts={1: "a", 2: "a", 3: "a", 4: "b"})
    assert set(pair) == {4} or all(texts[d] != texts[f] for f, d in pair.items())


def test_null_pairing_text_repair_holds_over_seeds():
    feats = list(range(8))
    texts = {f: "dup" if f < 4 else f"uniq{f}" for f in feats}
    for seed in range(100):
        pair = null_pairing(feats, seed, texts=texts)
        assert all(texts[d] != texts[f] for f, d in pair.items()), seed
        assert len(set(pair.values())) == len(pair), seed  # still injective


def test_cap_pairing_subsets_deterministically():
    pair = {f: (f + 1) % 20 for f in range(20)}
    a = cap_pairing(pair, 5, 42)
    assert len(a) == 5
    assert all(pair[f] == d for f, d in a.items())   # donors unchanged
    # a resume must re-pick the same subset or it pays for a second sample
    assert a == cap_pairing(pair, 5, 42)
    assert cap_pairing(pair, 5, 43) != a
    # no cap / cap above size is a no-op
    assert cap_pairing(pair, 0, 42) == pair
    assert cap_pairing(pair, 99, 42) == pair
    assert cap_pairing({}, 5, 42) == {}


def test_read_scores_upgrades_legacy_rows(tmp_path):
    p = tmp_path / "scores.csv"
    p.write_text("feature,mode,precision,recall,f1\n7,acts,1.0,0.8,0.889\n")
    rows, legacy = read_scores(p)
    assert legacy and rows[0]["kind"] == "self"   # pre-null rows are self
    assert rows[0]["f1"] == "0.889"

    p.write_text("feature,mode,kind,precision,recall,f1\n"
                 "7,acts,null,0.5,0.4,0.444\n")
    rows, legacy = read_scores(p)
    assert not legacy and rows[0]["kind"] == "null"
    assert read_scores(tmp_path / "absent.csv") == ([], False)


def test_null_items_identical_to_self_items():
    # the control only swaps the description: same rows, same truth, so the
    # base rate and any positive/negative surface tells are held fixed
    top_i, neg_i = _feature()
    a = detection_items(7, top_i, neg_i, 10, 5, 42)
    b = detection_items(7, top_i, neg_i, 10, 5, 42)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


@pytest.mark.parametrize("over", [
    dict(),
    dict(scaled=True),
    dict(fiber_rank=2),
    dict(scaled=True, concat=True),
])
def test_onto_probe_matches_the_forward(over, X):
    # `onto_probe` replays the stack to read every layer's assignments (the
    # code autointerp, headstruct --onto, splitting and refit consume); with
    # a router or fibers it must still write what the forward writes, or
    # every layer past the first classifies another model's residual
    import jax
    import jax.numpy as jnp
    from autointerp import onto_probe
    from ontologize.ontologizer import Ontologizer

    model = Ontologizer(**{**KW, "select": "ste", "resid_gain": True, **over})
    params = model.init(jax.random.PRNGKey(0), X)
    P = model.apply(params, X, 0.5, method=onto_probe)          # (b, l, h, k)

    def forward(module, X):
        E, _ = module.encode(X, 0.0, None)
        return module.classify(E, X_ref=X, temperature=0.5)[1]  # (l, b, hk)

    Ps = model.apply(params, X, method=forward)
    assert jnp.allclose(P, jnp.moveaxis(Ps, 0, 1).reshape(P.shape), atol=1e-6)


def test_entry_directions_keep_the_bias_out():
    # embed(P) = P_flat @ W + b is affine in the code: one decoded
    # direction per entry, the bias returned once as the offset, and the
    # pair reconstructs any code exactly (with the bias counted once, not
    # once per head)
    from autointerp import entry_directions
    l, h, k, d = 2, 3, 4, 5
    rng = np.random.default_rng(0)
    W = rng.normal(size=(l * h * k, d)).astype(np.float32)
    b = rng.normal(size=d).astype(np.float32)
    embed = lambda c: c.reshape(len(c), -1) @ W + b
    dirs, offset = entry_directions(embed, l, h, k, chunk=7)
    np.testing.assert_allclose(dirs, W, atol=1e-5)
    np.testing.assert_allclose(offset, b, atol=1e-6)
    P = rng.dirichlet(np.ones(k), size=(6, l, h)).astype(np.float32)
    np.testing.assert_allclose(P.reshape(6, -1) @ dirs + offset, embed(P),
                               atol=1e-4)


# ---------- language stratification ----------

def test_modal_language_share_and_ties():
    assert modal_language(np.array(["en", "fr", "en", "de"])) == ("en", 0.5)
    # a tie goes to the label that sorts first
    assert modal_language(np.array(["fr", "de", "fr", "de"])) == ("de", 0.5)


@pytest.mark.parametrize("pred", [
    [1, 2, 3, 4, 5],            # every positive, no false positive
    [1, 2, 6],                  # TP 2, FP 1
    [1, 6, 7, 8, 9, 10],        # TP 1, FP 5
    list(range(1, 11)),         # always match
])
def test_detection_accuracy_recovers_the_confusion(pred):
    # precision and recall as score() writes them, to three decimals
    truth = [1, 2, 3, 4, 5]
    p, r, _ = prf(pred, truth)
    tp, fp = len(set(pred) & set(truth)), len(set(pred) - set(truth))
    lo, hi = detection_accuracy(float(f"{p:.3f}"), float(f"{r:.3f}"), 5)
    assert lo == hi == pytest.approx((tp + 5 - fp) / 10)


def test_detection_accuracy_brackets_a_miss():
    # with no true positive the false positives cannot be recovered
    for pred in ([], [6], [6, 7, 8, 9, 10]):
        p, r, _ = prf(pred, [1, 2, 3, 4, 5])
        assert detection_accuracy(p, r, 5) == (0.0, 0.5)


def test_matched_rows_follow_the_target_languages():
    cand = np.arange(100, 160)
    langs = np.array(["en"] * 20 + ["fr"] * 20 + ["de"] * 20)
    target = np.array(["fr", "en", "fr", "de", "fr"])
    rows, n = matched_rows(target, cand, langs, np.random.default_rng(0))
    assert n == 5 and len(set(rows)) == 5
    lang_of = dict(zip(cand.tolist(), langs))
    assert [lang_of[r] for r in rows.tolist()] == target.tolist()
    # deterministic in the generator's seed
    again, _ = matched_rows(target, cand, langs, np.random.default_rng(0))
    assert np.array_equal(rows, again)


def test_matched_rows_fill_a_shortfall_from_the_rest():
    cand = np.arange(10)
    langs = np.array(["en"] * 9 + ["my"])
    rows, n = matched_rows(np.array(["my"] * 3), cand, langs,
                           np.random.default_rng(0))
    assert n == 1 and 9 in rows       # the one Burmese candidate
    assert len(set(rows.tolist())) == 3
    with pytest.raises(ValueError):
        matched_rows(np.array(["en"] * 11), cand, langs,
                     np.random.default_rng(0))


def test_span_bins_and_random_span():
    assert span_bins([1, 4, 10, 13], 15) == ["1-3", "4-9", "10-12", "13-15"]
    assert random_span(np.array([5.0]), 15) == pytest.approx(1.0)
    assert random_span(np.ones(86), 1) == pytest.approx(1.0)
    assert random_span(np.ones(86), 15) == pytest.approx(
        86 * (1 - (85 / 86) ** 15))


def test_ols_r2_and_spearman():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert ols_r2(2 * x + 1, x) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert ols_r2(rng.normal(size=500), rng.normal(size=500)) < 0.02
    assert spearman([1, 2, 3], [9, 4, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_trend_residuals_fit_only_the_chosen_campaigns():
    # the line is fitted in log rate through `fit`; a campaign outside it is
    # read off the line, and a code that never fires has no residual
    freq = np.array([0.01, 0.1, 1.0, 0.1, 0.0])
    f1 = 0.5 + 0.1 * np.log(np.where(freq > 0, freq, 1.0))
    f1[3] += 0.2
    res = trend_residuals(freq, f1, np.array([True, True, True, False,
                                              True]))
    np.testing.assert_allclose(res[:4], [0, 0, 0, 0.2], atol=1e-12)
    assert np.isnan(res[4])


def test_needed_rows_and_campaign_label(tmp_path):
    np.savez(tmp_path / "features.npz", sel=np.array([3]),
             top_i=np.array([[1, 2]]), neg_i=np.array([[5]]))
    np.savez(tmp_path / "lang_rows.npz", sel=np.array([3]),
             neg_i=np.array([[7]]), bg_i=np.array([[8, 9]]))
    assert needed_rows([tmp_path / "features.npz",
                        tmp_path / "lang_rows.npz"]) == {1, 2, 5, 7, 8, 9}
    assert campaign_label(tmp_path / "m5120_k32" / "sae", tmp_path,
                          "sae") == "m5120_k32"
    assert campaign_label(tmp_path / "onto" / "onto", tmp_path,
                          "onto") == "onto"
    assert campaign_label(tmp_path / "onto", tmp_path, "onto") == "onto"


def test_load_lang_rows_checks_the_sample(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_lang_rows(tmp_path, np.array([1, 2]), 5)
    np.savez(tmp_path / "lang_rows.npz", sel=np.array([1, 2]),
             neg_i=np.zeros((2, 5), int), bg_i=np.zeros((2, 10), int))
    assert load_lang_rows(tmp_path, np.array([1, 2]), 5)["bg_i"].shape \
        == (2, 10)
    with pytest.raises(ValueError):           # drawn for another sample
        load_lang_rows(tmp_path, np.array([1, 3]), 5)
    with pytest.raises(ValueError):           # too few negatives
        load_lang_rows(tmp_path, np.array([1, 2]), 6)


def test_campaign_languages_shares_and_scores(tmp_path):
    import json
    # rows cycle through four languages
    langs = np.array(["en", "fr", "de", "my"] * 10)
    # description rows en en en fr, held-out positives en de
    np.savez(tmp_path / "features.npz", sel=np.array([7]),
             top_i=np.array([[0, 4, 8, 1, 12, 2]]),
             neg_i=np.array([[3, 7, 11]]), freq=np.array([0.2]))
    (tmp_path / "meta.json").write_text(json.dumps(
        {"model": "sae", "n_desc": 4, "seed": 42, "rows": 40}))
    (tmp_path / "scores.csv").write_text(
        "feature,mode,precision,recall,f1\n"
        "7,acts,0.667,1.0,0.8\n7,params,0.0,0.0,0.0\n")
    np.savez(tmp_path / "lang_rows.npz", sel=np.array([7]),
             neg_i=np.array([[36, 13]]), bg_i=np.array([[16, 20, 24, 28]]))
    meta, feats, scored = campaign_languages(tmp_path, langs, 2, ("acts",))
    f = feats[7]
    assert (f["lang"], f["purity"], f["n_lang"]) == ("en", 0.75, 3)
    assert f["pos_share"] == 0.5          # en, de
    assert f["neg_share"] == 0.0          # my, my
    assert f["bg_share"] == 0.0           # the one background row left: 11
    assert (f["bg_share_lang"], f["neg_share_lang"]) == (1.0, 0.5)
    assert len(scored) == 1 and scored[0]["mode"] == "acts"
    # TP 2 of 2, FP 1 of 2: accuracy 3/4
    assert scored[0]["acc_lo"] == scored[0]["acc_hi"] == 0.75


def test_match_draws_matched_rows_below_the_median(tmp_path):
    # a toy top-k SAE and cache, harvested the way harvest() records it
    import argparse
    import json
    import jax
    import sae as sae_mod
    from autointerp import match

    d, m, n_rows, topk = 8, 16, 256, 4
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_rows, d)).astype(np.float32)
    np.save(tmp_path / "cache.npy", X)
    langs = np.array(["en", "fr", "de", "my"] * (n_rows // 4))
    np.save(tmp_path / "cache.langs.npy", langs)
    run = tmp_path / "sae_run"
    run.mkdir()
    params = sae_mod.init_params(jax.random.PRNGKey(0), d, m,
                                 X.mean(0))
    np.savez(run / "params.npz", **{k: np.asarray(v)
                                    for k, v in params.items()})
    (run / "meta.json").write_text(json.dumps(
        {"topk": topk, "groups": 0, "group_fn": "top1"}))
    A = np.asarray(sae_mod.encode({k: np.asarray(v) for k, v in
                                   params.items()}, X, topk))
    sel = np.array([0, 3, 5])
    top_i = np.argsort(-A[:, sel], axis=0)[:20].T
    q50 = np.quantile(A[:, sel].astype(np.float16).astype(np.float32), 0.5,
                      axis=0)
    neg_i = np.stack([rng.choice(np.setdiff1d(
        np.flatnonzero(A[:, f] <= q), t), 8, replace=False)
        for f, q, t in zip(sel, q50, top_i)])
    camp = tmp_path / "campaign"
    camp.mkdir()
    np.savez(camp / "features.npz", sel=sel, top_i=top_i, neg_i=neg_i,
             q50=q50, freq=(A[:, sel].astype(np.float16) > 0).mean(0))
    (camp / "meta.json").write_text(json.dumps(
        {"model": "sae", "ckpt": str(run / "params.npz"), "topk": topk,
         "rows": n_rows, "n_desc": 10, "seed": 42}))
    cfg = argparse.Namespace(dir=str(camp), ckpt="", cache=str(
        tmp_path / "cache.npy"), b=64, sub_batches=4, n_test=5)

    match(cfg)
    lr = np.load(camp / "lang_rows.npz")
    assert np.array_equal(lr["sel"], sel)
    for j, f in enumerate(sel):
        pos, desc = top_i[j, 10:15], top_i[j, :10]
        neg, bg = lr["neg_i"][j], lr["bg_i"][j]
        # matched, low-activation, and outside the feature's top rows
        assert lr["neg_matched"][j] == 5 and lr["bg_matched"][j] == 10
        assert np.array_equal(langs[neg], langs[pos])
        assert np.array_equal(langs[bg], langs[desc])
        assert (A[neg, f] <= q50[j]).all() and (A[bg, f] <= q50[j]).all()
        assert not set(neg) & set(top_i[j]) and not set(bg) & set(top_i[j])
        # the background never shows a row either negative set grades
        assert not set(bg) & (set(neg) | set(neg_i[j, :5]))
    first = {k: lr[k] for k in lr.files}
    match(cfg)                                    # deterministic
    again = np.load(camp / "lang_rows.npz")
    assert all(np.array_equal(first[k], again[k]) for k in first)

    # a checkpoint whose firing rates are not the harvest's is refused
    other = sae_mod.init_params(jax.random.PRNGKey(1), d, m, X.mean(0))
    np.savez(run / "other.npz", **{k: np.asarray(v)
                                   for k, v in other.items()})
    cfg.ckpt = str(run / "other.npz")
    with pytest.raises(ValueError, match="firing rates"):
        match(cfg)

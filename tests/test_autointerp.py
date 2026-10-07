"""autointerp.py pure helpers: contrastive background selection must never
leak what the score stage grades on, the null control must never grade a
feature against its own description, and both must be deterministic."""
import numpy as np
import pytest

from conftest import KW, X  # noqa: F401  (X is a fixture)
from autointerp import (CAL_PROMPT_CHARS, CONTRAST_PROMPT, SUMMARIZE_PROMPT,
                        call_weight, contrast_rows, detection_items,
                        cap_pairing, next_slot, null_pairing, read_scores,
                        snippet,
                        stratified_features)


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

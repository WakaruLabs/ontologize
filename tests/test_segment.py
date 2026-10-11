# Sentence segmentation for the SONAR sentence cache (ontologize.data.segment):
# SaT's threshold split with input newlines, the LCM's cap applied only to
# segments over it, the script check, and the choice of each document's run.
import numpy as np
import pytest

from ontologize.data.langs import MC4_TO_SONAR
from ontologize.data.segment import (capped_spans, cap_spans, dominant_script,
                                     expected_scripts, letter_script, run,
                                     threshold_spans, trim)

TEXT = "One two three. Four five six?\nSeven eight nine ten eleven twelve thirteen fourteen fifteen"


def probs():
    p = np.zeros(len(TEXT))
    p[TEXT.index(".")] = 0.9
    p[TEXT.index("?")] = 0.8
    for w, v in (("ten", 0.2), ("twelve", 0.05), ("fourteen", 0.1)):
        p[TEXT.index(w) + len(w) - 1] = v
    return p


def test_threshold_and_newline_split():
    assert [TEXT[a:b] for a, b in threshold_spans(TEXT, probs(), 0.25)] == [
        "One two three.", "Four five six?",
        "Seven eight nine ten eleven twelve thirteen fourteen fifteen"]


def test_cap_breaks_only_long_segments_at_the_best_boundaries():
    spans = capped_spans(TEXT, probs(), 0.25, 30)
    assert [(TEXT[a:b], c) for a, b, c in spans] == [
        ("One two three.", False), ("Four five six?", False),
        ("Seven eight nine ten", True), ("eleven twelve", True),
        ("thirteen fourteen fifteen", True)]
    assert all(b - a <= 30 for a, b, _ in spans)


def test_cap_makes_the_fewest_cuts():
    q = np.full(50, 1e-3)
    q[19], q[39] = 0.5, 0.4
    assert cap_spans(q, 25) == [20, 40]
    assert cap_spans(q, 100) == []          # under the cap: untouched
    assert cap_spans(np.zeros(50), 20) == [20, 40]   # nothing to cut at: fixed width


def test_trim_drops_zero_width_spaces():
    t = "​ ក ខ​"
    a, b = trim(t, 0, len(t))
    assert t[a:b] == "ក ខ"


@pytest.mark.parametrize("text,script", [
    ("Όμως, θα σας πω", "GREEK"), ("แล้วรถอยู่แถวไหนครับ", "THAI"), ("大财运注册", "CJK"),
    ("これは日本語です", "HIRAGANA"), ("한국어 문장", "HANGUL"), ("Ｆｕｌｌ", "LATIN"),
    ("30/05/2019 !!", None)])
def test_dominant_script(text, script):
    assert dominant_script(text) == script


def test_every_config_has_a_script_and_romanized_ones_are_latin():
    for config, tag in MC4_TO_SONAR.items():
        allowed = expected_scripts(config, tag)
        assert allowed
        if config.endswith("-Latn"):
            assert allowed == {"LATIN"}
    assert dominant_script("これは日本語です") in expected_scripts("ja", MC4_TO_SONAR["ja"])
    assert dominant_script("Όμως, θα σας πω") not in expected_scripts("cy", MC4_TO_SONAR["cy"])
    assert letter_script("7") is None


def test_run_keeps_all_when_few_and_k_consecutive_usable_otherwise():
    rng = np.random.default_rng(0)
    assert run([True, False, True], 16, rng) == [0, 2]
    usable = [i % 3 != 1 for i in range(60)]          # every third unusable
    idx = [i for i, u in enumerate(usable) if u]
    for seed in range(20):
        got = run(usable, 16, np.random.default_rng(seed))
        assert len(got) == 16
        start = idx.index(got[0])
        assert got == idx[start:start + 16]           # consecutive among usable
    starts = {run(usable, 16, np.random.default_rng(s))[0] for s in range(200)}
    assert len(starts) > 10                           # the start is random
    assert run(usable, 16, np.random.default_rng(5)) == run(usable, 16, np.random.default_rng(5))

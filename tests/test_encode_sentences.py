# encode_sentences.py's choice of a document's sentences: the length window,
# the script check, the language identifier's non-linguistic class, and the
# run. A stub stands in for GlotLID, so neither SaT nor fastText is needed.
from types import SimpleNamespace

import numpy as np

from encode_sentences import choose, entry, token_batches, usable
from ontologize.data.langs import MC4_TO_SONAR

CFG = SimpleNamespace(threshold=0.25, max_chars=200, min_chars=20, k=16)


class StubLID:
    """Calls a text non-linguistic if it has more digits than letters, else
    Welsh. (A text with no letters never reaches it: the script check drops
    it first.)"""
    def __init__(self):
        self.f = self
        self.calls = []

    def predict(self, text, k, threshold, on_error):
        self.calls.append(text)
        if sum(c.isdigit() for c in text) > sum(c.isalpha() for c in text):
            return [(0.99, "__label__zxx_Zzzz")]
        return [(0.95, "__label__cym_Latn")]


def doc(lines):
    text = "\n".join(lines)
    return text, np.zeros(len(text))     # newlines alone make the segments


def test_usable_reasons():
    t = "Short.\nΌμως, θα σας πω κάτι σήμερα.\nA sentence long enough to keep.\n2019-05-30 12:00:00 +0100"
    lines = t.split("\n")
    spans, a = [], 0
    for line in lines:
        spans.append((a, a + len(line)))
        a += len(line) + 1
    tag = MC4_TO_SONAR["cy"]
    # no letters at all counts as another script
    assert [usable(t, a, b, "cy", tag, 20, 200) for a, b in spans] == [
        "length", "script", None, "script"]


def test_choose_filters_then_keeps_the_rest_in_order():
    lines = ["Brawddeg gyntaf sy'n ddigon hir i'w chadw.",
             "Όμως, θα σας πω κάτι σήμερα.",                   # Greek in Welsh
             "ID 4F2A-99B3-77C1-0042-8813-5521",               # non-linguistic
             "Ail frawddeg sy'n ddigon hir hefyd, wir.",
             "Byr."]                                           # too short
    text, p = doc(lines)
    lid = StubLID()
    sents, why, nseg = choose(text, p, "cy", MC4_TO_SONAR["cy"], False, CFG, lid, [0, 0])
    assert nseg == 5
    assert [s[3] for s in sents] == [lines[0], lines[3]]
    assert [s[0] for s in sents] == [0, 3]                # segment ordinals
    assert all(text[s[1]:s[2]] == s[3] for s in sents)    # spans index the text
    assert why == {"script": 1, "nonlinguistic": 1, "length": 1}
    assert len(lid.calls) == 3          # only length- and script-passing segments
    # each line is its own segment here, so both are whole lines; the first
    # ends in a full stop (a sentence), the second too
    assert [(s[6], s[7]) for s in sents] == [("sentence", "nn"), ("sentence", "nn")]


def test_entries_from_before_the_marks_read_with_empty_kind():
    old = [3, 10, 52, "Brawddeg sy'n ddigon hir i'w chadw.", "cym_Latn", 0.95]
    new = old + ["line", "nn"]
    assert entry(old) == (*old, "", "")
    assert entry(new) == tuple(new)


def test_cut_document_drops_its_last_segment():
    lines = [f"Brawddeg rhif {i} sy'n ddigon hir i'w chadw." for i in range(3)]
    text, p = doc(lines)
    sents, _, nseg = choose(text, p, "cy", MC4_TO_SONAR["cy"], True, CFG, StubLID(), [0, 0])
    assert nseg == 2 and len(sents) == 2


def test_run_of_k_from_a_long_document():
    lines = [f"Brawddeg rhif {i:02d} sy'n ddigon hir i'w chadw." for i in range(40)]
    text, p = doc(lines)
    lid = StubLID()
    sents, _, _ = choose(text, p, "cy", MC4_TO_SONAR["cy"], False, CFG, lid, [0, 3])
    segs = [s[0] for s in sents]
    assert len(segs) == 16 and segs == list(range(segs[0], segs[0] + 16))
    assert len(lid.calls) == 16         # the identifier sees only the run


def test_a_nonlinguistic_segment_in_the_run_is_replaced_and_the_rest_kept():
    # forty usable sentences with a code in the middle: wherever the run
    # lands, it never holds the code, holds 16, and is consecutive among
    # the usable segments that remain
    lines = [f"Brawddeg rhif {i:02d} sy'n ddigon hir i'w chadw." for i in range(40)]
    lines[20] = "ID 4F2A-99B3-77C1-0042-8813-5521"
    text, p = doc(lines)
    usable_idx = [i for i in range(40) if i != 20]
    for s in range(30):
        sents, why, _ = choose(text, p, "cy", MC4_TO_SONAR["cy"], False, CFG, StubLID(), [0, s])
        segs = [x[0] for x in sents]
        assert 20 not in segs and len(segs) == 16
        start = usable_idx.index(segs[0])
        assert segs == usable_idx[start:start + 16]
        assert why.get("nonlinguistic", 0) <= 1


def test_token_batches_sort_and_respect_both_caps():
    lengths = np.array([5, 300, 40, 40, 7, 120, 2, 64])
    batches = token_batches(lengths, max_tokens=256, max_n=3)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(lengths)))             # each once
    assert [lengths[i] for i in flat] == sorted(lengths)          # by length
    for b in batches:
        assert len(b) <= 3
        assert len(b) == 1 or len(b) * lengths[b].max() <= 256    # pad budget

"""Sentence segmentation for SONAR: SaT's boundaries, the LCM's length cap,
what bounds each segment, a script check and run selection, as pure
functions over a document's text and SaT's boundary probabilities. SaT
predicts per XLM-R subword; wtpsplit returns an array over characters that
holds each token's probability at its last character and zero elsewhere.

`encode_sentences.py` builds the sentence cache with these, and
`experiments/sentence-split` tested them (its notes have the numbers). SaT
itself (wtpsplit) and the language identifier are not imported here: they
are the builder's run-time dependencies, not the package's.
"""
import unicodedata
from collections import Counter
from functools import lru_cache

import numpy as np
from jaxtyping import Float

# trimmed from segment edges with whitespace: Khmer and Lao separate words
# with zero-width spaces, which str.isspace does not count
INVISIBLE = {"​", "﻿"}


def blank(c: str) -> bool:
    return c.isspace() or c in INVISIBLE


def trim(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and blank(text[a]):
        a += 1
    while b > a and blank(text[b - 1]):
        b -= 1
    return a, b


def threshold_spans(text: str, probs: Float[np.ndarray, " chars"],
                    threshold: float) -> list[tuple[int, int]]:
    """SaT's split as `SaT.split` makes it (threshold, then input newlines),
    as trimmed character spans [start, end) of `text`. A boundary after
    character i ends its segment there; the whitespace after it is trimmed,
    as `strip_whitespace` does in wtpsplit's `indices_to_sentences`."""
    cuts = set((np.where(probs > threshold)[0] + 1).tolist())
    cuts |= {i for i, c in enumerate(text) if c == "\n"} | {i + 1 for i, c in enumerate(text) if c == "\n"}
    cuts = sorted(c for c in cuts if 0 < c < len(text)) + [len(text)]
    spans, start = [], 0
    for c in cuts:
        a, b = trim(text, start, c)
        if b > a:
            spans.append((a, b))
        start = c
    return spans


def cap_spans(probs: Float[np.ndarray, " chars"], max_len: int) -> list[int]:
    """Interior cut positions (1..n-1) that break a run of n characters into
    pieces of at most `max_len`, maximizing the summed log probability of the
    boundaries cut: the fewest, most probable extra splits. `probs[i]` is the
    probability of a boundary after character i.

    wtpsplit's own `max_length` maximizes the same objective over a whole
    text, so it packs several sentences into each piece; applied only inside
    a segment over the cap, it is the LCM's "next best split"."""
    n = len(probs)
    if n <= max_len:
        return []
    with np.errstate(divide="ignore"):
        logp = np.log(np.clip(probs, 0.0, 1.0))
    dp = np.full(n + 1, -np.inf)
    dp[0] = 0.0
    back = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        lo = max(0, i - max_len)
        j = lo + int(np.argmax(dp[lo:i]))
        if not np.isfinite(dp[j]):
            continue
        dp[i] = dp[j] + (logp[i - 1] if i < n else 0.0)
        back[i] = j
    if not np.isfinite(dp[n]):
        # no finite-probability cut within reach: fixed-width fallback
        return list(range(max_len, n, max_len))
    cuts, i = [], n
    while i > 0:
        i = int(back[i])
        if i > 0:
            cuts.append(i)
    return sorted(cuts)


def capped_spans(text: str, probs: Float[np.ndarray, " chars"], threshold: float,
                 max_len: int) -> list[tuple[int, int, bool]]:
    """SaT Capped: `threshold_spans`, with every span over `max_len` broken
    by `cap_spans`; each span carries whether the cap made it."""
    return [(a, b, "c" in ends) for a, b, ends in bounded_spans(text, probs, threshold, max_len)]


def bounded_spans(text: str, probs: Float[np.ndarray, " chars"], threshold: float,
                  max_len: int) -> list[tuple[int, int, str]]:
    """`capped_spans` with what bounds each span at its start and at its end,
    as a two-letter code: `n` an input newline or the text's edge, `s` a
    SaT boundary, `c` a cut the cap made. SaT's probabilities are nonzero
    only at token ends, so `s` and `c` both fall there."""
    spans = threshold_spans(text, probs, threshold)

    def between(b: int, a: int) -> str:
        return "n" if "\n" in text[b:a] else "s"

    out = []
    for i, (a, b) in enumerate(spans):
        first = "n" if i == 0 else between(spans[i - 1][1], a)
        last = "n" if i == len(spans) - 1 else between(b, spans[i + 1][0])
        if b - a <= max_len:
            out.append((a, b, first + last))
            continue
        edges = [a] + [a + c for c in cap_spans(probs[a:b], max_len)] + [b]
        pieces = []
        for s, e in zip(edges[:-1], edges[1:]):
            s, e = trim(text, s, e)
            if e > s:
                pieces.append([s, e, "c", "c"])
        pieces[0][2], pieces[-1][3] = first, last
        out += [(s, e, x + y) for s, e, x, y in pieces]
    return out


SENTENCE_FINAL = set(".!?。！？؟۔।॥။።፧…។៕։")
# closing quotes and brackets that may follow a sentence's final mark
# (German closes with the left-hand quotes “ and ‘)
CLOSERS = set("\"')]}»›”’“‘」』〉》】")


def kind(segment: str, ends: str) -> str:
    """`piece` if the cap made either end; `line` if input newlines (or the
    text's edges) bound it on both sides and it does not end in sentence
    punctuation: a title, label or list entry; else `sentence`. A whole line
    that ends in sentence punctuation is a one-sentence paragraph. Scripts
    without sentence-final punctuation (Thai, Lao) mark every whole line a
    `line`; `ends` keeps the raw boundaries for a finer rule."""
    if "c" in ends:
        return "piece"
    if ends == "nn":
        tail = segment.rstrip("".join(CLOSERS) + " ")
        return "sentence" if tail and tail[-1] in SENTENCE_FINAL else "line"
    return "sentence"


def default_threshold(model: str) -> float:
    """SaT's boundary threshold for a model, as wtpsplit chooses it and as
    the paper uses them: 0.25 for the supervised-mixture (`-sm`) models,
    0.01 without limited lookahead, 0.025 otherwise."""
    if "sm" in model:
        return 0.25
    if "no-limited-lookahead" in model:
        return 0.01
    return 0.025


# The first word of the Unicode names of each FLORES-200 script's letters.
# Japanese and Korean text mixes in Han characters.
SCRIPT_WORDS = {
    "Latn": {"LATIN"}, "Cyrl": {"CYRILLIC"}, "Grek": {"GREEK"},
    "Arab": {"ARABIC"}, "Hebr": {"HEBREW"}, "Armn": {"ARMENIAN"},
    "Geor": {"GEORGIAN"}, "Ethi": {"ETHIOPIC"}, "Deva": {"DEVANAGARI"},
    "Beng": {"BENGALI"}, "Guru": {"GURMUKHI"}, "Gujr": {"GUJARATI"},
    "Orya": {"ORIYA"}, "Taml": {"TAMIL"}, "Telu": {"TELUGU"},
    "Knda": {"KANNADA"}, "Mlym": {"MALAYALAM"}, "Sinh": {"SINHALA"},
    "Thai": {"THAI"}, "Laoo": {"LAO"}, "Khmr": {"KHMER"}, "Mymr": {"MYANMAR"},
    "Tibt": {"TIBETAN"},
    "Hans": {"CJK"}, "Hant": {"CJK"},
    "Jpan": {"CJK", "HIRAGANA", "KATAKANA"},
    "Hang": {"HANGUL", "CJK"},
}
# names whose first word is a width or form variant of a script
NAME_PREFIXES = (("FULLWIDTH LATIN", "LATIN"), ("HALFWIDTH KATAKANA", "KATAKANA"),
                 ("HALFWIDTH HANGUL", "HANGUL"), ("KATAKANA-HIRAGANA", "KATAKANA"))


@lru_cache(maxsize=1 << 16)
def letter_script(c: str) -> str | None:
    """The script word of a letter (`LATIN`, `CJK`, ...), or None for a
    non-letter."""
    if not unicodedata.category(c).startswith("L"):
        return None
    name = unicodedata.name(c, "")
    for prefix, word in NAME_PREFIXES:
        if name.startswith(prefix):
            return word
    return name.split(" ")[0] if name else None


def dominant_script(text: str) -> str | None:
    """The script of most of `text`'s letters, or None if it has none."""
    counts = Counter(s for s in map(letter_script, text) if s)
    return counts.most_common(1)[0][0] if counts else None


def expected_scripts(config: str, tag: str) -> set[str]:
    """Script words a config's text may be in. The romanized configs
    (`hi-Latn`, ...) are in Latin script whatever their tag names."""
    if config.endswith("-Latn"):
        return SCRIPT_WORDS["Latn"]
    return SCRIPT_WORDS[tag.split("_")[1]]


def run(usable: list[bool], k: int, rng: np.random.Generator) -> list[int]:
    """Indices of up to `k` consecutive usable segments, unusable ones
    skipped: all of them if there are at most `k`, otherwise `k` starting at
    a uniformly random usable segment that leaves room for all `k`."""
    idx = [i for i, u in enumerate(usable) if u]
    if len(idx) <= k:
        return idx
    start = int(rng.integers(0, len(idx) - k + 1))
    return idx[start:start + k]

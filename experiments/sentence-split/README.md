# sentence-split: SONAR-sized sentences from mC4

The SONAR cache (`mc4_4M`) holds one embedding per document: each row is
the masked mean over a document's first 512 tokens. SONAR is a sentence
encoder, and the LCM team found its auto-encoding degrades past about
250 characters, so they built their training data from sentences split by
Segment any Text (SaT; Frohmann et al. 2024) and capped at 200 characters
("SaT Capped"). This experiment tests that segmentation on our corpus
before a sentence-level cache is built.

## Run

From the repo root. `wtpsplit` is not a project dependency, so it is
layered on for the run:

```bash
uv run --with wtpsplit python experiments/sentence-split/segtest.py
```

It reads the documents `experiments/lang-tags/retag.py texts` replayed
(`data/out/sonar/langtags/texts.jsonl`: 400 per mC4 config, full text) and
writes under `--out` (default `data/out/sonar/sentsplit`).

## What `segtest.py` does

1. Takes the first `--per-lang` documents of each language (default 200),
   each cut to its first `--chars` characters (default 4,000); the last
   segment of a cut document is dropped as a fragment.
2. Runs `--model` (default `sat-3l-sm`, fp16 on the GPU) for per-character
   boundary probabilities, with `SaT.split`'s default stride.
3. Splits as `SaT.split` does: at probabilities above `--threshold` (0.25,
   the `-sm` models' default) and at the input's own newlines
   (`threshold_spans`).
4. Caps: every segment longer than `--max-chars` (200) is broken at the
   fewest, most probable further boundaries (`cap_spans`, a Viterbi over
   the segment's own probabilities). This is the LCM's "use the splitting
   probabilities to find the next best split".

`wtpsplit`'s own `max_length` is not used for step 4: with its uniform
prior the Viterbi maximizes the summed log boundary probability over the
whole text, so it makes as few splits as the cap allows and packs several
sentences into each segment rather than splitting into sentences.

## Outputs

| file | contents |
|---|---|
| `langs.csv` | per language: segments per 1,000 characters, segment length in characters (median, 90th percentile) and SONAR tokens (median, 99th percentile, max), the share of uncapped segments over the cap and of segments the cap made, the share under `--min-chars`, the share of model-made boundaries inside a word (space-separated scripts only), the share of segments ending in sentence-final punctuation, the share of documents with a segment in the eligible window [`--min-chars`, `--max-chars`], and boundary F1 against `--compare-model` |
| `summary.json` | the same pooled, plus throughput |
| `segments.jsonl` | each document's segments as character spans, with whether the cap made each |
| `examples.md` | the first segments of two documents for each language in `--examples` |

## `roundtrip.py`: which model gives SONAR better sentences

```bash
uv run --with wtpsplit python experiments/sentence-split/roundtrip.py
```

With no gold segmentation of mC4, `segtest.py` can only report agreement.
`roundtrip.py` scores two models (default `sat-3l-sm`, `sat-12l-sm`) by the
LCM's criterion for comparing segmenters: each segment is encoded with
SONAR under its config's tag without the L2 step, decoded greedy into the
same language, and scored by chrF2 against itself. Because shorter
segments survive better whatever their boundaries, it reports the scores
within length bins as well as per document. Writes `roundtrip.json` and
`roundtrip_bins.csv`.

## `lidtest.py`: language identification and retagging

```bash
uv run --with fasttext==0.9.3 python experiments/sentence-split/lidtest.py
```

Runs NLLB's fastText language identifier (SONAR's FLORES-200 codes) and
GlotLID over `segtest.py`'s segments and reports, per config, how many
match the config's tag, differ only in script, confidently name another
language (and which), or are uncertain, by length, and how many documents
keep an eligible segment under a filter. It then encodes a sample of
confidently mislabeled segments under the config's tag and the predicted
one and compares their round trips. fastText 0.9.3 is built from source
(no Python 3.13 wheel) and called through its C++ binding, because its
Python wrapper's `predict` breaks under numpy 2. Writes `lid_langs.csv`,
`lid_summary.json`, `lid_examples.md`, `lid_retag.json` and
`lid_retag.csv`. The round-trip comparison of tags is confounded (see
notes): chrF rewards the surface copying a foreign tag produces.

The identifiers' licenses differ: NLLB's is CC-BY-NC-4.0; GlotLID's model
card says "other".

## The cache itself

The tests here chose the settings of the root `encode_sentences.py`, which
builds the sentence cache (`mc4_sent16`), and `encode_after_split.sh`
chained its two stages as systemd units. `backfill_marks.py` adds the
line/sentence marks and the version record to a cache whose split predates
them, from a re-split under another name that it checks chose the same
sentences; `marks_backfill.sh` chains the two:

```bash
uv run --with wtpsplit --with fasttext==0.9.3 python encode_sentences.py split --name mc4_sent16_marks
uv run python experiments/sentence-split/backfill_marks.py
```

## Caveats

- SaT was trained on 85 languages. Eleven of the 86 mC4 configs are not
  among them (ht, lb, lo, mi, ny, sd, sm, sn, so, st, su; mC4's `iw` is
  SaT's `he`), and the five romanized configs (`*-Latn`) are in a script
  SaT saw for those languages only incidentally. SaT takes no language
  code, so it runs on all of them; `langs.csv` is how to see whether it
  works there.
- The mid-word rate cannot be read for scripts that do not separate words
  with spaces (zh, ja, th, lo, km, my); it is left blank there.
- The 12-layer comparison measures agreement, not accuracy: there is no
  gold segmentation of mC4.

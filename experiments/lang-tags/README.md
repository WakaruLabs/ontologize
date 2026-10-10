# lang-tags: what the source-language tags cost the SONAR cache

SONAR's text encoder reads a source-language tag as the first token of
every sequence (`tokenizer.src_lang`; NLLB's convention, `[tag] text
</s>`). `mc4_4M.npy` was built with the tags in `langs.MC4_4M_TAGS`,
which cover en/fr/es/de/zh and give everything else `eng_Latn`. So 80 of
the 86 mC4 configs in `langs.MC4_TO_SONAR` (everything except en,
en-multi, fr, es, de, zh) were encoded under the English tag instead of
their own. Every position attends to the tag and the mean pool includes
it, so the cache holds a different embedding of each of those rows than
SONAR would give with its intended tag. `TokenizeTransform` now tags by
`MC4_TO_SONAR`, and new caches record that as `src_tags` in their
`meta.json`; a cache without the field was built the old way.

`retag.py` measures the size of that difference and whether it matters
for what the project reports. The cache stores no text, so the first
stage recovers documents by replaying the deterministic mC4 stream, as
`autointerp.py`'s texts stage does, checking each row's language against
the `.langs.npy` sidecar.

## Run

From the repo root, in order (each stage reads the previous one's
output under `--out`, default `data/out/sonar/langtags`):

```bash
uv run python experiments/lang-tags/retag.py texts     # ~10 min, network
uv run python experiments/lang-tags/retag.py encode    # ~12 min, GPU
uv run python experiments/lang-tags/retag.py compare   # a few min, GPU JAX
uv run python experiments/lang-tags/retag.py english   # ~2 min, CPU
```

## Stages

| stage | does | writes |
|---|---|---|
| `texts` | replays rows `[0, 86 * --per-lang)` of the stream (the interleave is a strict round robin, so that is exactly `--per-lang` rows of every language) and keeps each row's full text | `texts.jsonl` |
| `encode` | encodes every row under its intended tag (`MC4_TO_SONAR`), and the first `--verify` rows of each language under the tag the cache used, with the cache's pooling (masked mean over the 512-token window, then L2) | `encodes.npz` |
| `compare` | compares the cached rows with their intended-tag versions and runs the shipped models on both | `summary.json`, `langs.csv` |
| `english` | asks whether the English tag makes a text's English content (its share of common English words and boilerplate) a larger factor of its embedding: rank correlation with the tag shift, variance linear in the share, and how well each encoding tells mixed from pure texts within a language | `english.json` |

`encode` pads each batch to its longest row instead of to 512; padding is
masked and M2M100's positions skip pad tokens, so the cached-tag
re-encodes reproduce the cache to float rounding (the `verify` check in
`compare` prints it).

## What `compare` measures

- **Shift.** Cosine between each row's cached and intended-tag vectors;
  the whitened FVU of the cached vector as a reconstruction of the
  intended one, on the same base as `pareto.py`'s FVU so it reads against
  the models' errors; the shift's energy split into a common offset,
  per-language offsets (with their chance level) and the rest; all by
  token count.
- **Language structure**, on both versions of the same rows: eta^2 of the
  language and script partitions (as `experiments/ste-arm/headeta.py`),
  mean within- and between-language cosine (as `embedgeom.py`), an 86-way
  ridge probe holding out every fourth row of each language
  (`headlang.ridge_probe`; a split on the row index itself would hold out
  only half the languages, since the stream cycles through 86), and the
  whitened spectrum's Gaussian reference at 380 and 1900 bits
  (`pareto.gaussian_reference`).
- **Models**, on both versions: whitened FVU (with the cache tail's as a
  held-out reference, since the sample rows are training rows); for
  Ontologizers, per-layer head agreement between the two inputs, against
  the same for an isotropic random shift of each row's own size and
  against chance, and best-head language NMI; for SAEs, the Jaccard
  overlap of active latents.

## The L2 normalization (`norm.py`)

`encode_corpus.py` also divides each embedding by its norm, which SONAR
does not do, and the decoding scripts rescale unit vectors to one corpus
constant (`textfid.SONAR_NORM`). `norm.py` measures what that discards,
from the pre-normalization norms `encode` keeps: the norm's spread and
what it tracks, the whitened FVU of the direction at a constant norm
against SONAR's own embeddings, how well a ridge on the direction predicts
the norm, and SONAR's decoder on the same rows at the true, constant and
predicted norms (plus the cache's own vector), scored by chrF2 between
decodes and by how much of the true embedding a re-encoded decode keeps
beyond a same-language shuffled row.

```bash
uv run python experiments/lang-tags/norm.py   # after encode; ~5 min, GPU
```

Writes `norm.json` and `norm_decodes.jsonl` beside the other outputs.

## Caveats

- The intended tag is `MC4_TO_SONAR`'s. For the five romanized configs
  (`*-Latn`) it names the native script, which their text is not in;
  they are reported separately.
- mC4's language labels are themselves noisy (cld3), so some rows'
  "intended" tag is wrong for their text.
- The sample is the head of the cache, which every model trained on.

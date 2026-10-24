# sentence-split notes

## 2026-10-10: SaT Capped on the replayed documents (`segtest.py`)

**Setup.** `sat-3l-sm` (fp16, RTX 3090) at its default threshold 0.25, split
also at input newlines, every segment over 200 characters broken at the
fewest, most probable further boundaries. 200 documents of each of the 86
mC4 configs from `retag.py texts` (cache rows 0–34,399's documents), each cut
to its first 4,000 characters (31.6M characters in all), the last segment of
a cut document dropped. `sat-12l-sm` on the first 25 per config for
agreement. Outputs in `data/out/sonar/sentsplit/`.

**Throughput.** 749k characters/s (407 documents/s); `sat-12l-sm` 218k/s,
3.4× slower. At these rates segmenting 4,000-character prefixes of all
3,974,400 documents takes about 2.7 h (3l) or 9.3 h (12l), on top of the
stream replay (~3 h).

**Segments.** 433,683, about 25 per document prefix. Median 57 characters
and 21 SONAR tokens (with the tag and `</s>`); 99th percentile 91 tokens,
maximum 201. A sentence cache can encode at a 256-token cap with batches
padded to their longest row, where the document cache pads every row to 512.
99.9% of documents have at least one segment of 20–200 characters (at
worst 98%, zh).

By group (median over configs):

| | configs | segments / 1k chars | median chars | over 200 before cap | made by the cap | under 20 chars | mid-word cuts | end in sentence punctuation | boundary F1 vs 12l |
|---|---|---|---|---|---|---|---|---|---|
| in SaT's 85 languages | 70 | 13.3 | 59 | 6.0% | 13.7% | 15.0% | 0.7% | 49.6% | 0.897 |
| not in them (ht lb lo mi ny sd sm sn so st su) | 11 | 13.8 | 53 | 5.5% | 14.9% | 16.7% | 0.8% | 33.4% | 0.820 |
| romanized (`*-Latn`) | 5 | 14.4 | 46 | 5.3% | 16.8% | 22.4% | 1.4% | 18.1% | 0.824 |

- **SaT under-splits Khmer, Burmese and Lao; the cap does the
  segmenting there.** Before the cap 29% (km), 27% (my) and 19% (lo) of
  segments exceed 200 characters, and half (km, my) or 39% (lo) of their
  final segments are cap-made; next are Urdu (12.5%, 26%) and Thai (10%,
  22%). In Khmer and Burmese the cap's splits mostly land on the full stops
  (`។`, `။`) whose probabilities sat below 0.25, so a lower threshold for
  these scripts might do the same job; untested.
- **The uncovered and romanized configs segment like the rest** by length
  and mid-word rate, but agree less with the 12-layer model (0.82 against
  0.90). `ja-Latn` is lowest (0.56), and its sample is mostly lists of
  romanized names and titles rather than prose. Agreement is not accuracy:
  mC4 has no gold segmentation.
- **Mid-word cuts are rare and mostly not SaT's errors.** The highest rates
  (zu 10.5%, ja-Latn 7.3%, ru-Latn 6.1%, mt 5.2%, so 4.2%, gd 4.1%, mr 4.1%)
  come from the cap breaking material with no boundaries (base64 in mt, DNA
  sequences in so, JavaScript in gd) and from correct cuts in text whose
  separators extraction removed (country and title lists glued together in
  zu and ja-Latn).
- Under 20 characters is not a fragment everywhere: zh (33%) and ja (31%)
  pack a sentence into fewer characters (median 27 and 31 characters, but
  24 and 20 tokens).

**What the examples show about the data, not the segmenter.** Some
documents are mojibake at the source (zh row 84: UTF-8 read as GBK), some
Burmese is in the legacy Zawgyi encoding rather than Unicode, and mC4's
language labels put Chinese and Thai spam pages in the Marathi config;
those rows get the wrong SONAR tag whatever the segmentation. Per-segment
language identification would catch the mislabeled ones and most of the
junk the cap has to break.

## 2026-10-10: 3 against 12 layers by SONAR round trip (`roundtrip.py`)

**Question.** The SaT paper's plots put `sat-12l-sm` above `sat-3l-sm` on
gold segmentations (read off Figures 1 and 4: Ersatz ~94.8 to ~97.5 F1,
OPUS100 ~88.5 to ~91.8, UD ~97.1 to ~97.8, code-switching ~42 to ~54). mC4
has no gold segmentation, so the two are compared by the LCM's criterion:
encode a segment with SONAR (own tag, no L2 step), decode greedy into the
same language, chrF2 against the segment.

**Setup.** The first 25 documents of each config (2,150), first 4,000
characters, both models at threshold 0.25 with the 200-character cap;
64,707 distinct segments, each decoded once.

| | `sat-3l-sm` | `sat-12l-sm` |
|---|---|---|
| segments | 53,751 | 50,536 |
| median characters | 56 | 62 |
| round-trip chrF2 per document (character-weighted) | 0.593 | 0.583 |

Per document, 12 layers score 0.0095 ± 0.0011 lower (better on 33% of
documents, worse on 50%): −0.008 ± 0.001 on SaT's languages, −0.018 ±
0.005 on the eleven it lacks, −0.010 ± 0.005 on the romanized configs.
That is mostly length: the 12-layer model splits less, and longer segments
survive worse. Within length bins the two are level, with mixed signs:

| characters | 3 layers | 12 layers |
|---|---|---|
| under 50 | 0.720 | 0.714 |
| 50–99 | 0.624 | 0.625 |
| 100–149 | 0.560 | 0.564 |
| 150–200 | 0.458 | 0.451 |

So the larger model's better boundaries on gold benchmarks do not make
better SONAR inputs here: at matched length its segments round-trip no
better, and it costs 3.4× the time. `sat-3l-sm` is the choice for the
sentence cache. Round-trip quality itself is modest (0.72 for segments
under 50 characters, 0.45 at 150–200), and falls steadily with length well
inside the cap.

## 2026-10-10: language identification and retagging (`lidtest.py`)

**Setup.** NLLB's fastText identifier (labels in SONAR's FLORES-200 codes)
and GlotLID over the 433,683 segments of `segtest.py`. A segment
"matches" if the top label is its config's tag (`MC4_TO_SONAR`), is
"other" if it names another language at probability ≥ 0.9, "unsure"
otherwise.

**Agreement.** NLLB matches 64.7% of segments (74.0% of characters), other
10.7%, unsure 24.6%; GlotLID matches 61.1%. Agreement rises with length:
NLLB matches 34% of segments under 20 characters, 58% at 20–49, 77% at
50–99, 79% at 100–200. 32 of 86 configs match on at least 80% (median
74%). Filtering to matching segments would leave an eligible segment
(20–200 characters) in 75% of documents (12,920 of 17,187); GlotLID 73%.
Much of the disagreement is not mislabeling:

- **NLLB's `yue_Hant` is a catch-all for non-prose.** It is the top
  "other" label in nearly every config, English included, on product codes,
  dates, ISBNs, all-caps titles and single letters, at probability ≥ 0.9.
  GlotLID instead has non-linguistic classes (`zxx_Zzzz`, `zxx_Latn`),
  which take Maltese's base64 and much of `bg-Latn`.
- **Codes differ for some languages.** GlotLID labels Chinese `cmn_Hani`
  (SONAR `zho_Hans`), Hindi partly `anp_Deva`; NLLB has no romanized
  classes, so the five `*-Latn` configs never match it. GlotLID reads
  `hi-Latn` as romanized Hindi (`hin_Latn`, `hif_Latn`); the other four are
  mostly English or junk.
- **Related languages are confused,** by both identifiers alike, so the
  mislabel cannot be told from the error: ky as Tatar, st as Tswana and
  Northern Sotho, sn as Kinyarwanda, am as Tigrinya, gl as Spanish.

**Real mislabels, where the script settles it.** Welsh is 70% Greek
(both identifiers; examples are plain Greek prose). Marathi is 34% Thai
and Chinese spam. Sindhi is 47% English and Arabic. Smaller: Arabic in
Swedish, Korean and Tamil in Hindi, Spanish in Georgian. So the lang-tags
probe's "Welsh read as Greek" and "Marathi as Thai" confusions
(`experiments/lang-tags/notes.md`) are rows in those languages, not
artifacts of the English tag; and under per-config tags a language probe
can read the tag token itself, which labels those rows by their config.

**The retag check does not measure retagging.** On 2,565 segments NLLB
places in another language, encoding under the predicted tag rather than
the config's raises round-trip chrF2 (decoding both into the predicted
language) from 0.609 to 0.665, but all of that is the `yue_Hant`
segments, decoded into Cantonese they are not in (+0.179, n = 930).
Segments predicted English lose 0.018 ± 0.007 (n = 705); the rest lose
0.011 ± 0.008 (n = 930), and where the scripts differ, the clearest
mislabels, they lose 0.052 ± 0.014 (n = 334; Greek in Welsh −0.12, Thai in
Marathi −0.18). By hand: under its own tag SONAR decodes a Greek segment
as a fluent paraphrase ("Ωστόσο παραμένει το ερώτημα" → "Ωστόσο η ερώτηση
παραμένει", chrF 0.55), and under the Welsh tag it copies the surface form
back (1.0). Round-trip chrF rewards surface copying, so it can compare
segmenters at equal tags but cannot say which tag is right.

## 2026-10-10: the sentence cache (`encode_sentences.py`, `mc4_sent16`)

**Choices.** Up to 16 consecutive usable sentences per `mc4_4M` document
(all 3,974,400), from a random usable start, so a document's rows are a
passage for next-sentence prediction as well as rows for dictionary
learning. Usable: 20–200 characters after SaT Capped (`sat-3l-sm`, 0.25,
200-character cap) on each document's first 8,000 characters, letters
mostly in the config's script (romanized configs: Latin), and not called
non-linguistic by GlotLID. Config tags kept (no retagging), GlotLID's label
stored per row, SONAR's masked mean without the L2 step, float32.

**How many sentences.** Whole documents (the test sample, up to 20,000
characters): median 1,534 characters, 17 usable sentences (mean 30); 87%
have at least 4, 53% at least 16, 29% at least 32. Long runs would favor
long documents and starve some configs (16 or more in only 17% of jv and lb
documents), which is why runs are "up to" 16 over every document rather
than exactly 16 over the documents that have them.

**Trial (first 5,120 documents).** 56,853 sentences, 11.1 per document; 49%
of documents give the full 16, 3% none. Of 156,112 segments, 25,199 failed
the length window and 10,631 the script check; GlotLID dropped 705 of those
it saw. Every row's language matches `mc4_4M`'s for its document, and stored
rows match a direct encode (cosine ≥ 0.9999999). Sentence embeddings have
median norm 0.21 (documents ~0.30). The faster builder (GlotLID only on the
chosen run, which is re-chosen if it holds a non-linguistic segment) gives
the same run as filtering first for 5,101 of 5,120 documents; the rest start
elsewhere because a non-linguistic segment outside the run no longer shifts
the random start.

**Throughput.** Split: ~310 documents/s steady with SaT on the GPU, a reader
process and 6 workers (it was 91/s in one process, GlotLID alone at ~5,000
calls/s), ~3.6 h for the corpus. Encode: grouping a batch by tag held
batches to a few hundred sentences and the GPU to ~210 sentences/s at any
batch size; batches mixed across tags (the tag is only the first token)
and capped at 65,536 padded tokens run at ~530–565 sentences/s in float32
(~16k tokens/s, the document cache's rate). TF32 matmuls run at ~930/s with
cosine to float32 ≥ 0.999986 (median 0.9999999, relative error median 3e-4).
Decoded (the decoder in float32, greedy, into each sentence's own language;
24 trial sentences per config, 2,064), TF32 and float32 embeddings give
identical text for 99.2%; where they differ the two decodes share chrF2
0.70, in sentences the decoder already garbles, and round-trip chrF2
against the original is 0.6292 against 0.6291 (paired +0.0001 ± 0.0002).

**Run.** The split started 2026-10-10 13:37 as the systemd unit
`sent-split` (restart on failure; log `mc4_sent16.split.log`), at ~265
documents/s. The encode is queued as the unit `sent-encode`
(`encode_after_split.sh`: waits for the split to complete, then
`encode --tf32`; restart on failure; log `mc4_sent16.encode.log`).

## 2026-10-11: checks from the SaT reading (`scratch/literature/segment-any-text.md`)

On 30 documents per config (2,580; first 8,000 characters), `sat-3l-sm`:

- **Probabilities sit only at token ends.** SaT predicts per XLM-R subword;
  wtpsplit's per-character array puts each token's probability on its last
  character and exactly 0 elsewhere (78,991 positions above 1e-4, all at
  token ends). The cap's search therefore cuts only at token ends: 24 of
  its 7,151 cuts were not, all fixed-width fallbacks in stretches with no
  boundary probability at all.
- **fp16 is fp32's segmentation.** Identical for 2,556 of 2,580 documents,
  boundary F1 0.9997 (lowest 0.80).
- **A third of usable segments are whole lines.** SaT was trained to
  predict newlines stripped from its input, not to read them, and the split
  cuts at every input newline, as `SaT.split` does by default. Of 60,475
  usable segments, 29.8% have a newline at both ends and no SaT boundary,
  and only 24% of those end in sentence punctuation (64–67% of the rest):
  titles, labels, list entries and product names ("Schatzmeisterin: Frau
  Margit Kalus", "Referencia: 85612-Alice"). Without the newline cut these
  would mostly run into the next line rather than stand as sentences, so
  `mc4_sent16`'s rows are SONAR-sized units of text, a third of them lines,
  not only sentences. The sentences file stores spans but not documents, so
  which rows are lines cannot be read off the cache; a rule that keeps only
  sentences needs a new split, or a filter on final punctuation, which
  would also drop the scripts that do not use it (th, lo).

The paper's OPUS100 scores for Khmer (86.6 F1) and Burmese (89.4) at three
layers suggest their under-splitting here is mC4's (Burmese partly Zawgyi,
not Unicode) more than SaT's.

## 2026-10-11: marks, the threshold default and versions

`mc4_sent16` finished 2026-10-11: 44,113,823 rows from 3,974,400 documents
(11.1 per document), TF32, at ~960 sentences/s.

The builder now marks each chosen segment (`segment.kind`) by what bounds
it (`segment.bounded_spans`: `n` an input newline or the text's edge, `s` a
SaT boundary, `c` a cut the cap made): `piece` if the cap made either end;
`line` if newlines bound both ends and it does not end in sentence
punctuation (closing quotes allowed after the mark); else `sentence`, which
includes whole lines that are one-sentence paragraphs. The threshold
defaults by model as wtpsplit's does (0.25 for `-sm`, 0.025 otherwise), and
a split records `wtpsplit`, fastText, torch and transformers versions and
the SaT and GlotLID revisions in its settings, so a resume cannot mix two.

On the first 2,560 documents the marking builder chose the same sentences
as `mc4_sent16` for every document. Among chosen rows: 58.7% sentences,
22.3% lines, 19.0% pieces; boundary codes `nn` 29.7%, `ss` 21.4%, `sn` 15.1%,
`ns` 14.8%, cap-made 19.0%. Lines are commonest in `bg-Latn` and cy (58%), mt
(55%), zu (52%), lb (47%), and rarest in ml, kn, ta (8%) and en (1%): C4's
`en` config was filtered to lines ending in terminal punctuation, the mC4
configs (`en-multi` included) were not.

`mc4_sent16`'s split predates the marks, so they are backfilled from a
re-split (`marks_backfill.sh`, the systemd unit `sent-marks`, started
2026-10-11 08:02; log `mc4_sent16_marks.log`): `backfill_marks.py` checks the
re-split chose the same sentences for every document, then writes
`mc4_sent16.kind.npy`, swaps in the marked sentences file and adds the
versions to the metadata. The embeddings are not touched. Tested on the
5,120-document trial: every document matched.

## 2026-10-12: what the cache looks like (`cachestats.py`)

The marks backfill finished 2026-10-12 00:47. Outputs of `cachestats.py
scan` and `report` are in `data/out/sonar/sentcache/`.

**Rows.**
- 44,113,823 rows from 3,974,400 documents: 11.1 per document, and 2.7%
  of documents give none.
- No row is zero or has a non-finite entry.
- Norms: median 0.208, 1%–99% 0.131–0.330, range 0.090–0.439. Norm rises
  with length (rank correlation 0.49). Medians by kind: sentences 0.208,
  lines 0.197, pieces 0.226.
- Lowest median norm: km 0.161, my 0.167, te 0.168. Highest: ru-Latn 0.272,
  eo 0.247, ja-Latn 0.246.
- Characters: median 77, 10%–90% 28–170. Medians by kind: sentences 75,
  lines 45, pieces 147.
- Kinds over the whole cache: 59.1% sentences, 21.6% lines, 19.4% pieces.

**Repeated texts.**
- There are 42,754,793 distinct (config, text) pairs, so 3.1% of rows copy
  an earlier row. 4.4% of rows have a text that occurs more than once, and
  0.54% one that occurs 100 times or more.
- Copies are concentrated in a few configs: mt 17%, gd 13%, hi-Latn 9.9%,
  ja-Latn 9.1%, ru-Latn 8.5%, sd 7.8%.
- The most repeated texts (`repeats.csv`) passed both the script check and
  GlotLID's non-linguistic classes:
  - minified JavaScript (gl, 3,645 copies of one line; gd);
  - lists of Indian cities and phone numbers (hi-Latn, about 2,000 copies
    each);
  - lists of spam URLs (mt, about 1,000 each);
  - "AAAA…" (so) and porn tag lists (el-Latn);
  - templated hotel blurbs (ca) and WikiTrans boilerplate (eo).

**Against `mc4_4M`.** 262,144 uniform rows of each cache. "Unit" puts the
sentence rows on the sphere, where `mc4_4M`'s already are; "whitened" uses
each cache's own MSE weights.

| | `mc4_sent16` | `mc4_4M` |
|---|---|---|
| mean cosine between rows | 0.069 | 0.318 |
| participation ratio, unit | 243 | 109 |
| participation ratio, whitened | 411 | 270 |
| components for 50% / 90% of variance, unit | 177 / 695 | 161 / 660 |
| language's share of variance, raw | 4.7% | 6.5% |
| language's share of variance, unit | 5.1% | 6.5% |
| language's share of variance, whitened | 4.4% | 5.4% |
| held-out language accuracy, LDA on unit rows | 80.8% | 75.4% |

The kind explains 0.7% of the whitened variance. The language explains
6.2% within lines, 5.6% within pieces and 4.0% within sentences.

**The same documents.** 19,441 documents with rows, out of 20,000 drawn.
- The mean of a document's sentence rows ranks the document's own
  `mc4_4M` row first:
  - for 89.6% of the 1,382 documents in configs `mc4_4M` tagged with
    their own code, and 81.0% of the rest;
  - among documents of the same language only, 94.4% and 88.7%.
- Median centered cosine between the two: 0.47 and 0.42.

Mean cosine of sentence pairs, centered, and in parentheses with each
language's mean removed:

| pair | cosine |
|---|---|
| adjacent | 0.255 (0.216) |
| one document, two or more apart | 0.203 (0.162) |
| one language, two documents | 0.051 (−0.001) |
| two languages | about 0 (about 0) |

**The MSE weights** (inverse variance, mean 1, as `encode_sentences.py`
writes them).
- Coordinate 837 is nearly constant: mean 0.00495, standard deviation
  9.2e-5, so |mean|/sd is 54.
  - What variation it has tracks length: rank correlation −0.84 with
    characters and −0.77 with norm. Language explains 7% of it.
  - It takes weight 837, which is 82% of the weights' sum.
- The next largest weight is coordinate 652's at 13 (mean −0.0120, sd
  7.4e-4; rank correlation 0.90 with norm).
- The median weight is 0.17, and max/min is 92,317.
- `mc4_4M`'s two largest weights are on the same coordinates, but at 27
  and 3.2 (median 0.99, max/min 1,682). Its L2 step turns coordinate 837
  into a readout of 1/norm: on sentence rows put on the sphere, the rank
  correlation is 0.999. Its relative spread (sd/|mean|) goes from 0.019 to
  0.22.
- The two caches' log weights correlate 0.72.

**Why coordinate 837 is nearly constant** (`plane`, on the same 262,144
rows). SONAR's encoder ends in a LayerNorm, whose normalized vector sums
to zero over coordinates. Every token therefore satisfies
Σ_d (x_d − β_d)/γ_d = 0, and so does every masked mean of tokens. The rows
lie on a 1023-dimensional hyperplane whose normal is proportional to 1/γ.
- The rows satisfy the constraint to a median relative error of 1.5e-8.
- The norm's gain on coordinate 837 is −0.00012, against a median |γ| of
  0.080; coordinate 652's is next at −0.00076. So the normal puts 97% of
  its squared length on coordinate 837.
- The covariance's smallest eigenvalue is 1.2e-19, against a median of
  3.0e-5. Its eigenvector is the normal (|cosine| 1.0000).
- Coordinate 837 solved from the constraint and the other 1023
  coordinates matches it to within 1.9e-9, against its sd of 9.2e-5.

So coordinate 837 carries nothing the other coordinates do not fix. Its
small variance is that of a direction with none, lying nearly on its
axis. In `mc4_4M` the L2 step divides the constraint by each row's norm,
which is why there the coordinate reads out 1/norm.

What the weights do:
- **Whitened FVU is unaffected.** Inverse-variance weights give every
  coordinate the same whitened variance, so coordinate 837 is 1/1024 of
  it, like any other.
- **Vectors not shaped like the data are measured mostly on coordinate
  837.** It carries 82% of an isotropic direction's expected whitened
  squared norm (`mc4_4M`: 2.7%). This reaches random-direction nulls,
  whitened cosines between decoder atoms, and steering budgets in the
  whitened norm. Because coordinate 837 is nearly the plane's normal,
  its weight mostly penalizes leaving the LayerNorm plane, not getting
  content wrong.
- **The whitened MSE's scale is smaller.** Whitened variance per row is
  0.0072, against `mc4_4M`'s 0.61, 84× smaller. Raw variance per row is
  0.044 against 0.68: 15× of the ratio is the missing L2 step. Loss
  weights tuned against `mc4_4M`'s MSE do not carry over unchanged.
- **Without coordinate 837**, the remaining weights renormalized to mean
  1 have max 71 (coordinate 652), median 0.95 and max/min 1,437. Whitened
  variance per row is then 0.040. Coordinate 837 can be solved from the
  constraint before decoding, so leaving it out of the loss puts a
  reconstruction back on the plane without losing anything.
- **On `mc4_4M`'s trained models**, the error on this coordinate depends on
  the architecture. The k = 32 SAEs have 0.25× a typical coordinate's FVU
  on it, the hard stack 3.8–4.4× and `resid_nc` 14×. There it holds at most
  0.41% of any model's whitened error, apart from the k = 5120 SAE.
  Details, and what it suggests for this cache, are in
  `experiments/ste-arm/notes.md` ("The coordinate the MSE weights most").

## 2026-10-13: coordinate 837's weight in training (`mask_runs.sh`)

Three architectures trained on `mc4_sent16` in pairs, one run under each
weights file:
- `_w837`: the cache's MSE weights.
- `_no837`: `cachestats.py mask`'s copy, with coordinate 837 at 0 and every
  other weight unchanged.

All six runs used the cache's first 12.8M rows and the shipped recipes:
- **k32 SAE:** `sae.py --m 5120 --topk 32 --train-rows 12800000`, 144,000
  steps of 4,096.
- **Hard stack:** `train_ste.py` with `ste_h76_init01`'s flags, 50,000
  steps of 256.
- **Direct k = 128:** the same, plus `--h 54 --k 128 --signed --direct`.

The runs are in `data/out/sonar/sae_sent16/m5120_k32_*` and
`data/out/sonar/sent16/ste_{h76_init01,k128_h54_direct}_*`. They trained as
the unit `sent16-mask` from 2026-10-12 18:54 to 2026-10-13 02:08. The
direct `_w837` run ran out of GPU memory at step 47,634 and resumed from
its checkpoint at 47,500. Every other run finished in one attempt.

`coordfvu.py --coord 837` scored them on the last 32,765 rows, the
whole-document tail inside both trainers' holdouts, once under each
weights file. The results are in `data/out/sonar/coordfvu_sent16_{w837,no837}/`.
The SAE scores match `sae.py`'s own, 0.6705 and 0.6682. Coordinate 837's
standard deviation is 9.1e-5, which is 0.014 of the median coordinate's.

| architecture | trained | FVU, 1023 free coordinates | FVU, full weights | 837's FVU (× median coordinate's) | 837's share of the error, full weights |
|---|---|---|---|---|---|
| k32 SAE | with the weight | 0.6710 | 0.6705 | 0.19 (0.28×) | 0.03% |
| | without | 0.6682 | 2.02 | 1,396 (2,044×) | 67% |
| hard stack | with | 0.8702 | 0.8873 | 18.6 (21×) | 2.0% |
| | without | 0.9075 | 2.84 | 1,998 (2,175×) | 68% |
| direct k = 128 | with | 0.5817 | 0.6014 | 20.9 (36×) | 3.4% |
| | without | 0.5813 | 21.0 | 21,046 (36,162×) | 97% |

The SAE trained with the weight ends with 265 of its 5,120 latents dead;
trained without it, none.

Movement between the Ontologizer runs' last two kept checkpoints, 250
steps apart. "Parameters" are the decoder's row for the coordinate, or the
direct run's atoms.

| run | FVU 250 steps earlier, 1023 coordinates | 837's drift, of its variance (× median) | parameters' change on 837, × median | their spread, × median |
|---|---|---|---|---|
| hard stack, with | 0.8704 | 9.7 (910×) | 1.97 | 0.24 |
| hard stack, without | 0.9076 | 34 (6,088×) | 0 | 0.99 |
| direct k = 128, with | 0.5813 | 17 (373×) | 7.1 | 0.25 |
| direct k = 128, without | 0.5808 | 1,780 (39,300×) | 0.41 | 3.5 |

- **For the SAE, the weight costs latents, not fit.**
  - Trained with it, the SAE fits coordinate 837 better than a typical
    coordinate (0.28×).
  - But 265 of its latents die, and none die without the weight.
  - Its FVU on the free coordinates is 0.0028 worse with the weight.
- **The hard-code models do not fit the coordinate.**
  - Trained with the weight, their FVU on it is 21× (stack) and 36×
    (direct) a typical coordinate's.
  - Their output there moves by 9.7× and 17× its variance in 250 steps.
    That is 910× and 373× a typical coordinate's movement on its own
    scale.
  - The parameters that write it move 2.0× (decoder rows) and 7.1×
    (atoms) as much as a typical coordinate's. Their spread is a quarter
    of the median coordinate's, against the data's 0.014.
- **So the full weights bias the comparison against them.**
  - Under the full weights, coordinate 837 holds 2.0% of the stack's
    whitened error and 3.4% of the direct run's, against 0.03% of the
    SAE's.
  - Their full-weight FVUs therefore exceed their free-coordinate FVUs by
    0.017 and 0.020, while the SAE's do not differ.
  - These runs are far from converged (the stack is at 0.87), and the
    share grows as the rest of the error falls.
  - The extrapolation from `mc4_4M` (ste-arm notes) put coordinate 837's
    FVU at about 220× a typical coordinate's for the stack, 45× for the
    direct run and 14× for the SAE. Measured, it is 21×, 36× and 0.28×.
- **The effect on the free coordinates has no consistent sign.**
  - Training with the weight leaves the SAE 0.0028 worse, the direct run
    0.0004 worse, and the stack 0.037 better (0.8702 against 0.9075).
  - Each run's FVU moves by at most 0.0005 over its last 250 steps, so
    the stack's gap is not end-of-run jitter.
  - One pair cannot separate an effect of the weight from two
    trajectories diverging. A second seed pair of the stack would.
- **Without the weight, nothing ties the output on coordinate 837 to the
  data.**
  - Every masked model's FVU on it is 1,396 to 21,046.
  - In raw units its error there is 0.65–2.7× the median coordinate's,
    off the LayerNorm plane. The stack's decoder row for it never moves
    from its initialization.
  - So a masked model's reconstruction needs coordinate 837 solved from
    the plane before it is decoded or compared in raw space.
- **Side result, with mc4_4M's recipes untuned:** on this cache and at
  these lengths, the direct k = 128 stack fits best (0.58), then the k32
  SAE (0.67), then the hard stack (0.87–0.91). On `mc4_4M` the hard stack
  converged far ahead of both. `mc4_sent16`'s typical coordinate is about
  4× smaller than `mc4_4M`'s, so the same learning rates are relatively
  hotter here.

## 2026-10-22: the full runs under the masked weights (`full_runs.sh`, `sae_whole_run.sh`)

Three runs on the whole cache, all under `mse_weights_no837.npy`, with the
shipped recipes:
- the direct k = 128 stack and the hard stack, 2 epochs each (344,383 steps
  of 256), unit `sent16-full`;
- the k32 SAE, 144,000 steps of 4,096 with block-shuffled batches
  (`--block-rows 65536`), unit `sent16-sae`.

All three finished on 2026-10-14, after a host crash on 2026-10-13 and
repeated GPU out-of-memory restarts caused by other jobs on the GPU. The
hard stack resumed from step 80,250, because its step-80,500 checkpoint was
zeroed by the crash; that checkpoint is in
`data/out/sonar/sent16/corrupt/`.

`coordfvu.py --coord 837` scored them on the CPU on the last 32,765 rows
(`data/out/sonar/coordfvu_sent16_e2/`). FVU is over the 1023 free
coordinates; "250 steps earlier" is the previous kept checkpoint.

| run | FVU | 250 steps earlier |
|---|---|---|
| k32 SAE | 0.6679 (its own log: 0.6680) | — |
| hard stack | 0.8728 | 0.8725 |
| direct k = 128 | **1.0337** | 1.0335 |

The direct run ends worse than the constant predictor, against 0.5813 for
its 50k-step run on the head of the cache (2026-10-13 section). It
degraded during training, not at evaluation:
- Its training MSE (`loss.csv` column 1, 1000-step means) fell from 5.9e-6
  to 4.4e-6 by step 25,000, then rose steadily to 7.3e-6 at the end.
- The two stacks' held-out FVU over training MSE agree to within 2%, so the
  held-out score follows the training error.
- The hard stack's training MSE fell throughout, 7.1e-6 to 6.3e-6.

**What it traded the reconstruction for.** The total loss stayed flat from
step 25,000, because it is almost all the batch mean-entropy term:
`--s-hm 1e-4` × KL_m ≈ 2.05.

| run | MSE's share of the loss, step 25k → end | cossim_k_max, 25k → end |
|---|---|---|
| direct k = 128, `mc4_sent16` | 2.1% → 3.5% | 0.05 → 0.99 |
| hard stack, `mc4_sent16` | 12.1% → 12.0% | 0.65 → 0.99 |
| direct k = 128, `mc4_4M` (`ste_k128_h54_direct`) | 51% → 60% | 0.04 → 0.32 |
| hard stack, `mc4_4M` (`ste_h76_init01`) | 77% → 77% | 0.42 → 0.30 |

- **The weight's scale.** The recipe's auxiliary weights were set against
  `mc4_4M`'s MSE, which is about 2e-4. On this cache the MSE is 4–7e-6,
  about 50× smaller, so the same `--s-hm` leaves the reconstruction a few
  percent of the loss.
- **Collapse.** KL_m sits at 2.05 for the whole run on both caches, and the
  direct arm's MSE also rose on `mc4_4M` (2.2e-4 to 3.2e-4). On this cache,
  in both stacks, some head's atoms collapsed onto one another
  (cossim_k_max 0.99).
- **What to change.** Before more runs on this cache, rescale the auxiliary
  weights to its loss scale (`s_hm` about 50× smaller), or scale the MSE
  weights so the whitened variance per row matches `mc4_4M`'s. Then rerun.

## 2026-10-24: the full runs with the mean-entropy weight rescaled (`full_runs.sh 2e-6`)

The two Ontologizer runs of the 2026-10-22 section again, with
`--s-hm 2e-6`: the recipe's 1e-4 divided by 50, the factor by which this
cache's MSE is smaller than `mc4_4M`'s. Everything else is the same: the
masked weights, 2 epochs, and the direct k = 128 stack before the hard
stack.
- **Where:** unit `sent16-shm`, 2026-10-23 01:07 to 2026-10-24 03:48; runs
  `data/out/sonar/sent16/*_no837_e2_shm2e-6`.
- **Interruptions:** both resumed from their own checkpoints.
  - The home volume filled at step 177,500 of the direct run, so every save
    after it failed. Training resumed once space was freed.
  - Another job's memory on the GPU caused one out-of-memory restart.

`coordfvu.py --coord 837` scored both on the CPU, on the same 32,765 rows
under the same masked weights as before
(`data/out/sonar/coordfvu_sent16_e2_shm2e-6/`). FVU is over the 1023 free
coordinates.

| run | FVU, `s_hm` 1e-4 (2026-10-22) | FVU, `s_hm` 2e-6 | 2e-6, 250 steps earlier |
|---|---|---|---|
| k32 SAE (no `s_hm`) | 0.6679 | — | — |
| hard stack | 0.8728 | **0.6940** | 0.6941 |
| direct k = 128 | 1.0337 | **0.4997** | 0.4997 |

The diagnostics come from `loss.csv`, as 1,000-step means: steps 0–1,000,
around step 25,000, and the last 1,000 steps. The script reproduces the
2026-10-22 table's numbers for the 1e-4 runs.

| run | training MSE: start → 25k → end | MSE's share of the loss, 25k → end | cossim_k_max, 25k → end | KL_m, 25k → end |
|---|---|---|---|---|
| direct, 1e-4 | 6.5e-6 → 4.4e-6 → 7.3e-6 | 2.1% → 3.5% | 0.05 → 0.99 | 2.05 → 2.04 |
| direct, 2e-6 | 6.5e-6 → 4.3e-6 → 4.0e-6 | 41% → 40% | 0.10 → 0.89 | 2.05 → 2.04 |
| hard stack, 1e-4 | 7.3e-6 → 6.5e-6 → 6.3e-6 | 12.1% → 12.0% | 0.65 → 0.99 | 0.45 → 0.45 |
| hard stack, 2e-6 | 7.3e-6 → 6.3e-6 → 5.0e-6 | 69% → 68% | 0.68 → 0.53 | 0.75 → 0.53 |

- **The rescale removes the degradation.**
  - The MSE's share of the loss is now 40–69%, near `mc4_4M`'s balance
    under the recipe (51–77%, 2026-10-22 table).
  - The direct run's training MSE falls through most of the run; its
    lowest 1,000-step mean is at step 320,500. It now holds out at 0.4997,
    against 0.5813 for its 50k-step run on the head of the cache.
  - The hard stack's falls throughout. Its lowest 1,000-step mean is at
    step 342,500, so it may not have converged at 2 epochs.
- **Against the SAE.** The direct stack's 0.4997 is below the k32 SAE's
  0.6679, and the hard stack's 0.6940 is above it.
  - The codes differ in what they send. The SAE sends 32 continuous
    coefficients; each Ontologizer head sends a choice of one of its atoms,
    plus a gain. So this is FVU at the shipped recipes, not at a matched
    rate. `quantrate.py` would price both; it has not been run on these
    models.
  - One seed each.
- **Collapse is gone from the hard stack, not from the direct one.**
  - The hard stack's cossim_k_max falls from 0.68 to 0.53 (0.30 on
    `mc4_4M`).
  - The direct run's still rises, from 0.10 to 0.89 (0.32 on `mc4_4M`). Its
    KL_m is 2.05 → 2.04 at both values of `s_hm`, as on `mc4_4M`, so the
    mean-entropy weight does not move it.
  - Which head collapses, and whether it costs the fit, is not checked.
- **What this leaves open.**
  - The 2026-10-13 paired runs, which measured coordinate 837's weight,
    trained at `s_hm` 1e-4, in the regime where the MSE was a few percent
    of the loss.
  - The other auxiliary weights (`s_g`, `s_bcossim`, `s_support`, `s_L1F`)
    and Adam's ε are also fixed in absolute units and were not rescaled.

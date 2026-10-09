# lang-tags notes

## 2026-10-09: the measurement

**Setup.** 80 of the 86 mC4 configs were encoded under `eng_Latn` rather
than their intended tag: everything but en, en-multi (whose intended tag
is `eng_Latn` anyway), fr, es, de and zh, so 93.0% of the cache's rows.
NLLB's tokenizer puts the tag first (`[tag] text </s>`), so a mis-tagged
row differs from a correct one in exactly one of up to 512 input tokens.

Sample: cache rows 0–34,399, which the strict round-robin interleave
makes exactly 400 rows of each language. `texts` replayed them in 1.5 min
after ~4 min of dataset setup, and every row's language matched the
`.langs.npy` sidecar. `encode` took ~15 min on the 3090 (fp32, ~35
rows/s at the 512-token cap).

**Replay check.** Re-encoding 1,376 rows (16 per language) under the
cache's own tag reproduces the cached vectors (cosine 1.000000 to six
places), and the six configs whose tag was already right come out
identical under their intended tag. Everything below is the tag alone.

### How far the tag moved the embeddings

On the 32,000 changed rows:

| | |
|---|---|
| cosine(intended, cached): median / 5th percentile / min | 0.919 / 0.744 / 0.249 |
| whitened FVU of the cached vector as a reconstruction of the intended one | **0.305** (0.284 over all rows) |
| for scale: the reference hard code's reconstruction FVU (`ste_h76_init01`, tail) | 0.154 |
| shift energy in a common offset / per-language offsets / row-specific | 0.041 / 0.083 (chance 0.003) / 0.876 |

By the objective's own metric a cached row is about twice as far from its
correctly tagged embedding as the hard code's reconstruction is from the
cached row, and seven eighths of the shift is row-specific, so no
per-language correction recovers the intended embeddings.

Shorter texts move more (the tag is a larger share of the pool and of what
attention reads): FVU 0.50 at up to 32 tokens, 0.45 at 33–64, 0.39 at
65–128, 0.35 at 129–256, 0.28 at 257–511, 0.25 at the 512-token cap (45%
of rows). By language the shift is smallest for ru (0.11), hi (0.13),
ja-Latn (0.14), mr and hi-Latn (0.15), and largest for Latin-script
European languages: fi 0.62, et 0.60, lt 0.56, ga 0.53, hu 0.52, is 0.50,
so 0.48, sl 0.47. The five romanized configs: 0.25.

Side measurement: SONAR's pooled norm before the cache's L2 step has
median 0.303, SD 0.051 (intended tags), so the unit norm is the
pipeline's, not SONAR's.

### The distribution is not harder or easier to code

| | cached | intended |
|---|---|---|
| whitened participation ratio | 268 | 266 |
| Gaussian reference FVU at 380 / 1900 bits | 0.421 / 0.0498 | 0.424 / 0.0506 |
| whitened variance (ratio) | 1 | 1.008 |

The cached 0.0498 matches the writeup's 0.0499 on the tail.

### Language structure

Same 34,400 rows, both versions:

| | cached | intended |
|---|---|---|
| eta^2 language, raw / whitened | 0.067 / 0.056 | 0.074 / 0.058 |
| eta^2 script | 0.023 | 0.026 |
| mean cosine within / between language | 0.363 / 0.318 | 0.353 / 0.302 |
| floor (squared norm of the mean) | 0.319 | 0.303 |
| within minus between | 0.045 | 0.051 |
| 86-way ridge probe, 100 held-out rows per language | 0.700 | 0.745 |

Language is a somewhat stronger factor with correct tags (the cached
eta^2 here, 6.7%, is the writeup's 6.51% on the tail plus ~0.25 points of
sample-size bias). The change is not uniform. Probe accuracy rises for
44 of the 80 changed languages and falls for 34: up for non-Latin scripts
and romanized configs (ru 0.50 to 0.84, mr 0.37 to 0.79, hi-Latn 0.82 to
0.98), down for the Latin-script European languages that moved most (sl
0.85 to 0.35, hu 0.92 to 0.54, et 0.89 to 0.51, fi 0.94 to 0.69). Tagged
English, a Latin-script European text kept a distinctive signature; under
its own tag SONAR's translation training pulls it toward its neighbors.

The mis-tagged cache also makes confusions with no linguistic basis:
Welsh read as Greek (0.40 of Welsh rows), Marathi as Thai (0.30), Shona as
romanized Hindi (0.16). The correctly tagged embeddings' top confusions
are all between related languages (Hindi/Marathi 0.16, Xhosa/Zulu,
Macedonian/Serbian, Kyrgyz/Kazakh, Danish/Swedish, Slovak/Czech,
Javanese/Sundanese, 0.11–0.15), plus en-multi/en, which are both English.

### The shipped models on correctly tagged inputs

The sample rows are training rows; the tail column shows in-sample and
held-out FVU agree to ~0.01. FVU on the 32,000 changed rows:

| model | tail | cached | intended | change |
|---|---|---|---|---|
| `ste_h76_init01` (5×76, k=32) | 0.154 | 0.152 | 0.166 | +9.8% |
| `ste_l1_h380_i01` (flat 1×380) | 0.287 | 0.281 | 0.304 | +8.1% |
| `resid_nc` (softmax) | 0.0054 | 0.0054 | 0.0056 | +4.3% |
| SAE `m11264_k32` | 0.494 | 0.477 | 0.525 | +10.0% |
| SAE `m11264_k160` | 0.259 | 0.246 | 0.276 | +12.2% |

Per language the hard code's FVU rises by a median 9.6%, at most 26%.

Head agreement between the two inputs on changed rows, per layer, against
an isotropic random shift of each row's own size and against chance (two
unrelated rows):

| model | tag shift | same-size random shift | chance |
|---|---|---|---|
| `ste_h76_init01` | 0.514 0.291 0.198 0.144 0.109 | 0.542 0.306 0.204 0.148 0.109 | 0.032 |
| `ste_l1_h380_i01` | 0.457 | 0.463 | 0.031 |
| `resid_nc` | 0.499 0.395 0.321 0.268 0.893 | 0.507 0.365 0.259 0.211 0.888 | 0.055–0.109, 0.880 |

The SAEs keep 42% (k32) and 44% (k160) of a row's active latents
(median Jaccard). Best-head language NMI: hard code 0.186 to 0.175 (null
0.010; the writeup's 0.18), flat 0.214 to 0.195, softmax 0.061 to 0.069.

On the six unchanged configs, where the two inputs differ only by float
rounding, the hard code keeps 99.4–100% of heads but `resid_nc` keeps
only 95.3% at layer 3: its deeper softmax heads sit close enough to their
boundaries for rounding to flip them.

### What it means

- **Comparisons between models stand.** Every model trained on the cache
  loses 4–12% on correctly tagged inputs, the hard code and the SAEs by
  about the same proportion, so the orderings in the reconstruction
  tables do not depend on the tags, and the Gaussian reference is the
  same on both versions. Whether models retrained on a correct cache
  reach the same numbers is untested.
- **The data is not what the writeup calls it.** 93% of rows are SONAR
  embeddings computed under the English source tag, which differ from
  SONAR's own embeddings of those texts by more than the hard code's
  reconstruction error. Every description of the data needs to say so.
- **Language results move by 5–15%** (eta^2 6.7% to 7.4%, the cosine gap
  0.045 to 0.051, best-head NMI 0.186 to 0.175). The qualitative findings
  (language weak and residual, no head encodes it, regions rather than
  directions) hold; individual confusions do not all survive, and some of
  the cache's are artifacts of the tag.
- **Code fragility, as a side result.** A shift of whitened size 0.3
  reassigns half the hard code's layer-0 heads and almost all deeper ones,
  exactly as a same-size random shift does.

### Options

1. Keep the cache and disclose it in the data description.
2. Re-encode the eval tail (replaying the stream to row 3.9M is ~3 h at
   ~370 rows/s; encoding 65,536 rows ~30 min) and report held-out numbers
   on correctly tagged inputs beside the cached ones.
3. Re-encode the whole cache (~31 h of encoding at ~35 rows/s plus the
   replay) and retrain the headline models.

Before any re-encode, `TokenizeTransform` has to look tags up in
`MC4_TO_SONAR` instead of its 5-entry `LANG_MAP`.

## 2026-10-09: the L2 normalization (`norm.py`)

The cache also divides every embedding by its norm, which SONAR does not
do; decoding scripts rescale the unit vector to `textfid.SONAR_NORM`
(0.307). Measured on the same 34,400 rows, from the pre-normalization
norms `retag.py encode` kept (intended tags).

**What the norm is.** Median 0.303, SD 0.051 (CV 0.17); per-language
medians 0.19–0.35, language explaining 31% of its variance; correlation
+0.32 with log token count. (`textfid.py`'s comment gives per-language
medians 0.30–0.34; that sample was smaller and English-tagged.)

**What dropping it costs.** The direction at one constant norm
reconstructs SONAR's own embeddings at whitened FVU 0.036 (0.039 at
`SONAR_NORM`), a quarter of the hard code's 0.154. But the direction
predicts the norm with held-out R^2 0.965 (ridge on the unit vector), and
at the predicted norm the FVU is 0.0013: the unit vectors carry almost
everything the norm did. Language eta^2 is 0.074 on the unit vectors and
0.073 on the raw ones.

**Decoding.** 1,032 rows (12 per language), decoded as the eval scripts
decode (forced English, greedy, 48 tokens), from SONAR's own embedding
and from three substitutes. "Excess" is the cosine of the re-encoded
decode to its row's true direction minus the same to another row of its
language (the shared mean direction gives every pair about 0.09);
"change" is the paired difference from the true-norm decode:

| decoder input | chrF2 to the true-norm decode | identical text | excess | change |
|---|---|---|---|---|
| SONAR's embedding (true norm) | 1 | 1 | 0.260 | --- |
| direction at `SONAR_NORM` | 0.566 | 0.138 | 0.258 | −0.0014 ± 0.0019 |
| direction at the predicted norm | 0.785 | 0.428 | 0.260 | +0.0004 ± 0.0011 |
| the cache's vector (English tag) at `SONAR_NORM` | 0.345 | 0.016 | 0.252 | −0.0074 ± 0.0030 |

The constant rescale changes the surface text (86% of decodes differ
from the true-norm decode; greedy decoding is sensitive to scale) but not
how much of the embedding the decode keeps. The source tag costs a small
amount (about 3% of the 0.26 row-specific signal, 2.5 SE).

**Verdict.** The normalization is benign for the embedding-space work:
it discards one number per row that the direction almost determines.
What it affects is wording: the cache holds SONAR's directions, not its
embeddings, and absolute decoded text differs from what SONAR would
decode. Decoding at the norm predicted from the direction (or storing the
norm as a sidecar in any re-encode) removes most of the textual
difference; anything that feeds steered vectors to a model reading
SONAR's own scale, such as an LCM, needs one of the two.

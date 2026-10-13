# next-sentence: notes

## 2026-10-13: built and smoke-tested; baselines

**Status.** The experiment is built and smoke-tested. The full runs (two seeds of `next` and two of
`self`, `README.md`) have not started.

**Smoke test.** A 32-head code (160 bits) trained for 250 steps on the first 80,000 rows, then scored
by `eval_next.py`, on the CPU. This was a test of the pipeline, not a result.
- 64,327 adjacent pairs in those rows.
- The checkpoint loaded and scored, with FVU_w 1.30 against the next sentence and 1.31 against the
  input. A code that short and small is worse than the mean.

**Pairs.**
- In the first 4M rows of `mc4_sent16`, 3,208,887 rows (80.2%) have an adjacent successor in the same
  document.
- The held-out tail, snapped to whole documents, gives 26,335 pairs.

**How predictable is the next sentence?** FVU_w against x_{t+1} on the 26,335 held-out pairs:

| Predictor | FVU_w |
|---|---|
| Training mean | 1.0001 |
| Copy x_t | 1.5325 |
| Ridge from x_t, 20,000 training pairs from the first 80,000 rows (α 1e-3) | 0.9468 |
| Ridge from x_t, 200,000 training pairs (α 1e-3) | 0.8967 |
| Ridge from x_t, 200,000 training pairs (α 1e-1) | 0.8961 |

*Reading:*
- **Linear prediction explains about 10%.** The current sentence linearly explains about 10% of the
  next sentence's whitened variance, and the regularization strength hardly matters. The gain from 20k
  to 200k fitting pairs says the linear floor isn't saturated yet.
- **Copying is worse than the mean.** Adjacent sentences share a component, but the rest of x_t
  differs from x_{t+1} by more than x_{t+1} varies.
- **What a code can learn.** A next-sentence code can only learn the shared component, which is the
  point of the experiment. Its FVU_w should be read against the ridge floor (about 0.90), not against
  reconstruction numbers.

## 2026-10-13: head transitions and pair concentration (`transitions.py`)

**Why.** The identifiability results behind this experiment
(`scratch/literature/identifiability.md`) assume two things that can be checked before building a pair
objective:
- adjacent observations differ in a few factors, as SlowVAE measured in natural video;
- positive pairs are more concentrated than the data, which contrastive identification needs.

**Setup.**
- Rows: the held-out tail of `mc4_sent16`, the last 32,768 rows (2,850 documents, 86 languages).
- Pair types:

  | Pair type | Pairs |
  |---|---|
  | adjacent (`d1`) | 26,333 |
  | adjacent, both full sentences (`d1_sent`) | 12,870 |
  | same document, 2 / 4 / 8 / 12 segments apart | 24,347 / 20,350 / 13,174 / 7,043 |
  | another document in the same language (`lang`), per `d1` anchor | 26,333 |
  | another document (`random`), per `d1` anchor | 26,333 |

  51% of the adjacent pairs involve a line or a cap piece.
- Command: `transitions.py --ckpt` with the three models below. Output is in
  `data/out/sonar/next-sentence/transitions/`.

### Pair concentration (no model)

| Pairs | Whitened squared distance ÷ random | Cosine | Log-norm correlation |
|---|---|---|---|
| `d1` | 0.769 | 0.303 | 0.468 |
| `d1_sent` | 0.803 | 0.264 | 0.427 |
| `d2` | 0.785 | 0.288 | 0.438 |
| `d4` | 0.809 | 0.265 | 0.418 |
| `d8` | 0.822 | 0.245 | 0.392 |
| `d12` | 0.815 | 0.243 | 0.394 |
| `lang` | 0.959 | 0.114 | 0.214 |
| `random` | 1 | 0.068 | 0.004 |

**AUC of each similarity, separating a pair type from random / same-language pairs:**

| Similarity | `d1` | `d1_sent` | `d12` |
|---|---|---|---|
| raw cosine | 0.923 / 0.846 | 0.919 / 0.834 | 0.870 / 0.767 |
| whitened cosine | 0.860 / 0.790 | 0.847 / 0.770 | 0.802 / 0.720 |
| raw squared distance | 0.698 / 0.660 | 0.681 / 0.641 | 0.661 / 0.622 |
| whitened squared distance | 0.685 / 0.649 | 0.666 / 0.627 | 0.651 / 0.614 |

**Is a sentence's successor the nearest of it and 255 negatives?** Chance is 0.4%.

| Similarity | Random negatives: top-1, median rank | Same-language negatives: top-1, median rank |
|---|---|---|
| cosine | 53.0%, 1 | 40.3%, 4 |
| whitened distance | 22.2%, 32 | 20.7%, 41 |

### Head transitions

κ is Cohen's kappa between the entries a head assigns to the two sentences, averaged over heads: 0 is
chance and 1 is always the same entry. The last columns give the share of heads whose entry changes.

| Model (step; heads, k) | κ `d1` | κ `d2` | κ `d8` | κ `d12` | κ `d1_sent` | κ `lang` | max κ `d1` | heads changing, `d1` / `random` |
|---|---|---|---|---|---|---|---|---|
| `ste_h76_init01_no837` (50,000; 5×76, 32) | 0.048 | 0.046 | 0.032 | 0.033 | 0.033 | 0.004 | 0.081 | 92.2% / 96.9% |
| `ste_k128_h54_direct_no837` (50,000; 5×54, 128) | 0.020 | 0.021 | 0.013 | 0.015 | 0.012 | 0.001 | 0.041 | 97.2% / 99.2% |
| `ste_k128_h54_direct_no837_e2` (344,250; 5×54, 128) | 0.011 | 0.012 | 0.006 | 0.009 | 0.005 | 0.000 | 0.013 | 98.1% / 99.2% |

- κ on `random` pairs is 0.000 for every model, as constructed.
- Every head has uniform entry usage: 5.00 bits for k = 32, and 6.95–6.99 bits for k = 128.

### Readings

- **Adjacent sentences share a lot, and most of it is document-level.**
  - Cosine separates adjacent from random pairs with AUC 0.92.
  - The successor is the nearest of 256 sentences 53% of the time, against 0.4% chance.
  - Sentences 12 apart keep 74% of the adjacent pair's cosine above random: (0.243 − 0.068) /
    (0.303 − 0.068).
  - Language accounts for 20%: (0.114 − 0.068) / 0.235.

  So contrastive identification would not fail for lack of concentration, as long as it compares
  *directions*.
- **The norm hides it from the training metric.**
  - Under whitened squared distance the same separation is AUC 0.69 and top-1 22%.
  - The decomposition shows the norm costs most of that: cosine 0.92 → raw distance 0.70. Whitening
    costs less: cosine 0.92 → whitened cosine 0.86.
  - Sentence norms vary (CV 0.21), and adjacent log norms correlate only 0.47.
  - For this experiment, part of the MSE target is x_{t+1}'s norm, whose log correlates only 0.47
    with x_t's. Gain-shape scales the output by ‖x_t‖, so that assumption (README) costs error.
- **Head transitions are dense.**
  - Between adjacent sentences the hard stack keeps 7.8% of its 380 heads (about 30), against 3.1%
    (about 12) for random pairs. 92% of heads change entry.
  - No head is slow: the largest per-head κ is 0.081.
  - Agreement is spread thinly over all heads rather than concentrated in a few that persist. The
    document-level content above is real, but no head carries it as a variable.
  - In SlowVAE's terms, the slow and fast factors are mixed into every head.
- **Longer training makes it weaker.**
  - For the direct stack, κ on `d1` falls from 0.020 at 50,000 steps to 0.011 at 344,250.
  - Layer 0's lead (0.034 against 0.017 above it at 50,000 steps) is gone by then (0.011 against
    0.010).
  - The hard stack, at 50,000 steps, has the most persistent heads, again led by layer 0 (0.054
    against 0.046).
- **No language heads on sentences.**
  - Per-head NMI with the language label is at most 0.083 (median 0.029 in the hard stack).
  - κ on `lang` is at most 0.004, and same-language sentences of other documents share almost no
    entries.
  - Across heads, κ on `d1` rises with NMI(language) (Spearman 0.87 and 0.88 for the two 50,000-step
    models; 0.34 for e2). The heads with more document-level information carry a little of
    everything.
- **Consequences for the pair objectives** (literature note, item 3):
  - A sparse-transition prior would have to *create* slow heads, not select existing ones.
  - Any pair similarity should be a cosine or a head agreement, not the whitened distance.
  - Whether slow heads reproduce across seeds can't be asked of these models: they have no slow
    heads.
  - Half the adjacent pairs involve a line or a cap piece. `d1_sent` is the cleaner set, and it is
    less concentrated (cosine 0.264 against 0.303).

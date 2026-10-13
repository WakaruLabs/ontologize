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

## 2026-10-13: pilot sweep of the pair objective (`pilot.sh`): the slow heads collapse

**Setup.**
- Model: the flat model (1 × 380 heads, k = 32) with the shipped STE recipe, under the masked
  weights.
- Slow heads: the first 38.
- Pairs: same-document pairs at segment distance ≥ 2, from the first 1.3M rows (1.28M pairs).
- Training: one epoch, 5,011 steps of 256 pairs, one seed.
- Evaluation: on the held-out tail, with `eval_next.py` (FVU_w of each sentence's reconstruction)
  and `transitions.py --slow-heads 38`.

**Run history.**
- The first launch died at step 553 of the control: another job on the shared card took memory
  mid-step. `pilot.sh` now reserves a pool (BFC allocator, 35% of the card).
- The 1e-5 arm was interrupted at step 3,250 when the session ended. It resumed from that checkpoint.

**Calibration** (`train_pair.py --calibrate 8`, at initialization). The pair term's gradient norm is
1.05, all of it on the classifier. The stock loss's is 1.7e-4 on the classifier (5.6e-4 overall). So
the two pull equally at s_pair ≈ 1.6e-4, and the grid brackets that.

| s_pair | FVU_w (self) | κ `d1` | κ `d2` | κ `d8` | κ `d12` | κ `lang` | Entry entropy (bits) | NMI(lang) | Training switch share |
|---|---|---|---|---|---|---|---|---|---|
| 0 (control) | 0.6618 | 0.072 | 0.068 | 0.049 | 0.050 | 0.008 | 5.00 | 0.047 | 0.924 |
| 1e-5 | 0.6617 | 0.077 | 0.073 | 0.053 | 0.054 | 0.010 | 5.00 | 0.051 | 0.920 |
| 3e-5 | 0.6614 | 0.086 | 0.082 | 0.060 | 0.061 | 0.013 | 4.99 | 0.059 | 0.913 |
| 1e-4 | 0.6684 | 0.056 | 0.047 | 0.049 | 0.048 | 0.006 | **0.09** | 0.007 | 0.201 |
| 3e-4 | 0.6674 | 0.067 | 0.051 | 0.066 | 0.061 | 0.003 | **0.04** | 0.004 | 0.194 |
| 1e-3 | 0.6672 | 0.065 | 0.052 | 0.064 | 0.063 | 0.003 | **0.04** | 0.004 | 0.194 |
| 3e-3 | 0.6672 | 0.066 | 0.054 | 0.065 | 0.064 | 0.003 | **0.04** | 0.003 | 0.194 |

- **Columns.**
  - The κ, entropy and NMI columns are means over the 38 slow heads, on held-out pairs.
  - The training switch share is the last 250 steps of `loss.csv` column 2.
- **The other 342 heads** read κ `d1` 0.070, κ `d12` 0.049 and 5.00 bits in every arm.
- **Unrelated to the pair term:** FVU_w is high (0.66) because 5,000 steps is short. The shipped flat
  arm reaches 0.275 after 369,500 steps.

**Readings.**
- **At s_pair ≥ 1e-4 the slow heads collapse.**
  - Each one settles on a single entry: 0.04–0.09 bits held out, out of 5.
  - That satisfies the agreement term trivially. The training switch share falls to the floor that
    winner dropout sets (about 0.18), with no document structure at all: κ stays at the control's
    level, and NMI with language falls to 0.004.
  - The collapse is gradual. At 1e-4 the training share falls 0.82 → 0.64 → 0.39 → 0.20 at steps
    0, 800, 1,600 and 5,000.
  - The batch usage statistic `KL_m` rises tenfold over the control's (0.45 against 0.047), but
    `s_Hm` (1e-4) is far too weak to resist.
  - The losing heads cost 1% of FVU_w (0.668 against 0.662).
  - The equal-pull weight from calibration (1.6e-4) lies above the collapse threshold.
- **Below the threshold, the slow heads slow a little and evenly.**
  - At 3e-5, κ `d1` rises from 0.072 to 0.086 and κ `d12` from 0.050 to 0.061, at no cost in FVU_w.
  - No slow head passes κ 0.5 in any arm, so nothing like a document-level head forms.
- **The agreement term alone is the wrong objective.** "Same entry for both sentences" has a
  solution that ignores the input, and it is the one gradient descent finds. The sketch flagged this
  risk. Every arm fails the sketch's selection criterion (slow-head κ on `d2` and beyond above 0.5,
  with FVU_w within a few percent of the control).
- **Next.** The term has to reward agreement *given* uniform usage of the slow heads.
  - Either: agreement minus each slow head's batch marginal entropy, a per-head lower bound on the
    temporal mutual information. With 512 rows over 32 entries its plug-in bias is about 0.03 nats.
  - Or: the mutual-information term of `pair_objective.md`, with its larger estimation bias.
  - Both make the constant head score worst rather than best.

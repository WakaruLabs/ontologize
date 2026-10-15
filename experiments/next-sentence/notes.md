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
  mid-step. The sweep was then rerun with a reserved pool (BFC allocator, 35% of the card).
- The 1e-5 arm was interrupted at step 3,250 when the session ended. It resumed from that checkpoint.
- **The reserved pool was a mistake on a shared card.** The Fano sweep's pools, its evaluations, and
  then `long.sh` (which held 12.3 GB) ended the `sent16-full` training run (the hard stack on
  `mc4_sent16`) 12 times from 20:04, mostly on a refused allocation of about 970 MiB. The other
  session running it reported this.
  - `long.sh` was stopped at control step 6,750 to let that run finish.
  - Neither script reserves a pool any more.
  - `eval_next.py` now sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing jax; otherwise
    JAX reserves 75% of the card at its first touch.

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

## 2026-10-13: the same sweep with the Fano term (`pilot.sh fano`): no collapse, slow progress

**The term.** `pair_term="fano"` subtracts each slow head's lower bound on the mutual information
between the entries it gives the two sentences, as a fraction of log k:

  H(a) − H_b(s) − s·log(k−1)

- H(a) is the head's batch usage entropy, and s is its switch share. Fano's inequality bounds
  H(a | a′) by the last two terms.
- A collapsed head scores 0, and so does one whose entries are independent across the pair.
- With winner dropout, a perfectly slow head that uses every entry evenly caps at about 0.69.

**Setup.**
- Calibration: the term's classifier gradient is 0.91 at initialization, so it pulls level with the
  stock loss at s_pair ≈ 1.9e-4.
- Everything else is as in the agreement sweep. The control is the same run.

| s_pair | FVU_w (self) | κ `d1` | κ `d2` | κ `d8` | κ `d12` | κ `lang` | Largest κ `d1` | Entry entropy (bits) | NMI(lang) | Training bound |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 (control) | 0.6618 | 0.072 | 0.068 | 0.049 | 0.050 | 0.008 | 0.077 | 5.00 | 0.047 | — |
| 1e-5 | 0.6618 | 0.073 | 0.069 | 0.050 | 0.051 | 0.009 | 0.079 | 5.00 | 0.048 | −0.002 |
| 3e-5 | 0.6619 | 0.076 | 0.072 | 0.052 | 0.053 | 0.010 | 0.081 | 5.00 | 0.050 | −0.001 |
| 1e-4 | 0.6619 | 0.082 | 0.077 | 0.057 | 0.058 | 0.012 | 0.088 | 5.00 | 0.056 | 0.000 |
| 3e-4 | 0.6619 | 0.088 | 0.084 | 0.062 | 0.063 | 0.014 | 0.098 | 5.00 | 0.062 | 0.002 |
| 1e-3 | 0.6619 | 0.094 | 0.089 | 0.067 | 0.067 | 0.016 | 0.108 | 5.00 | 0.067 | 0.003 |
| 3e-3 | 0.6619 | 0.096 | 0.091 | 0.069 | 0.069 | 0.017 | 0.111 | 5.00 | 0.069 | 0.003 |
| 1e-2 | 0.6620 | 0.097 | 0.092 | 0.069 | 0.070 | 0.017 | 0.113 | 5.00 | 0.069 | 0.003 |

- **Columns.**
  - Means are over the 38 slow heads, held out, at step 5,000.
  - "Largest κ `d1`" is the slowest single slow head.
  - The training bound is the last 250 steps of column 2.
- **The other 342 heads** read κ `d1` 0.070, κ `d12` 0.049 and 5.00 bits in every arm.

**The trajectory** (`transitions.py --step 2750`, in `transitions_fano_step2750/`). Slow-head κ
`d1` at steps 2,750 and 5,000:

| Arm | Step 2,750 | Step 5,000 |
|---|---|---|
| control | 0.072 | 0.072 |
| s_pair 1e-3 | 0.082 | 0.094 |
| s_pair 1e-2 | 0.083 | 0.097 |

κ `d12` for the 1e-3 arm went from 0.058 to 0.067.

**Readings.**
- **The term does what the agreement term didn't.** No slow head collapses: 5.00 bits at every
  weight. Reconstruction is unaffected (FVU_w within 0.0002 of the control), and the slow heads become
  steadily slower at every distance.
  - The gain is document-level, not local: κ `d12` rises about as much as κ `d1` (+0.020 against
    +0.025 at 1e-2).
  - Slow heads also share more with same-language sentences of other documents (κ `lang` 0.008 →
    0.017) and carry more language information (NMI 0.047 → 0.069).
- **The effect is small, and the step size limits it, not the weight.**
  - At 5,000 steps the slowest head reaches κ 0.11, far from the 0.5 the sketch asks for. The
    training bound is about 0.003 of its 0.69 ceiling.
  - Above 1e-3 the weight hardly matters (κ `d1` 0.094, 0.096, 0.097), and the 1e-3 and 1e-2 arms
    match at both checkpoints. Once the pair term dominates a classifier parameter's gradient, Adam
    sets the step by the learning rate (1e-5), not the weight.
  - κ is still rising roughly linearly, by about +0.005 per 1,000 steps at 1e-3. Collapse took about
    3,000 steps under the agreement term: an easy direction. Reorganizing a head's partition around
    the document is a slow one.
- **Next.**
  - A longer arm at s_pair 1e-3 against the control, over the full 4M-row pair set (one epoch is
    about 15,400 steps), with κ at several checkpoints to see whether the rise continues or levels.
  - If it is too slow, a learning-rate multiplier for the slow heads' classifier rows, which the
    pair term reaches alone.

## 2026-10-14: the long run (`long.sh`): the slow heads keep slowing, faster

`long.sh` trains the control and the Fano arm at s_pair 1e-3 for one epoch over all 3,947,664 pairs
of the first 4M rows (15,420 steps), keeping a checkpoint every 2,500 steps. Both finished at step
15,420; the last checkpoint is 15,250.

Slow-head means (38 heads), held out, at each kept checkpoint. The control's slow heads read κ `d1`
0.071–0.072 throughout, κ `d12` 0.049–0.050 and NMI(lang) 0.046–0.047. The other 342 heads of both
runs read κ `d1` 0.069–0.071 throughout.

| Step | κ `d1` | κ `d2` | κ `d8` | κ `d12` | κ `lang` | Largest κ `d1` | Entry entropy (bits) | NMI(lang) | Training bound |
|---|---|---|---|---|---|---|---|---|---|
| 2,500 | 0.081 | 0.076 | 0.056 | 0.057 | 0.011 | 0.089 | 5.00 | 0.055 | −0.0004 |
| 5,000 | 0.094 | 0.089 | 0.067 | 0.068 | 0.016 | 0.111 | 5.00 | 0.067 | 0.0028 |
| 7,500 | 0.113 | 0.108 | 0.084 | 0.083 | 0.025 | 0.142 | 4.99 | 0.083 | 0.0076 |
| 10,000 | 0.140 | 0.134 | 0.107 | 0.106 | 0.038 | 0.183 | 4.99 | 0.105 | 0.0147 |
| 12,500 | 0.171 | 0.164 | 0.134 | 0.133 | 0.054 | 0.222 | 4.98 | 0.129 | 0.0248 |
| 15,000 | 0.204 | 0.195 | 0.165 | 0.162 | 0.072 | 0.259 | 4.97 | 0.152 | 0.0353 |

The training bound is column 2 of `loss.csv`, a mean over the 250 steps before each checkpoint.

Held-out FVU_w at step 15,250 (`eval_e1.json`): 0.5202 for the control, 0.5232 for the Fano arm.

**Readings.**
- **The rise doesn't level off; it speeds up.**
  - Slow-head κ `d1` gains +0.013, +0.019, +0.027, +0.031 and +0.033 per 2,500 steps. At the
    epoch's end it is about three times the control's (0.204 against 0.071).
  - The training bound accelerates with it.
  - Entry usage stays near 5 bits, so this is not collapse.
  - The pilot's 5,000 steps had seen only the start.
- **It is document-level.**
  - κ `d12` follows κ `d1` (0.162 against 0.204 at step 15,000), so what the slow heads keep
    persists across a document rather than between neighbours.
  - Part of it is language: κ on same-language sentences of other documents rises from 0.008 to
    0.072, and NMI with the language label from 0.047 to 0.152.
  - The rest, κ `d12` minus κ `lang` (about 0.09 at 15,000), is shared within documents beyond their
    language.
- **Not there yet.**
  - No slow head reaches κ 0.5. The slowest is at 0.26, and the mean at 0.20.
  - At the last interval's rate (+0.033 per 2,500 steps), the mean would reach 0.5 after about 22,000
    more steps, about 1.4 epochs. That assumes the rate holds, which it hasn't so far: it has risen
    every interval.
- **Cost.** 0.6% of FVU_w (0.5232 against 0.5202).
- **Next.**
  - Continue both runs for more epochs, keeping checkpoints.
  - Once the slow heads are substantially slow, train a second seed of each arm. The block-level
    question (do seed A's slow heads carry what seed B's do?) then has something to measure.
  - If the time is too long, raise the learning rate on the slow heads' classifier rows.

**Interruptions.** These are the control's, since its numbers depend on them. The Fano arm ran
uninterrupted, from about 16:25 to 16:47 (last checkpoint 16:46:40), and the evaluations ran
16:47–16:49.

- **Held.** It was stopped at step 6,750 on 2026-10-13 so `sent16-full` could finish, and held until
  that run and `sent16-sae` had both completed (09:13 and 11:23 on 2026-10-14).
- **Restarted at 16:04 on demand, with no cap.** It grew to 9.1 GB beside another user's training
  benchmark, which reached 12.4 GB and left about 1.5 GB of the card free. That job ended at about
  16:06. Whether it ran out of memory is unknown; it restarted a minute later with a shorter
  sequence length.
- **Two instances for about 40 seconds.** An attempt to stop the run for a capped restart missed its
  process group, so a second instance started beside the first.
  - Both wrote checkpoints and `loss.csv` rows: the file reached 9,250 rows against a latest
    checkpoint of 9,000.
  - Both were stopped. Checkpoints 8,750 and 9,000, either of which the two could have written, were
    moved to `long/doc2_sp0_ph38.discarded_8750_9000/`.
  - The run resumed from 8,500, written before the second instance started. The resume truncated
    `loss.csv` to 8,500 rows.
- **From 16:14, one instance to the end.** It ran under the BFC allocator without preallocation,
  capped at 35% of the card (9.1 GB with CUDA's own). The launch allowed up to six retries, but the
  first attempt finished both runs. The control's last checkpoint is at 16:24.

## 2026-10-14: three epochs (`long.sh 3`): the slow heads reach κ 0.46, but the rise is slowing

`long.sh 3` continued both runs from their epoch-one checkpoints (15,250) to three epochs, step
46,261. The last checkpoint is 46,250.
- **Runs.** Both ran uninterrupted, one at a time: the control 17:48–18:31, then the Fano arm
  18:31–19:15.
- **Allocator.** As in epoch one's last stretch: BFC without preallocation, capped at 35% of the card.

Slow-head means (38 heads), held out, at every kept checkpoint of the Fano arm:

| Step | κ `d1` | κ `d2` | κ `d8` | κ `d12` | κ `lang` | Largest κ `d1` | Entry entropy (bits) | NMI(lang) | Training bound |
|---|---|---|---|---|---|---|---|---|---|
| 2,500 | 0.081 | 0.076 | 0.056 | 0.057 | 0.011 | 0.089 | 5.00 | 0.055 | −0.0004 |
| 5,000 | 0.094 | 0.089 | 0.067 | 0.068 | 0.016 | 0.111 | 5.00 | 0.067 | 0.0028 |
| 7,500 | 0.113 | 0.108 | 0.084 | 0.083 | 0.025 | 0.142 | 4.99 | 0.083 | 0.0076 |
| 10,000 | 0.140 | 0.134 | 0.107 | 0.106 | 0.038 | 0.183 | 4.99 | 0.105 | 0.0147 |
| 12,500 | 0.171 | 0.164 | 0.134 | 0.133 | 0.054 | 0.222 | 4.98 | 0.129 | 0.0248 |
| 15,000 | 0.204 | 0.195 | 0.165 | 0.162 | 0.072 | 0.259 | 4.97 | 0.152 | 0.0353 |
| 17,500 | 0.236 | 0.228 | 0.195 | 0.191 | 0.091 | 0.290 | 4.95 | 0.174 | 0.0480 |
| 20,000 | 0.268 | 0.258 | 0.224 | 0.220 | 0.109 | 0.321 | 4.94 | 0.195 | 0.0616 |
| 22,500 | 0.297 | 0.287 | 0.252 | 0.248 | 0.126 | 0.349 | 4.93 | 0.214 | 0.0736 |
| 25,000 | 0.323 | 0.313 | 0.278 | 0.273 | 0.142 | 0.371 | 4.91 | 0.230 | 0.0861 |
| 27,500 | 0.347 | 0.337 | 0.301 | 0.296 | 0.156 | 0.396 | 4.90 | 0.246 | 0.0973 |
| 30,000 | 0.369 | 0.358 | 0.322 | 0.316 | 0.169 | 0.413 | 4.89 | 0.260 | 0.1093 |
| 32,500 | 0.388 | 0.378 | 0.342 | 0.335 | 0.182 | 0.429 | 4.88 | 0.272 | 0.1206 |
| 35,000 | 0.406 | 0.395 | 0.359 | 0.351 | 0.193 | 0.443 | 4.88 | 0.284 | 0.1298 |
| 37,500 | 0.422 | 0.411 | 0.375 | 0.368 | 0.204 | 0.455 | 4.87 | 0.295 | 0.1380 |
| 40,000 | 0.437 | 0.425 | 0.389 | 0.382 | 0.214 | 0.466 | 4.86 | 0.304 | 0.1480 |
| 42,500 | 0.450 | 0.438 | 0.402 | 0.394 | 0.223 | 0.473 | 4.86 | 0.313 | 0.1538 |
| 45,000 | 0.463 | 0.451 | 0.415 | 0.407 | 0.231 | 0.487 | 4.85 | 0.322 | 0.1617 |

- **Columns.** As in the epoch-one table. The training bound averages the 250 steps before each
  checkpoint.
- **The control** is unchanged. Its slow heads read throughout:
  - κ `d1` 0.067–0.072;
  - κ `d12` 0.047–0.050;
  - NMI(lang) 0.044–0.047;
  - 4.99–5.00 bits.
- **The other 342 heads** of both runs read κ `d1` 0.066–0.071 throughout. None exceeds 0.08.

Held-out FVU_w at step 46,250 (`eval_e3.json`, 26,335 pairs):

| Run | Against the input | Against the next sentence |
|---|---|---|
| control | 0.4593 | 1.2136 |
| Fano arm | 0.4679 | 1.1951 |

The next-sentence baselines are unchanged: the training mean 1.0001, copying x_t 1.5327, ridge 0.8967.

**Readings.**
- **The rise peaked around the turn of the first epoch and is slowing.**
  - Slow-head κ `d1` gained +0.029 to +0.033 per 2,500 steps from 10,000 to 22,500. After that it
    gained less each interval, down to +0.013 by 45,000.
  - At step 45,000 the mean is 0.463. The slow heads are bunched: from 0.423 to 0.487, where at
    15,000 they spread from 0.163 to 0.259.
  - No head has reached 0.5. If the increments keep shrinking by about 0.0015 per interval, the mean
    crosses 0.5 in about four more intervals (10,000 steps). It may instead level off just below.
- **It is still document-level, and language is about half of it.**
  - κ `d12` is 0.407 against κ `d1`'s 0.463, so what the slow heads keep persists across a document.
  - κ `lang` (same-language sentences of other documents) is 0.231, half of κ `d1`. NMI with the
    language label is 0.322, against the control's 0.047.
  - The rest, κ `d12` minus κ `lang`, is shared within a document beyond its language. It is about
    0.18, twice its value at epoch one (0.09).
- **No collapse.** Entry usage drifts from 5.00 to 4.85 bits.
- **The cost grows.**
  - FVU_w against the input is 0.4679, against the control's 0.4593: 1.9% worse, up from 0.6% at
    epoch one.
  - Both runs improved by about 0.06 over the two added epochs.
  - The Fano arm's reconstructions are slightly closer to the next sentence (1.195 against 1.214).
    Both stay above the training mean's 1.000, since they reconstruct the current sentence.
- **The training bound** is 0.162, against the 0.69 a perfectly slow head reaches under winner
  dropout.
- **Next.**
  - The slow heads are now substantially slow (κ 0.46 against the control's 0.07). Train a second
    seed of each arm, and measure whether seed B's slow heads carry what seed A's do: the
    block-level reproducibility question.
  - One more epoch would show whether κ levels off below 0.5.

## 2026-10-15: seed 43: the slow heads reproduce across seeds

`long.sh 1 43` then `long.sh 3 43` trained a second seed of both arms. That is seed 42's history: one
epoch, then continued to three, with checkpoints every 2,500 steps.
- **What the seed changes:** the initialization and the draw of each row's same-document partner.
- **What it doesn't:** the rows themselves (the same 3,947,664) and their order, since the loader's
  shuffle seed is fixed at 42.
- **Runs.** Uninterrupted, 05:21–07:33.

**The trajectory replicates.** Slow-head means, held out (`transitions_s43_*`):

| Step | κ `d1`, seed 42 | κ `d1`, seed 43 | Seed 43: κ `d12` | κ `lang` | Slow heads' κ `d1` range | Entry entropy (bits) | NMI(lang) |
|---|---|---|---|---|---|---|---|
| 2,500 | 0.081 | 0.080 | 0.056 | 0.011 | 0.071–0.086 | 5.00 | 0.055 |
| 5,000 | 0.094 | 0.093 | 0.066 | 0.017 | 0.085–0.104 | 5.00 | 0.067 |
| 10,000 | 0.140 | 0.139 | 0.105 | 0.037 | 0.117–0.162 | 4.99 | 0.106 |
| 15,000 | 0.204 | 0.203 | 0.162 | 0.071 | 0.173–0.240 | 4.97 | 0.152 |
| 20,000 | 0.268 | 0.267 | 0.219 | 0.107 | 0.230–0.308 | 4.94 | 0.194 |
| 25,000 | 0.323 | 0.322 | 0.271 | 0.139 | 0.283–0.359 | 4.91 | 0.229 |
| 30,000 | 0.369 | 0.365 | 0.312 | 0.164 | 0.332–0.402 | 4.89 | 0.257 |
| 35,000 | 0.406 | 0.401 | 0.346 | 0.187 | 0.368–0.436 | 4.88 | 0.281 |
| 40,000 | 0.437 | 0.432 | 0.376 | 0.207 | 0.396–0.465 | 4.86 | 0.301 |
| 45,000 | 0.463 | 0.457 | 0.402 | 0.224 | 0.419–0.491 | 4.85 | 0.318 |

- Seed 43's control: slow heads at κ `d1` 0.066–0.070 throughout, and the other heads of both seed-43
  runs at 0.066–0.071.
- Held-out FVU_w at step 46,250 (`eval_s43_e3.json`): 0.4668 for the Fano arm against 0.4586 for the
  control, 1.8% worse. Seed 42 read 0.4679 and 0.4593.

**Do the two seeds' heads learn the same partitions?** This is `seedmatch.py`.
- **Method.** It extends `experiments/ste-arm/partition.py`, which matches heads one-to-one by the NMI
  between their partitions of the held-out rows (Hungarian), against a null that shuffles one model's
  rows. Here the match is split by block. Step 46,250, the cache's last 32,768 rows:

| Arm | Block | Heads | Matched NMI (median; max) | Null |
|---|---|---|---|---|
| Fano | slow | 38 | 0.414 (0.411; 0.458) | 0.005 |
| Fano | rest | 342 | 0.017 (0.017; 0.022) | 0.005 |
| Fano | all | 380 | 0.056 (0.017; 0.458) | 0.005 |
| control | heads 0–37 | 38 | 0.016 (0.016; 0.022) | 0.005 |
| control | rest | 342 | 0.017 (0.017; 0.023) | 0.005 |

- **The whole-model assignment.** In the Fano arm it pairs **all 38** of seed 42's slow heads with seed
  43's slow heads (chance 10%). In the control it pairs 7.9% of heads 0–37 with heads 0–37.
- **Language.**
  - **Method.** The slow heads carry language, so two seeds could agree only because both track it.
    `--labels mc4_sent16.langs.npy` scores the matched slow pairs inside each language and averages,
    weighted by rows. The null shuffles one seed's codes within each language.
  - **Rows.** NMI over 32 entries is biased upward on a few hundred rows, and the 32,768-row tail
    gives about 390 per language. So this uses the last 262,144 rows: 84 languages with at least
    1,000 rows, 260,453 rows in all. Every row past the 4M-row training prefix is held out.
  - **Fano slow pairs:** within-language NMI 0.313 (median 0.311), against a null of 0.041.
  - **Control:** 0.069 against 0.051.

**Readings.**
- **The slow block reproduces across seeds.** Its heads match at NMI 0.41, about 25 times the 0.017 of
  every other head and of the control's heads. Every slow head finds its counterpart in the other
  seed's slow block.
- **Nothing else changes.** The 342 other heads reproduce no better than the control's. 0.017 is the
  same as the flat arm's 0.018 on mC4 documents (`experiments/ste-arm`), the result that motivated
  this experiment.
- **Most of it is not language.** Within a language, the slow pairs still agree at 0.31 against a
  0.04 null, while the control's excess over its null is 0.02. So the two seeds agree on how sentences
  are partitioned inside a language: the within-document, cross-sentence factors the Fano term
  rewards.
- **Caveats.**
  - One seed pair.
  - The two seeds share the row order. Still, the control shares it as well and does not reproduce.
  - The cost is 1.8% of FVU_w.
- **Next.**
  - What do the slow heads encode beyond language? Enrichment against document-level labels where we
    have them; otherwise autointerp of the slow cells.
  - A third seed.
  - Whether the result holds with a larger slow block, or at matched cost.

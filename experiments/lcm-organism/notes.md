# lcm-organism: notes

## 2026-11-20: Phase 2 on a TPU: codes on the LCM's states agree across seeds at over twice SONAR's level, and the spectrum doesn't explain it

**In short.**
- **The plan's claim holds.** Codes on LCM block 4's states match across seeds well above codes on
  SONAR or on the untrained LCM: 2.4 times SONAR's excess over the null on all heads, and 4.2 times
  within a language. A third seed agrees with the first two at the same level.
- **It is not effective dimension, and not context averaging.** SONAR recoloured to the LCM's
  spectrum closes 76% of the gap in FVU_w, but only 22% of the gap in agreement, and 10% within a
  language. The untrained LCM mixes in context, and agrees no more than SONAR.
- **It is still not heads identified one by one.** Each head's best partner beats its runner-up by
  at most 0.001, and the matched NMI, 0.034, is far below the pair objective's 0.41.

**Runs.** Every arm is `train_pair.py --epochs 3 --pair-term agree --s-pair 0`, the SONAR control's
recipe (`doc2_sp0_ph38`), to step 46,250, keeping only the last checkpoint.
- **Where.** A v5e-4 under Flex-start through `experiments/cloud`, one run per chip, at matmul
  precision `high`. Each run took about 1 h 57 min, against the plan's estimate of about an hour.
- **Scoring,** at `highest`: `seedmatch.py --labels <harvest>.langs.npy --label-rows 262144
  --min-rows 1000`, with `--c` for three seeds, then `eval_next.py`. The queues are in `queues/` and
  the outputs in `data/out/lcm-organism/phase2_tpu/`.
- **The inputs.**
  - **LCM:** `harvest_lcm.py` on Phase 1's seed 42 (step 7,216) at block 4: the rows below 4,400,014
    and the tail, at float16, with matched MSE weights (`lcm_b4`).
  - **Untrained:** the same harvest of that run's step 0 (`init_b4`).
  - **SONAR:** `mc4_sent16_f16.npy` with the control's `no837` weights.
  - **Recoloured SONAR** (`recolour.py`, below), added after the spectrum check.
- **Interruptions.** LCM seed 44 and both recoloured runs died together when the VM's RAM disk filled,
  and resumed from their last checkpoints: step 7,000 for seed 44, and within the first 500 steps for
  the others.

**The readouts.**

| Arm | Seeds | All 380 heads: matched NMI (null) | First 38 heads, within language (null) | Best over runner-up | FVU_w, each seed |
|---|---|---|---|---|---|
| LCM, block 4 | 42, 43, 44 | **0.0340** (0.0048) | **0.1269** (0.0495) | 0.0010 | 0.2820 / 0.2812 / 0.2798 |
| SONAR recoloured to the LCM's spectrum | 42, 43 | 0.0206 (0.0048) | 0.0751 (0.0508) | 0.0006 | 0.3241 / 0.3236 |
| SONAR, TPU | 42, 43, 44 | 0.0169 (0.0048) | 0.0692 (0.0507) | 0.0004 | 0.4593 / 0.4585 / 0.4597 |
| SONAR, GPU (`long.sh`) | 42, 43, 44 | 0.0168 (0.0048) | 0.0692 (0.0507) | 0.0004 | 0.4593 / 0.4586 / 0.4596 |
| Untrained LCM, block 4 | 42, 43 | 0.0155 (0.0048) | 0.0749 (0.0509) | 0.0003 | 0.4370 / 0.4370 |

How to read the columns:
- **The matched NMI** is between seeds 42 and 43. The null shuffles B's rows.
- **Within language** scores the first 38 heads' matched pairs inside each language's rows, against a
  null that shuffles within each language (84 languages, 260,453 rows). At `--s-pair 0` nothing
  distinguishes those 38 heads from the rest.
- **Best over runner-up** is the median, over those heads, of the best partner's NMI minus the second
  best's.
- **FVU_w** is held-out, against the input, under each arm's own weights.

**The TPU retrain matches the GPU runs** within 0.0001 on every NMI and FVU_w, seed by seed, so the
arms compare across the two machines. Only the loop count below differs, by one head.

**Against the plan's predictions.**
- **The untrained arm** was predicted to read near SONAR. It reads 0.0155, slightly below, so mixing in
  context at init does not make heads agree.
- **The LCM arm** was predicted to read above both, and it does. Its excess over the null is 0.0292,
  against SONAR's 0.0120 and the init's 0.0107.
- **Language.** Within a language, the LCM's excess is 0.077 against SONAR's 0.019. So the lead is not
  language: it grows once language is held fixed.
- **Block-level.** The margin was predicted near 0.003. It reads 0.0010.
- **What would have counted against,** the three arms reading alike, did not happen.
- **Seed 44,** the plan's third seed if the result was positive, matches seeds 42 and 43 at the same
  level: 0.0341 and 0.0324 on the first 38 heads, against 0.0326 between 42 and 43.
  - Its matches close a loop (42→43→44 lands where 42→44 does) for 3 of the 38 heads, against 1 by
    chance.
  - SONAR's three seeds close 1 on the TPU and 2 on the GPU.
  - These counts are too small to read much from.

**Still not heads identified one by one.** On the first 38 heads:
- **Head-specific agreement.** The matched NMI minus a random one-to-one pairing's is 0.0026–0.0029
  for the LCM's three seed pairs. It is 0.0013–0.0014 for SONAR on the TPU (0.0012–0.0014 on the
  GPU), 0.0016 recoloured and 0.0012 untrained.
- **Most of each arm's NMI is a level any pairing reaches.** The LCM doubles the head-specific part
  too.
- **A head agrees with its match in another seed scarcely more than two heads of one seed agree:**
  0.032–0.034 against 0.028–0.032 for the LCM, and 0.016–0.017 against 0.014–0.015 for SONAR.

**Is it effective dimension?** `spectrum.py` measures each arm in the frame its loss weighs (rows
scaled by the square roots of their MSE weights), on 262,144 prefix rows and the tail's 262,144. The
tail:

| Arm | Participation ratio | Effective rank | Directions for 50% / 90% | Share between languages | Within-language participation ratio |
|---|---|---|---|---|---|
| LCM, block 4 | 141 | 315 | 63 / 366 | 0.154 | 143 |
| SONAR recoloured | 149 | 324 | 66 / 371 | 0.072 | 166 |
| Untrained LCM | 332 | 602 | 143 / 684 | 0.057 | 387 |
| SONAR | 413 | 701 | 211 / 711 | 0.043 | 461 |

- **The LCM's states are much lower-dimensional than SONAR's.** A code of fixed capacity covers more
  of such a source, and might agree more for that reason alone.
- **The gap in FVU_w is the spectrum.**
  - Take the rate at which reverse water-filling on each arm's spectrum (`pareto.gaussian_reference`)
    reaches the arm's FVU_w: 305 bits for the LCM, 315 for the untrained LCM and 363 for SONAR.
  - Measured against its own spectrum, SONAR's code is not the worse one.
- **The untrained arm already argued against dimension.** It is lower-dimensional than SONAR, and it
  agrees slightly less.

**The control (`recolour.py`).**
- **The map.** SONAR's rows go through one linear map that gives them the LCM's weighted-frame spectrum
  rank for rank:
  - y = c R diag(s) Uᵀ(√w ⊙ x), where U and λ are SONAR's weighted-frame eigenvectors and eigenvalues;
  - s = √(λ_LCM / λ_SONAR);
  - R is a random rotation, so the trainer's inverse-variance weights stay near uniform;
  - c brings each coordinate's variance near 1;
  - the one direction the `no837` weights null is zeroed.
- **Layout and training.** It is laid out as the `lcm_b4` harvest, with its sidecars and its own
  matched weights. It trains as the LCM arm, with only the cache and the weights changed.
- **Fidelity.** Under its own weights its participation ratio reads 149, against the target's 143.
  The float16 copy rounded 102 of its 4.8 billion values to zero.

What it shows:
- **FVU_w** reads 0.324: 76% of SONAR's gap to the LCM.
- **Agreement.** The excess on all heads, 0.0158, closes 22% of the gap. Within a language, 0.024
  closes 10%.
- **So the spectrum accounts for most of the LCM arm's lower error, and little of its agreement.**
- **At nearly the same spectrum,** the LCM's rows code to 0.281 and the recoloured rows to 0.324: 305
  Gaussian-equivalent bits against 255. The LCM's states have structure beyond their covariance that
  the code uses, which the recoloured rows lack.

**Context averaging does not explain it either.** `eval_next.py`'s copy baseline, which predicts each
row by the one before, measures how alike adjacent rows are. Its FVU_w is 0.44 for the LCM, 0.77 for
the untrained LCM, 1.37 recoloured and 1.53 for SONAR. The untrained LCM's adjacent rows are far more
alike than SONAR's, yet its codes agree no more.

**Caveats.**
- **The control matches the total spectrum, not the split between languages.** Its share between
  languages is 0.072 against the LCM's 0.154, and its within-language participation ratio is 166
  against 143.
  - A control that recolours the within- and between-language covariances separately would close
    that.
  - It matters most for the all-heads score, and less within a language, where the LCM leads most.
- **Seeds.** The control and the untrained arm have two seeds each.
- **Next-sentence FVU_w doesn't compare across arms,** since the LCM's rows carry their document's
  context. For the record, against each arm's copy and ridge baselines:
  - LCM: 0.559–0.561 (copy 0.439, ridge 0.309);
  - untrained: 0.746 (0.766, 0.542);
  - SONAR: 1.213–1.214 (1.533, 0.897);
  - recoloured: 1.190 (1.369, 0.832).

**Not yet run:**
- `closure.py` on each arm;
- the plan's across-models test, a dictionary on LCM seed 43's states, which needs `--cache-b`;
- Phase 3.

## 2026-11-20: Phase 1 on the RTX 3090: both gates pass, and the seeds' states agree up to nearly a rotation

**Runs.** `train_lcm.py --cache data/sonar_embeddings/mc4_sent16_f16.npy --epochs 1 --seed 42`, then
`--seed 43`: the default 104.9M-parameter SwiGLU model, 7,216 steps each at TF32, 51 minutes for both.
The outputs are in `data/out/lcm-organism/lcm_l8_d1024_s{42,43}`. The readings come from `phase1.sh`, into
`data/out/lcm-organism/phase1/`, with its tail-only harvests in `phase1/tail`.

**Gate A: it has learned dynamics.** Held-out next-sentence FVU_w on the 26,335 pairs:

| Step | Seed 42 | Seed 43 |
|---|---|---|
| 0 (untrained) | 2.587 | 2.580 |
| 2,500 | 0.842 | 0.845 |
| 5,000 | 0.829 | 0.829 |
| 7,216 | **0.8245** | **0.8250** |

- **The margin.** Against the context ridge's 0.870, the LCM is 0.045 better. The seeds differ by
  0.0005, so the margin is about 90 times the spread. Against the ridge from x_t (0.898), it is 0.073
  better.
- **Most of the gain came early.** The last 2,216 steps added 0.004, so a second epoch would probably add
  little.

**Gate B: the state is more than its input.** The held-out R² of the state on the scored tail's 32,771
rows, from ridges fitted on 227,791 rows of 20,000 training documents:

| Block | Seed | x_n | x_n + earlier mean | x_n read alone |
|---|---|---|---|---|
| 4 | 42 | 0.544 | 0.686 | 0.730 |
| 4 | 43 | 0.573 | 0.714 | 0.736 |
| 8 | 42 | 0.348 | 0.501 | 0.582 |
| 8 | 43 | 0.343 | 0.497 | 0.584 |

- **What the context adds.** Even counting everything the model computes from x_n alone, x_n explains
  73% of block 4's state and 58% of block 8's. The context supplies the remaining 26-27% at block 4 and
  42% at block 8.
- **The plan's reading.** It said to move to a later block only if x_n explained nearly all of the
  state, so both blocks qualify. Block 8 holds more context, by this measure.
- **The prefix mean carries much of it.** A linear readout from x_n and the mean of the earlier rows
  already reaches 0.69-0.71 at block 4, close to the 0.73 of the alone state.

**Agreement between the seeds' states** (`stateagree.py`, on the tail's last 65,536 rows, maps fitted on
the first half and scored on the second):

| Block | Models | CKA | Ridge R², 42→43 / 43→42 | Procrustes | Whitened Procrustes |
|---|---|---|---|---|---|
| 4 | trained | 0.912 | 0.868 / 0.865 | 0.183 | 0.801 |
| 4 | untrained inits | 0.378 | 0.153 / 0.154 | 1.380 | 1.654 |
| 8 | trained | 0.911 | 0.910 / 0.911 | 0.102 | 0.678 |
| 8 | untrained inits | 0.297 | 0.077 / 0.080 | 1.518 | 1.781 |

The Procrustes columns are residuals: 0 is identical, and about 2 means independent. CKA with rows
shuffled is at most 0.005.

- **Training makes the two seeds' representations far more alike** than their inits are.
- **Nearly a rotation.**
  - A rotation, after scaling each side to unit total variance, explains 82% of the other seed's
    variance at block 4 and 90% at block 8.
  - A general linear map explains 87% and 91%.
  - So in the variance-weighted metric, the states are identified nearly up to a rotation and a scale.
    The plan expected the rotation test to read clearly worse than the linear map, and it reads only a
    little worse.
- **But the agreement is in the high-variance directions.**
  - Whitened, every direction counts equally, and the residual is 0.80 at block 4 and 0.68 at block 8.
    That is well short of the inits' 1.65-1.78, and far from identity.
  - So the shared part is a high-variance subspace, while most of the 1,024 directions, each with
    little variance, are not shared.
  - Some of the whitened residual may be estimation noise in the smallest directions, fitted from
    32,768 rows. The canonical correlations would show how large the shared subspace is.
- **What the theory suggested by analogy** was that learned dynamics may fix a subspace but not a basis.
  Here the subspace is shared, and within it the two seeds differ by little more than a rotation.
  Categorical heads cannot rotate, so Phase 2 asks whether the dictionary supplies the basis.

**Scale and range.** The trained states are large.
- The largest magnitudes in the tail harvests are 150 (seed 42) and 101 (seed 43) at block 4, and 2,320
  and 1,943 at block 8. The untrained inits stay under 19.
- All of these fit float16, whose relative precision is the same at any magnitude. But a few very large
  coordinates at block 8 are a scale question for the dictionary recipe in Phase 2, alongside the norm
  question noted on 2026-11-19.

## 2026-11-19: throughput pilot on the RTX 3090: 23 minutes per epoch

**Run.** `train_lcm.py --cache data/sonar_embeddings/mc4_sent16_f16.npy --seed 42 --max-steps 175`, at the
default size (104.9M parameters, 512 documents per step) and JAX's default matmul precision on the GPU
(TF32). That is about 1.06M rows. This run's MLP was built from `nn.Dense`, and its output was deleted
when the MLP moved to `NLinear`. The rerun below is in `data/out/lcm-organism/pilot_gpu`.

**Speed.**
- **Training.** From step 50 on it trains at 31,300 rows per second: 6,058 real rows per step in 0.194 s.
  An epoch is 7,216 steps, which comes to 23 minutes.
- **FLOP rate.** Counting the padded positions, which are 26% of each batch, that is 27 TFLOP/s at
  6 × parameters per position. A 3090's dense TF32 peak is about 36 TFLOP/s.
- **Not the loader.**
  - A rerun with every batch already in the page cache trained at the same rate. There the loader alone
    delivers 637,000 rows per second.
  - Read cold from disk, the loader delivers 31,900 rows per second. That is nearly the same rate,
    which is a coincidence: random reads through mmap fault one page at a time.
  - So an epoch here that isn't in the page cache would also be held to about this rate by the disk. A
    TPU host that holds the cache in tmpfs reads it from memory.

**Learning.** The loss fell from 0.111 at step 25 to 0.044 by step 75, and flattened there: with 175
steps, the learning rate was still warming up. The held-out next-sentence FVU_w was 2.589 untrained and
0.966 at step 175. That is below the training mean (1.000), still short of the ridge from x_t (0.898).

**Reproducible.** The rerun matched every logged loss and both FVU_w values to every digit.

**Checkpoints.** 1.1 GB each, holding the parameters and both of Adam's moments. Step 0 is 372 MB.

**For the TPU estimate.**
- The plan's figure of "ten minutes per epoch at 30% of bfloat16 peak" leaves out two things. Matmul
  precision `high` takes three bfloat16 passes, which puts a v5e chip's ceiling near 66 TFLOP/s. And a
  quarter of every batch is padding.
- At this pilot's FLOP rate, an epoch on one chip would be about as long as on the 3090. That is an
  estimate until a TPU pilot runs.

**Rerun with the MLP from `NLinear`, SwiGLU and bilinear (same day).** The same 175 steps, after `GLU`
replaced the `nn.Dense` MLP. The init draws differ, because the parameter tree changed.

| Run | Rows/s (steps 50-175) | Loss at 175 | Held-out FVU_w, step 0 → 175 |
|---|---|---|---|
| `--gate silu` (SwiGLU), `pilot_gpu` | 30,900-31,300 | 0.0442 | 2.587 → 0.970 |
| `--gate none` (bilinear), `pilot_gpu_bilinear` | 31,100-31,500 | 0.0426 | 2.585 → 0.935 |

- **Speed.** Unchanged from the `nn.Dense` MLP (31,300 rows/s). Holding both up-projections in one
  weight costs nothing here.
- **The bilinear run is ahead at step 175.** But that is one seed, still inside the learning-rate
  warmup, so it doesn't compare the two MLPs.

## 2026-11-19: Phase 0 built; Gate A's baselines on mc4_sent16

**The baselines** (`train_lcm.baselines`), on `eval_next.py`'s 26,335 held-out adjacent pairs under the
`no837` weights. The ridges are fitted on the 191,440 adjacent pairs of 20,000 training documents:

| Predictor of x_{t+1} | Held-out FVU_w |
|---|---|
| the training mean | 1.0001 |
| copying x_t | 1.5327 |
| ridge from x_t | 0.8977 |
| ridge from x_t and the mean of the run's earlier rows | 0.8697 |

- The first three reproduce `eval_next.py`'s 1.0001, 1.53 and 0.896 (2026-10-13). The ridge differs
  in the third decimal because it is fitted on another sample.
- The context mean is worth 0.028 of FVU_w. Gate A asks the LCM to beat 0.870 by more than its seed
  spread.

**Checks on CPU.**
- **A planted dynamic.** On a synthetic cache whose next row is a rotation of the row before last, a
  two-block LCM reaches held-out FVU_w 0.23 in 400 steps. The ridge from x_t reads 0.995 there, and the
  context ridge 0.66 (`tests/test_lcm_organism.py`).
- **A harvest drops into the dictionary trainer.**
  - I trained a one-block, 1024-wide LCM for 20 steps on the float16 cache, and harvested 200k prefix
    rows plus the 262,144-row tail.
  - `train_pair.py` then ran a 77-step epoch on the harvest with its matched weights, and the loss fell
    from 1.21e-4 to 1.04e-4.
  - The harvest's scored tail is the cache's, 32,771 rows.
- **The states' scale.** Rows of that barely trained harvest have mean norm about 40, against about
  0.2 for SONAR rows. The dictionary init rescales to the input, and the matched weights put the loss on
  SONAR's scale. Whether the recipe's other settings are scale-free is still to check, in Phase 2.

## 2026-11-19: the float16 copy of mc4_sent16

`f16cache.py --bulk <dir> --weights data/sonar_embeddings/mc4_sent16.mse_weights.npy --weights
data/out/sonar/sentcache/mse_weights_no837.npy` wrote `data/sonar_embeddings/mc4_sent16_f16.npy`:
44,113,823 rows × 1024, 84.1 GiB. It took 8 minutes, reading at 0.36 GiB/s. The array is on another
disk behind the symlink (`float16.array` in its `meta.json`), and the sidecars are hard links to
mc4_sent16's.

- **Range.** The largest magnitude is 0.165, far inside float16's range, and no value is non-finite.
- **Small values.**
  - 0.8% of values (359M of 45.2G) fall in float16's subnormal range, below 6.1e-5, where the
    spacing is fixed at 6e-8.
  - 175,165 nonzero values, 4e-6 of the total, round to zero.
- **What rounding costs.** The figures are the same on 32,768 rows spread over the training prefix as
  on the 32,768-row tail:
  - each row moves by 2.07e-4 of its norm (median; the largest is 2.8e-4);
  - the whitened FVU of the rounding is 1.95e-7 under `mc4_sent16.mse_weights.npy`, and 5.4e-8
    under the control's `no837` weights.

  This matches the plan's measurement on the tail (2026-11-18).
- **Read back.**
  - Every checked row equals the source rounded to float16.
  - Through the symlink, `StackedPairSource` returns float32 rows equal to the rounded source.

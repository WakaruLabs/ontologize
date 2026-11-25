# vertices: notes

## 2026-11-25: where the atoms sit, in the hard stack and the softmax model

**Short version.**
- **The hard stack's atoms sit at their cells' centroids.** Read about each head's means, an atom
  lies 0.03 of its cell's spread from the cell's least-squares atom. That is about the sampling
  floor for some 1,000 rows a cell. Half its cell's rows lie past it along the ray from the head's
  centre (0.49–0.52 in layers 0–3), as they would for a centroid of a symmetric cell.
  - Layer 4's atoms are pulled toward their head's centre: a centred length of 0.74 of their
    centroids', with 0.67 of the rows past them.
  - `condrefit.py` found the same layer furthest from its refit optimum.
- **Each hard head also carries a shared offset**, 0.10 of its RMS spread, in every layer.
  - It is a residual component proportional to the gain. Each head's own refit would absorb all
    of it, while a joint refit removes it once.
  - Its price is 0.0015–0.0016 FVU for every head of every layer, with equal quartiles, so it is
    one shared component. That is close to what `condrefit.py`'s joint refit of layer 1 gained
    (0.0017–0.0019, depending on the ridge).
  - *Corrected the same day:* the first run read it as placement. Without centring, atoms seemed
    1.2–1.7 times farther from their head's centre than their centroids, past 98–99.8% of their
    cells' rows. `place.py` now sets the shared part apart, and a planted test checks that it
    does.
- **The softmax model's vertices sit far outside their cells.**
  - About the head's means, a vertex is 10–20 times as far from its head's centre as its argmax
    cell's centroid (layer medians).
  - It lies past every row of its cell: none of its cell's rows project past it. It sits 7.6–15.8
    times beyond its cell's 95th-percentile projection.
- **They are out of distribution by the nearest-row test too.** All distances below are ratios to
  the typical held-out row's nearest-row distance.
  - The end-to-end argmax decode lies at 3.99, against 1.00 for the soft reconstruction. Its norm
    about the data mean is 4.6 times the rows' median.
  - With one layer-0 head forced (frozen) to another entry's vertex, a row lies at 1.40. Forced to
    its own argmax vertex, it lies at 1.27. Forced to that entry's trimmed extreme (the mean of its
    cell's rows between the 80th and 95th percentiles along the ray), it lies at 1.00.
  - So the soft model's trimmed extremes are the in-distribution vertices the sparsemax–argmax
    schedule aims for, and its actual vertices are not.
- **Extremes cost a hard code.** Moving one hard head's atoms to their trimmed extremes would add
  about 0.0009 FVU, 0.6% of the stack's 0.1534, per head moved. For the softmax model, the same move
  on one layer-0 head would cut its hard-decode error by 0.66 FVU. Per-head prices do not add
  across heads.
- **The archetypal SAE's angular score cannot see this.**
  - At layer 0, one minus the best cosine to a data row reads 0.754 for the hard stack's atoms,
    0.829 for the softmax model's, 0.734 for the k32 SAE's decoder directions, and 0.779 for random
    directions with the rows' covariance.
  - Later layers read 0.85–0.87 for both models, against 0.80–0.83 for random directions.
  - The softmax model's vertices are wrong in length, not in direction, so a measure of angle
    misses them.
- **The nearest-row test is blind for the hard stack.** One head of 76 moves a row too little: a
  forced atom and a forced extreme both read 0.951, as the unforced reconstruction does.

**Setup.**
- `place.py` on CPU, about 6 minutes a model, on the first 32,768 rows of `mc4_fresh`, which
  neither model trained on.
- 1,024 of those rows are forced and scored. The other 31,744 are the nearest-row reference, and
  16,384 of them the reference for the best-cosine score.
- Cells with fewer than 32 rows are skipped, and the trimmed extremes use the 80–95% band.
- Each model runs at its final temperature, from `log.jsonl`: 0.00015 for `ste_h76_init01`
  (argmax forward) and 0.03 for `resid_nc`.
- Gain × atom decomposes both forwards to 1e-5. FVU_w reproduces the findings: 0.1534 for the
  hard stack, and 17.37 for the softmax model's argmax decode (0.0054 soft).

`ste_h76_init01` (step 369,500; 5 × 76 heads × 32 entries; `ste`, gain–shape):

| layer | entries | off | off_raw | common | beyond | reach | centred atom / centroid | extreme price per head (FVU) |
|---|---|---|---|---|---|---|---|---|
| 0 | 2432 | 0.032 | 0.106 | 0.101 | 0.492 | 0.71 | 0.97 | +9.5e-4 |
| 1 | 2432 | 0.033 | 0.107 | 0.101 | 0.508 | 0.68 | 0.95 | +9.2e-4 |
| 2 | 2432 | 0.033 | 0.107 | 0.101 | 0.512 | 0.65 | 0.92 | +9.0e-4 |
| 3 | 2432 | 0.034 | 0.107 | 0.101 | 0.524 | 0.63 | 0.88 | +8.6e-4 |
| 4 | 2432 | 0.037 | 0.108 | 0.101 | 0.668 | 0.52 | 0.74 | +8.5e-4 |

`resid_nc` (step 372,600; 5 × 32 heads × 32 entries; softmax, no gain–shape):

| layer | entries | off | off_raw | common | beyond | reach | centred atom / centroid | extreme price per head (FVU) |
|---|---|---|---|---|---|---|---|---|
| 0 | 925 | 7.06 | 7.06 | 0.78 | 0.000 | 9.9 | 14.8 | −0.66 |
| 1 | 954 | 5.52 | 5.48 | 0.75 | 0.000 | 10.6 | 15.2 | −0.19 |
| 2 | 838 | 3.10 | 3.11 | 0.63 | 0.000 | 7.6 | 12.1 | −0.034 |
| 3 | 755 | 2.18 | 2.22 | 0.60 | 0.000 | 15.8 | 20.2 | −0.016 |
| 4 | 127 | 0.93 | 1.05 | 0.49 | 0.000 | 10.2 | 10.5 | 0.0 |

All values are medians over the layer's entries (heads, for `common` and the price).
- Only 127 of the softmax model's 1,024 layer-4 entries have 32 rows or more. So 99 of its 1,024
  layer-4 forced rows have an extreme to force to; every other layer has 759 or more.

Forced rows' nearest-row distance (ratio to the held-out rows' median, 1.00):

| layer forced | hard: atom | hard: extreme | soft: atom | soft: own vertex | soft: extreme |
|---|---|---|---|---|---|
| 0 | 0.951 | 0.952 | 1.399 | 1.267 | 1.002 |
| 1 | 0.951 | 0.952 | 1.126 | 1.090 | 1.002 |
| 2 | 0.951 | 0.952 | 1.049 | 1.017 | 1.006 |
| 3 | 0.951 | 0.952 | 1.035 | 1.007 | 1.005 |
| 4 | 0.951 | 0.951 | 1.042 | 1.002 | 1.006 |

The reconstructions themselves read 0.951 (hard) and 1.002 (soft). One model and one seed each.

**Not yet done.**
- Whether forcing a trimmed extreme steers better than forcing the vertex, read through M2M100,
  for the softmax model, where the two differ. `steer.py` (README.md) was queued 2026-11-25,
  after the seed test. It adds the cell's centroid as a third target and takes each step at
  matched lengths too.
- The one-head seed pair started from spherical k-means: done (its section is below).
- The planted toy: done (the last section).

```bash
JAX_PLATFORMS=cpu uv run python experiments/vertices/place.py \
    --ckpt data/out/sonar/multilingual/ste_h76_init01 \
    --sae data/out/sonar/sae_conv/m5120_k32 --out data/out/sonar/vertices
JAX_PLATFORMS=cpu uv run python experiments/vertices/place.py \
    --ckpt data/out/sonar/multilingual/resid_nc \
    --sae data/out/sonar/sae_conv/m5120_k32 --out data/out/sonar/vertices
```

## 2026-11-25: the one-head seed test from a k-means start, fitted and queued

**The question.** The stock one-head pair (`width_seeds.sh`, h = 1) agrees at NMI 0.418.
Spherical k-means at 32 cells agrees at 0.732 in the whitened row frame. Is the shortfall the
start, or the training? The archetypal SAE's stability is confounded the same way. The runs:
- `km42_s42` and `km43_s43`: the stock recipe, each seed started from its own spherical
  k-means fit;
- `km42_s43`: seed 43 started from seed 42's fit, the shared start.

**The fits** (`kmstart.py fit`; `data/out/sonar/vertices/kmstart/fit/summary.json`).
- They are fitted on `kmeans_seeds.py`'s 131,072 training rows, in the frame the classifier
  reads: the shape, centred and weighted by sqrt(`mse_weights_med837`).
- They are scored on the cache's last 32,768 rows, the rows `partition.py` scores.
- **In this frame the two fits agree at NMI 0.705** (null 0.0045). The 0.732 above was in the
  row frame, which the classifier cannot represent, because it sees the shape and not the norm.
- **The stock heads match these fits better than they match each other**: 0.465–0.474 against
  each fit, against 0.418 between themselves.
- Cells hold 1.3–5.8% of the training rows.
- **The logit scale.** The trained stock head's median top-two gap is 0.00457, which is 30.4 T.
  The starts are scaled to it, by 0.205 for both fits.

**The start takes.** The smoke test built seed 43's model from seed 42's start on CPU, through
`train_kmstart.py`'s flags. That model selects the start's cell on all 512 probe rows, and
decodes every atom to its target within 6e-9. Its median gap on those rows is 0.00426.

**Reading the outcome.**
- If the own-start pair keeps most of 0.705, the agreement belongs to the start and the
  optimizer, and a centroid start may explain the archetypal SAE's stability too.
- If it falls to about 0.42, the shared-start pair says how much of that loss training adds by
  itself, from a common point.

Queued 2026-11-25 through `kmstart_seeds.sh` (README.md). It waits for an idle GPU, trains three
50,000-step runs, and scores into `data/out/sonar/vertices/kmstart/scores/`. On the RTX 3090 the
first run, `km42_s42`, took 41 minutes (20 steps a second).

## 2026-11-25: the one-head seed test, results

**Short version.** Training from spherical k-means keeps nearly all of the start's agreement. The
two runs from their own starts agree at NMI 0.680, against 0.705 for the starts and 0.418 for the
stock pair. So most of the stock pair's disagreement comes from its start, not from training.
- **From one shared start**, seeds 42 and 43 agree at 0.872. Training noise alone, from a common
  point, costs 0.13.
- **Each run moves away from its start.** A run agrees with its own start at 0.766–0.773 and with
  the other seed's at 0.646–0.658. The two runs agree with each other (0.680) better than either
  agrees with the other's start, so some of what training changes, it changes the same way in
  both.
- **The k-means start also trains to a lower error.** Over the last 2,000 steps the training MSE
  is 6.645–6.650e-6 for the three k-means-start runs and 6.711–6.720e-6 for the stock pair. The
  k-means-start runs average 6.67e-6 over their first 500 steps, already below where the stock
  runs end.
- **For the archetypal SAE.** Its dictionary is built on fixed k-means centroids, and its reported
  stability is not separated from that start. Here a k-means start alone gives a head trained by
  the stock recipe most of k-means' agreement, so the start may explain much of that stability
  too.

**Setup.** `kmstart_seeds.sh` (README.md) trains `width_seeds.sh`'s h = 1 recipe for 50,000
steps. Each run took 41–46 minutes on the RTX 3090 (18–20 steps a second). `kmstart.py score`
scores the runs on the cache's last 32,768 rows, beside rows-shuffled nulls, which read 0.004
throughout. The rows against the other seed's start come from the same scorer, given that start
(`scores/cross.json`).

| pair | NMI |
|---|---|
| the starts, s42 against s43 (`fit`) | 0.705 |
| `km42_s42` against `km43_s43` (each from its own start) | 0.680 |
| `km42_s42` against `km42_s43` (both from s42's start) | 0.872 |
| `km43_s43` against `km42_s43` | 0.681 |
| the stock pair (`width_seeds.sh`) | 0.418 |
| each run against its own start | 0.766–0.773 |
| each run against the other seed's start | 0.646–0.658 |

One head of 32 entries at one layer, and one pair per arm. The errors are training MSE, with the
recipe's training noise, not held-out FVU.

## 2026-11-25: the planted-simplex toy

**Short version.**
- **Where a head's vertices sit follows how hard its code is, against the data's own code.** A
  code harder than the planted one pulls the vertices in toward the cells' centroids. A code no
  harder than the planted one puts them at the planted corners.
- **Softmax puts the vertices on the hull, not outside it.** Its vertices sit at λ 0.96 (α 0.2)
  and 1.01 (α 0.05), as archetypal analysis's do (0.97 and 1.02). So the far-out vertices of
  `resid_nc` are not a property of softmax selection on data like these. The toy has one head,
  so a sum of many heads, depth, or data that are not a simplex remain as explanations.
- **Jade's schedule does what it aims for when the planted code is sparse, and so does sparsemax
  at the same temperature without it.**
  - At α 0.05 the schedules' vertices sit at λ 0.95–0.96, error 0.06. Their code is about as
    sparse as the planted one: 2.5 entries a row and 23–24% of rows decided, against 2.3 and 20%.
  - Fixed-temperature sparsemax reaches the same medians.
  - The argmax phase's visible effect is reliability. Each schedule found every corner in 9 of 10
    seed runs over the two α, against 6 of 10 for fixed-temperature sparsemax. Ten runs is too
    few to call that settled.
- **The stock anneal makes sparsemax too sparse.** At the anneal's end temperature (0.03),
  sparsemax decides 69% of the rows at α 0.05, where the planted code decides 20%. At α 0.2 it
  decides 30%, against 0.2%. Its vertices sit at λ 0.86 and 0.68. The decided-share setpoint
  ends the same way.
- **`ste` leaves the vertices at the cells' centroids**, as it does on SONAR: λ 0.49 against
  centroids at 0.52 (α 0.2). It merged corners in 6 of its 10 seed runs, more than any other
  arm.
- **For the SONAR proposal.** The trained stock head's median top-two logit gap is 30
  temperatures (`kmstart.py fit`), and sparsemax gives a one-entry code wherever that gap exceeds
  one temperature. At the recipe's temperature the median row would be decided with a wide
  margin, so by the toy its vertices would stay near the centroids, where `ste` leaves them. On
  the toy, what moved the vertices was the sparsemax phase's temperature, not the switch to
  argmax and back.

**Setup.** `simplex.py`'s defaults:
- 8 atoms in 16 dimensions, N(0, 1) per coordinate, and noise SD 0.05 per coordinate;
- 32,768 training rows and 8,192 held out;
- one head of 8 entries, with `e_dec` 32 and the quadratic classifier with its constant
  coordinate;
- 4,000 steps of batch 256 at lr 1e-3, as in `dictenc-toy`.

The stock anneal takes T from 1 to 0.03 over 3,000 steps, ramps `p_drop` from 0 to 0.1, and takes
the logit noise from 0.02 to 0.006. The fixed-temperature arms hold T 1, `p_drop` 0.1 and logit
noise 0.02. The short schedule switches to `ste` at 75% of the steps and back at 90%, the long one
at 60% and 75%. The setpoint holds 50% of rows decided. Five seeds of every arm took about 2.5
minutes on CPU. Outputs are in `data/out/vertices/simplex/k8_d16_a{0.2,0.05}_s0.05_x5`.

λ is a point's position along the ray from the planted atoms' mean through its matched atom: 1
at the atom, below 1 inside. err is its distance from the atom, over the atom's distance from
the mean. Clean seeds have acc ≥ 0.9: a seed that merges two corners into one entry leaves a
corner unmatched. Support counts entries carrying at least 1% of a row. Decided rows have one
entry at 99% or more. Medians over five seeds:

α 0.2. The planted code has support 4.51 and decides 0.2% of rows. The planted cells' centroids
sit at λ 0.53, and the noise floor is FVU 0.0077.

| arm | clean | λ vertex | err vertex | λ centroid | support | decided | FVU | FVU, argmax |
|---|---|---|---|---|---|---|---|---|
| `ste` | 3 | 0.490 | 0.512 | 0.524 | 1.00 | 1.000 | 0.282 | 0.282 |
| `softmax` | 5 | 0.960 | 0.083 | 0.532 | 7.34 | 0.000 | 0.053 | 0.750 |
| `sparsemax` | 4 | 0.677 | 0.324 | 0.530 | 2.06 | 0.302 | 0.095 | 0.322 |
| `sparsemax_fixed` | 2 | 0.900 | 0.477 | 0.463 | 4.58 | 0.004 | 0.019 | 1.313 |
| `sparsemax_setpoint` | 3 | 0.620 | 0.381 | 0.528 | 1.81 | 0.480 | 0.153 | 0.311 |
| `schedule_short` | 4 | 0.875 | 0.128 | 0.529 | 3.95 | 0.017 | 0.021 | 0.587 |
| `schedule_long` | 4 | 0.900 | 0.114 | 0.531 | 4.10 | 0.011 | 0.020 | 0.624 |
| `archetypal` | 5 | 0.972 | 0.037 | | 4.58 | 0.007 | 0.006 | |

α 0.05. The planted code has support 2.31 and decides 20.4% of rows. The planted cells'
centroids sit at λ 0.80, and the noise floor is FVU 0.0041.

| arm | clean | λ vertex | err vertex | λ centroid | support | decided | FVU | FVU, argmax |
|---|---|---|---|---|---|---|---|---|
| `ste` | 1 | 0.601 | 0.422 | 0.661 | 1.00 | 1.000 | 0.186 | 0.186 |
| `softmax` | 5 | 1.012 | 0.051 | 0.794 | 5.22 | 0.000 | 0.030 | 0.189 |
| `sparsemax` | 5 | 0.860 | 0.146 | 0.794 | 1.42 | 0.692 | 0.046 | 0.127 |
| `sparsemax_fixed` | 4 | 0.955 | 0.066 | 0.793 | 2.54 | 0.191 | 0.014 | 0.161 |
| `sparsemax_setpoint` | 5 | 0.865 | 0.140 | 0.794 | 1.44 | 0.674 | 0.041 | 0.127 |
| `schedule_short` | 5 | 0.956 | 0.063 | 0.793 | 2.53 | 0.241 | 0.018 | 0.158 |
| `schedule_long` | 5 | 0.954 | 0.060 | 0.793 | 2.55 | 0.227 | 0.016 | 0.160 |
| `archetypal` | 5 | 1.017 | 0.019 | | 3.27 | 0.023 | 0.003 | |

- **Medians over failed seeds.** A median can fall on a failed seed. At α 0.2,
  `sparsemax_fixed`'s error of 0.48 is one: its two clean seeds both read 0.11.
- **The setpoint at α 0.05.** It ended at the starting temperature, which is its bound, with 67%
  of rows decided. Once the classifier's logits had grown under a lower temperature, the
  controller could not soften the code back to the target.
- **Archetypal analysis.** Its λ slightly above 1 at α 0.05 is the noise widening the hull.
- **Caveats.** One head at one layer, with as many entries as atoms, and one noise level. This
  classifier can represent the barycentric coordinates exactly, since they are affine in x; a
  SONAR head's need not.

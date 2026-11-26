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

## 2026-11-25: the toy at fifty seeds, taken toward the stack, and a Voronoi head

**Short version.**
- **The argmax phase of the sparsemax–argmax–sparsemax schedule repairs merged corners.** Over
  fifty seeds per arm, with the same data and seeds for every arm, the schedules found every
  corner in 37–43 runs, and fixed-temperature sparsemax in 23. Paired by seed, they rescued 14–20
  runs and lost none. Where both found every corner, the vertices sit in the same places.
- **Two of the stack's features make soft codes overshoot: unit-norm rows and depth.** More heads
  don't, and neither does a layer 0 without its constant coordinate.
  - On unit-norm rows, one softmax head's vertices move to λ 1.38, past every row.
  - At layer 2 of three, softmax atoms sit at 7.3 times their centroids' length (reach 4.6), as
    in `resid_nc`.
  - The short schedule cut both (λ 1.04; reach 2.0 at layer 2). That came from its short last
    phase: with a last phase of a quarter of the steps the vertices went back out, to λ 1.39 on
    unit rows and reach 2.7 at layer 2.
  - With every knob on together, the cut mostly goes. The fixed-temperature soft arms overshoot
    at every layer, and annealed softmax, whose code hardened there, doesn't. So the overshoot
    follows a soft code under unit-norm rows or depth.
- **A Voronoi head, whose logits are tied to its atoms, is the more reliable hard code.**
  - The Voronoi STE found every corner in 7–8 of 10 seeds at α 0.2 and 10 of 10 at α 0.05,
    from atoms started near the rows' mean.
  - The stock `ste` head found them in 4 and 5 of 10 with its training noise, and in 2 and 6 of
    10 without it.
  - The Voronoi STE also fits better (argmax FVU 0.265–0.268 against 0.29–0.30, and 0.117–0.119
    against 0.147–0.156).
- **As a soft code the Voronoi head matches stock softmax, at a fixed temperature.**
  - It fits about as well (FVU 0.055 against 0.053 at α 0.2, 0.025 against 0.031 at α 0.05),
    finds the corners as often, and puts its vertices just past them (λ 1.06–1.10, nothing
    beyond).
  - Under the stock anneal it hardened instead. Its one scale is learned as log β, which Adam
    at lr 1e-3 moves about one e-fold per 1,000 steps, while the anneal lowers the temperature
    33-fold in 3,000. A Voronoi head's logit scale has to be free to follow the temperature.
- **Training noise holds stock softmax's vertices on the hull.** Without it, they sit at λ 1.17
  past the corners, and the soft FVU is lower.

**The schedule at fifty seeds** (`sparsemax_fixed`, `schedule_short`, `schedule_long`; α 0.05 and
0.2, otherwise `simplex.py`'s defaults; `data/out/vertices/simplex/k8_d16_a{0.05,0.2}_s0.05_x50`).
A clean run has acc ≥ 0.9; one that isn't merged two corners into one entry. Rescued means the
schedule was clean and `sparsemax_fixed` wasn't, on the same seed; lost is the reverse. The
p-values are exact McNemar tests.

| | α 0.05 | α 0.2 |
|---|---|---|
| `sparsemax_fixed`, clean | 23 / 50 | 23 / 50 |
| `schedule_short`, clean | 43 / 50 | 37 / 50 |
| `schedule_long`, clean | 43 / 50 | 38 / 50 |
| `schedule_short`, rescued / lost | 20 / 0 (p = 2e-6) | 14 / 0 (p = 1e-4) |
| `schedule_long`, rescued / lost | 20 / 0 (p = 2e-6) | 15 / 0 (p = 6e-5) |

- **Placement where both arms were clean (23 seeds at each α).** λ is 0.956 for both at α 0.05.
  At α 0.2 it is 0.900 for `sparsemax_fixed` against 0.883 (short) and 0.898 (long).
- **The five-seed table overstated `sparsemax_fixed`.** Its four clean seeds of five at α 0.05
  were a lucky draw.

**The knob grid** (`simplex_grid.sh`; α 0.2, five seeds, 4,000 steps). The measures are
`place.py`'s, per layer, as medians of atom over centroid / reach / beyond. Vertices at the
corners read atom over centroid above 1, since the centroids sit inside, with reach near 1.
Vertices past the data read reach well above 1 with nothing beyond.

| configuration | `softmax` | `sparsemax_fixed` | `schedule_short` | `ste` |
|---|---|---|---|---|
| one head | 1.81 / 1.07 / 0.014 | 1.85 / 1.07 / 0.015 | 1.67 / 0.99 / 0.054 | 0.94 / 0.56 / 0.52 |
| `--h 4` | 1.69 / 1.03 / 0.034 | 1.83 / 1.12 / 0.010 | 1.47 / 0.92 / 0.114 | 1.05 / 0.57 / 0.45 |
| `--h 16` | 1.65 / 1.01 / 0.047 | 1.93 / 1.29 / 0.001 | 1.54 / 0.90 / 0.100 | 1.25 / 0.54 / 0.36 |
| `--no-const0` | 1.80 / 1.07 / 0.013 | 1.70 / 1.02 / 0.043 | 1.65 / 0.98 / 0.059 | 0.94 / 0.57 / 0.53 |
| `--unit` | 2.16 / 1.69 / 0 | 2.86 / 2.39 / 0 | 1.38 / 1.07 / 0 | 0.93 / 0.73 / 0.65 |
| `--no-const0 --unit` | 1.81 / 1.43 / 0 | 2.12 / 1.73 / 0 | 1.26 / 0.99 / 0.081 | 0.94 / 0.74 / 0.64 |
| `--l 3`, layer 0 | 1.67 / 1.14 / 0 | 1.61 / 1.04 / 0.010 | 1.54 / 1.02 / 0.031 | 1.03 / 0.81 / 0.44 |
| `--l 3`, layer 1 | 2.70 / 1.78 / 0 | 2.20 / 1.55 / 0 | 1.44 / 0.97 / 0.067 | 1.04 / 0.71 / 0.43 |
| `--l 3`, layer 2 | 7.30 / 4.56 / 0 | 7.80 / 5.04 / 0 | 3.17 / 2.02 / 0 | 1.14 / 0.67 / 0.34 |
| all knobs, layer 0 | 1.32 / 0.97 / 0.10 | 2.48 / 1.73 / 0 | 2.19 / 1.55 / 0 | 1.01 / 0.80 / 0.48 |
| all knobs, layer 1 | 1.33 / 0.72 / 0.26 | 5.32 / 2.70 / 0 | 4.85 / 2.36 / 0 | 0.89 / 0.51 / 0.59 |
| all knobs, layer 2 | 3.58 / 1.09 / 0.036 | 8.72 / 3.09 / 0 | 8.32 / 2.43 / 0 | 2.15 / 0.70 / 0.15 |

- **What `--l 3` means.** It runs with `--deepsup --resid-norm`.
- **"All knobs"** is `--h 16 --l 3 --deepsup --resid-norm --no-const0 --unit`. It took 25 minutes
  for five seeds of the four arms on CPU, beside other CPU jobs; the whole grid took 48.
  - There annealed softmax hardened, to 3.86 entries a row with 7.9% of rows decided, and its
    vertices stay at or inside the hull.
  - The fixed-temperature arms stay soft (5.2–5.7 entries a row, none decided) and overshoot at
    every layer.
  - The short schedule's cut mostly goes: its reach is 1.55, 2.36 and 2.43 at layers 0–2, against
    1.73, 2.70 and 3.09 for `sparsemax_fixed`.
- **The corner readout**, available with one head at one layer:
  - On `--unit`, λ is 1.377 for softmax, 1.372 for `sparsemax_fixed` and 1.038 for
    `schedule_short`.
  - On `--no-const0 --unit` it is 1.337, 1.305 and 0.967.
- **Soft FVU on `--unit`:** 0.059, 0.038 and 0.101 for the same three arms, so the schedule paid
  for its placement in reconstruction.
- **The mechanism on unit-norm rows.** Convex mixtures of points on a sphere lie inside the ball,
  so to reach rows on the sphere by mixing, the atoms have to sit outside it.
- **The long schedule on the two knobs** (run afterwards, same data and seeds;
  `data/out/vertices/simplex/grid_long_{unit,l3}_x5`). With a last sparsemax phase of a quarter of
  the steps instead of a tenth, the vertices go back out.
  - On `--unit`: λ 1.393, with atom over centroid / reach / beyond at 1.81 / 1.43 / 0 and soft
    FVU 0.042. That is `sparsemax_fixed`'s λ 1.372.
  - On `--l 3`: 1.59 / 1.03 / 0.017, 2.03 / 1.35 / 0 and 4.53 / 2.70 / 0 at layers 0–2, between the
    short schedule and `sparsemax_fixed`.
  - So the short schedule's cut came from its short last phase. The schedule delays the overshoot
    rather than preventing it.

**The Voronoi head** (`voronoi_ste`, `voronoi_softmax`; ten seeds per arm, medians; `simplex.py`'s
defaults except where noted; `data/out/vertices/simplex/voronoi_a*`, `stock_nonoise_a*`).
- "Mean" starts its atoms near the rows' mean, and "rows" at k training rows drawn at random.
- "Fixed T" holds the temperature at 1 (`--temperature-end 1`). The rest anneal as the stock
  arms do.
- The stock arms' noise is the recipe's: logit noise, feature noise and winner dropout. "No
  noise" turns all three off.

α 0.2 (the planted cells' centroids at λ 0.53):

| arm | clean | λ vertex | err vertex | support | decided | FVU | FVU, argmax |
|---|---|---|---|---|---|---|---|
| `ste` | 4 | 0.420 | 0.597 | 1.00 | 1.000 | 0.292 | 0.292 |
| `ste`, no noise | 2 | 0.437 | 0.576 | 1.00 | 1.000 | 0.302 | 0.302 |
| `voronoi_ste`, mean | 7 | 0.501 | 0.500 | 1.00 | 1.000 | 0.268 | 0.268 |
| `voronoi_ste`, rows | 8 | 0.523 | 0.478 | 1.00 | 1.000 | 0.267 | 0.267 |
| `voronoi_ste`, mean, fixed T | 8 | 0.534 | 0.467 | 1.00 | 1.000 | 0.265 | 0.265 |
| `softmax` | 9 | 0.962 | 0.089 | 7.34 | 0.000 | 0.054 | 0.751 |
| `softmax`, no noise | 7 | 1.171 | 0.207 | 7.96 | 0.000 | 0.036 | 1.279 |
| `voronoi_softmax`, mean | 9 | 0.587 | 0.413 | 2.34 | 0.368 | 0.170 | 0.276 |
| `voronoi_softmax`, rows | 8 | 0.840 | 0.216 | 6.28 | 0.001 | 0.076 | 0.515 |
| `voronoi_softmax`, mean, fixed T | 8 | 1.064 | 0.175 | 7.62 | 0.000 | 0.055 | 0.994 |

α 0.05 (the planted cells' centroids at λ 0.80):

| arm | clean | λ vertex | err vertex | support | decided | FVU | FVU, argmax |
|---|---|---|---|---|---|---|---|
| `ste` | 5 | 0.673 | 0.342 | 1.00 | 1.000 | 0.156 | 0.156 |
| `ste`, no noise | 6 | 0.776 | 0.225 | 1.00 | 1.000 | 0.147 | 0.147 |
| `voronoi_ste`, mean | 10 | 0.790 | 0.211 | 1.00 | 1.000 | 0.119 | 0.119 |
| `voronoi_ste`, rows | 5 | 0.715 | 0.295 | 1.00 | 1.000 | 0.139 | 0.139 |
| `voronoi_ste`, mean, fixed T | 10 | 0.795 | 0.205 | 1.00 | 1.000 | 0.117 | 0.117 |
| `softmax` | 10 | 1.017 | 0.051 | 5.28 | 0.000 | 0.031 | 0.191 |
| `softmax`, no noise | 10 | 1.170 | 0.185 | 7.60 | 0.000 | 0.019 | 0.318 |
| `voronoi_softmax`, mean | 9 | 0.796 | 0.204 | 1.03 | 0.975 | 0.116 | 0.119 |
| `voronoi_softmax`, rows | 4 | 0.831 | 0.324 | 2.41 | 0.285 | 0.080 | 0.208 |
| `voronoi_softmax`, mean, fixed T | 10 | 1.100 | 0.134 | 6.39 | 0.000 | 0.025 | 0.247 |

- **The start matters for the Voronoi STE.** From training rows it found every corner in 5 of 10
  seeds at α 0.05, against 10 from near the mean. Like k-means, it has local minima.
- **The toy favours a Voronoi head.** The planted cells are close to the corners' own Voronoi
  cells, so the toy can't test whether a free quadratic classifier finds covers that Voronoi
  cells miss. What it shows is that tying costs a soft code nothing here, and makes a hard code
  more reliable.

## 2026-11-25: the k-means-start heads' classifiers, against the stock heads'

**The question** (asked in the agent forum). The k-means start builds a linear classifier: one
factor on the constant coordinate and the other on a centroid. Did training from it add quadratic
curvature, or only move linear boundaries? The runs read are the three k-means-start runs and the
stock pair (`l1_h1_s42`, `l1_h1_s43`), all at step 50,000 and temperature 0.00015.

**Short version.** Linear boundaries moved. The k-means-start heads kept affine cells and gained
little curvature. The stock heads are quadratic: their cells are cones, and partly blind to sign.

**`bilinspec.py`** (`data/out/sonar/vertices/kmstart/bilinspec/<run>`). Per entry, the logit is
(w·u + a)(v·u + b), with w and v the two factors' semantic parts and a and b their constants.

| | \|cos(w, v)\| median | \|a\|/\|w\| | \|b\|/\|v\| | dominant eigenvalue negative |
|---|---|---|---|---|
| `km42_s42`, `km43_s43`, `km42_s43` | 0.332–0.336 | 0.059–0.061 | 1.74–1.82 | 0% |
| `l1_h1_s42`, `l1_h1_s43` | 0.028–0.032 | 0.015–0.038 | 0.034–0.035 | 31–47% |

- **The k-means-start factor never changes sign.** In those runs v is the factor that started on
  the constant. Its constant is still 1.7–1.8 times its semantic part, so v·u + b keeps its sign
  for every unit u, and each logit is an affine form times a nearly constant positive one.
- **The stock heads' forms are saddles.** Both factors are semantic, with constants of a few
  percent of their norms.
- **The odd share doesn't separate them.** It reads 1.000 for the k-means-start runs and
  0.68–0.73 for the stock ones, but the script notes that on isotropic inputs a small constant
  dominates it in high dimensions.

**`cellshape.py`** (the cache's last 32,768 rows; `data/out/sonar/vertices/kmstart/cellshape`).
These are the shares of means of 2, 8 and 32 of an entry's rows, and of its centroid, that keep
the entry.

| | rescaled: mean2 / mean8 / mean32 / centroid | raw | negation |
|---|---|---|---|
| `km42_s42` | 0.930 / 0.705 / 0.598 / 0.562 | 1.000 throughout | 0.019 |
| `km43_s43` | 0.936 / 0.740 / 0.617 / 0.594 | 1.000 throughout | 0.021 |
| `km42_s43` | 0.910 / 0.721 / 0.592 / 0.562 | 1.000 throughout | 0.019 |
| `l1_h1_s42` | 0.994 / 1.000 / 1.000 / 1.000 | 1.000 throughout | 0.111 |
| `l1_h1_s43` | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 throughout | 0.265 |

- **Every head's cells are convex**, since every raw mean keeps its entry.
- **The stock heads' cells are convex cones,** which depend only on a row's direction. They are
  also partly sign-blind: 11–27% of negated rows keep their entry, against 1/32 by chance.
- **The k-means-start heads' cells are not cones.** They lose 7–44% of rescaled means, as the
  affine cells of the start's centred frame would. They are not sign-blind either: about 2% of
  negated rows keep their entry, as a linear head's would.

**Reading it.** The agreement the k-means start gives (NMI 0.680 against the stock pair's 0.418)
comes with heads whose cells keep the start's affine shape. Training from k-means gave v a
semantic part (it has none at the start) and so some curvature, but not enough to make the cells
cones or the logits sign-blind. So the comparison with a Voronoi head should expect the k-means-start
stock head to behave much like one already.

`bilinspec.py` now restores checkpoints as numpy, as `pareto.load_onto` does, so it runs under
`JAX_PLATFORMS=cpu`. It took seconds a run, and `cellshape.py` took 77 seconds for all five.

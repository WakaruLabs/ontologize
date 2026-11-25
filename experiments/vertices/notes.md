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
- Whether forcing a trimmed extreme steers better than forcing the vertex: decoding forced rows
  through M2M100, for the softmax model, where the two differ. It waits for an idle GPU.
- The one-head seed pair started from spherical k-means. It is built (`kmstart.py`,
  `train_kmstart.py`, `kmstart_seeds.sh`; README.md) but not yet run.
- The planted toy.

```bash
JAX_PLATFORMS=cpu uv run python experiments/vertices/place.py \
    --ckpt data/out/sonar/multilingual/ste_h76_init01 \
    --sae data/out/sonar/sae_conv/m5120_k32 --out data/out/sonar/vertices
JAX_PLATFORMS=cpu uv run python experiments/vertices/place.py \
    --ckpt data/out/sonar/multilingual/resid_nc \
    --sae data/out/sonar/sae_conv/m5120_k32 --out data/out/sonar/vertices
```

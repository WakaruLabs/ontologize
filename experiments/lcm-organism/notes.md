# lcm-organism: notes

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

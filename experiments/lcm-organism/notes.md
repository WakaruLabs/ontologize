# lcm-organism: notes

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

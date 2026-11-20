# lcm-organism: notes

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

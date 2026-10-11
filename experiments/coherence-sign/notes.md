# coherence-sign: notes

## 2026-10-09: checks on the copy, before launch

- `train_old.py --dry-run` gives a spec equal to `resid_nc`'s stored spec
  on all 43 of its fields. The copy adds two loss flags that postdate the
  run, both False.
- On CPU it reproduced `resid_nc`'s first 100 training steps to within
  0.1% in every logged statistic, so it starts from the same
  initialization and data order.
- `data/sonar_embeddings/mc4_4M.npy` and `data/out/sonar/mse_weights.npy`
  are byte-identical to the files `resid_nc` trained on.
- The environment is newer than the one the copy's code last ran in: JAX
  0.11.2 against 0.10.0, orbax 0.12 against 0.11.
- `check_copy.py` passes on the copy and fails on a scratch copy with one
  default changed.

## 2026-10-09: the between-head cosine penalty sets the sign

### `resid_nc_hc` is `resid_nc_hm` without the mean-entropy bonus

`data/out/sonar/multilingual/resid_nc_hc` was trained on the earlier
codebase on 19 August and ran all 372.6k steps. Its `log.jsonl`
configuration and stored spec differ from `resid_nc_hm`'s first segment
(steps 0 to 372.6k, before its first resume) only in `s_Hm` (0 against
1e-6) and `hmean_loss`, the flag that weight switches on. So it splits the
three settings without a retrain.

`headcoh.py` mean ζ_cos by layer. 96% of `resid_nc`'s heads are below −2;
every head of `resid_nc_hc` and of `resid_nc_hm` at 370k is above +2:

| run | step | L0 | L1 | L2 | L3 | L4 | mean |
|---|---|---|---|---|---|---|---|
| `resid_nc` | 372.6k | −8.2 | −28.2 | −18.5 | −18.5 | −14.4 | −17.6 |
| `resid_nc_hc` | 370k | +92.6 | +122.6 | +129.0 | +124.3 | +129.4 | +119.6 |
| `resid_nc_hm` | 370k | +90.8 | +122.5 | +132.5 | +125.8 | +131.3 | +120.6 |
| `resid_nc_hm` | 4.38M | −4.5 | +13.6 | +118.4 | +114.6 | +124.6 | +73.3 |

The bonus is not what makes `resid_nc_hm` coherent. By 4.38M its first two
layers have lost most of their coherence.

### All six runs at step 160k

Every run has a step-160000 checkpoint, including the three arms, which
were still training (below):

| run | `e` | `s_hcossim` | `s_Hm` | mean ζ_cos | median | heads > +2 | heads < −2 |
|---|---|---|---|---|---|---|---|
| `resid_nc` | 4096 | 0 | 0 | −13.1 | −14.1 | 0.09 | 0.88 |
| `resid_nc_ctl` | 4096 | 0 | 0 | −11.2 | −11.9 | 0.10 | 0.86 |
| `resid_nc_shm1e-6` | 4096 | 0 | 1e-6 | −14.5 | −14.7 | 0.06 | 0.91 |
| `resid_nc_hcos1e-5` | 4096 | 1e-5 | 0 | +93.3 | +113.2 | 0.99 | 0.00 |
| `resid_nc_hc` | 2048 | 1e-5 | 0 | +88.4 | +114.2 | 1.00 | 0.00 |
| `resid_nc_hm` | 2048 | 1e-5 | 1e-6 | +90.0 | +118.0 | 1.00 | 0.00 |

- The copy reproduces `resid_nc`: the control gives −11.2 against −13.1.
- `s_hcossim` alone flips the sign at width 4096. The bonus alone does not,
  and width does not change it: +93.3 at 4096 against +88.4 at 2048, both
  with `s_hcossim`.
- One seed (42) throughout, so none of this measures seed variance. Up to
  160k all six runs also share one data order.

Outputs: `<run>/headcoh_160000` for all six.

### Five runs at step 370k

Every run but the `s_Hm` arm, which stopped at 191.4k, reached 370k.
`headcoh.py` mean ζ_cos by layer, with head counts out of 160:

| run | L0 | L1 | L2 | L3 | L4 | mean | median | heads > +2 | heads < −2 |
|---|---|---|---|---|---|---|---|---|---|
| `resid_nc` | −8.1 | −28.3 | −18.1 | −18.8 | −14.4 | −17.5 | −28.1 | 6 | 154 |
| `resid_nc_ctl` | −7.6 | −28.0 | −17.6 | −14.0 | −18.8 | −17.2 | −24.7 | 8 | 151 |
| `resid_nc_hcos1e-5` | +78.7 | +118.8 | +128.0 | +121.7 | +127.3 | +114.9 | +121.7 | 160 | 0 |
| `resid_nc_hc` | +92.6 | +122.6 | +129.0 | +124.3 | +129.4 | +119.6 | +125.3 | 160 | 0 |
| `resid_nc_hm` | +90.8 | +122.5 | +132.5 | +125.8 | +131.3 | +120.6 | +126.9 | 160 | 0 |

- The 160k result holds. The control matches `resid_nc` in the mean
  (−17.2 against −17.5) and to within 0.6 at layers 0 to 2. At layers 3
  and 4 their values are swapped: −14.0 and −18.8 against −18.8 and
  −14.4. With `s_hcossim` alone, every head is above +2 at width 4096, as
  at 2048.
- Every run's mean is larger in magnitude than at 160k.
- In every run, layer 0 is the layer nearest zero. The `s_hcossim` arm's
  layer 0 is 12 to 14 below the other two positive runs'.
- After their first interruption the arms' data order no longer follows
  `resid_nc`'s (Runs, below), so at 370k the control matches `resid_nc`
  without having seen the same sequence of batches.

Outputs: `<run>/headcoh_370000` for these five.

### Runs

- All three arms ran out of GPU memory in `jit_update` at 12:30, within
  11 s of each other, at steps 175.4k (control), 166.8k (`s_hcossim`) and
  175.5k (`s_Hm`). The GPU is shared with other users' processes.
- The `s_Hm` arm is left stopped at 175.5k: `resid_nc_hc` answers its
  question.
- The control and the `s_hcossim` arm were resumed from their checkpoints
  at 14:17 so they can be scored at 370k as the README describes.
- The machine crashed and rebooted at 18:46, with the control at 296.9k
  and the `s_hcossim` arm at 284.5k. Each run's last two committed
  checkpoints came back with empty `_METADATA` files (296.8k-296.9k and
  284.4k-284.5k), the control also left a partial 297k checkpoint, and
  each `loss.csv` ended in NUL bytes where its last ~1,200 and ~1,000
  rows had been. The bad checkpoints, the partial one and the damaged
  `loss.csv` are kept in `<run>.crashed/` beside each run. Both runs
  resumed at 19:29 from 296.7k and 284.3k, restored and checked
  finite first. Their `loss.csv` rows for the lost steps (295,701-296,700
  and 283,501-284,300) are NaN, so a row's index is still its step.
- The `s_Hm` arm was resumed with them by mistake and stopped again at
  191.4k. Its 175.5k checkpoint rotated out; the 160k checkpoint scored
  above is kept.
- Both runs ran out of GPU memory again at 21:37, the control at 351.7k
  and the `s_hcossim` arm at 336.8k. Their systemd units restart a failed
  run five minutes later from its latest checkpoint. On 10 October the
  control ran out of memory twice more, at steps 430,898 (00:49) and
  445,538 (01:28), and resumed from 430,800 and 445,500. A restart at
  01:33 also ran out of memory before its first step.
- In the copy, a resumed run trains `--epochs` more epochs from where it
  resumed, and its data order and random draws start over. The copy's
  `TrainingEnv.init` builds a new loader whose sampler begins at its
  first index, and its `Hyperparams.train` makes a new `PRNGKey(42)`. So
  neither arm would have stopped at 372.6k. After each interruption, an
  arm repeated the start of `resid_nc`'s batch sequence rather than
  continuing it. At 370k the control was 18.3k steps past its last
  resume and the `s_hcossim` arm 33.2k. `resid_nc`, `resid_nc_hc` and
  `resid_nc_hm` each trained steps 0 to 372.6k without a resume.
  `log.jsonl` records each start's `resume_from`.
- This repository's loader has continued the batch stream and stopped at
  the run's final step since `c077140` (2026-10-03), which the copy
  predates. Its `Hyperparams.train` still starts every run from a new
  key. Since 10 October `train_old.py` resumes as this repository does
  (README, Run). Every resume above predates that.
- Both arms were stopped at 02:05 on 10 October, the control at 457.0k
  and the `s_hcossim` arm at 446.1k. Their 18 and 17 checkpoints past
  372.6k (374 MiB each, 12.8 GiB in all) are not used here.

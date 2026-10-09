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
- The three runs that have reached 370k all keep their 160k sign there,
  and grow in magnitude.
- One seed (42) and one data order throughout, so none of this measures
  seed variance.

Outputs: `<run>/headcoh_160000` for all six, `resid_nc_hc/headcoh_370000`.

### Runs

- All three arms ran out of GPU memory in `jit_update` at 12:30, within
  11 s of each other, at steps 175.4k (control), 166.8k (`s_hcossim`) and
  175.5k (`s_Hm`). The GPU is shared with other users' processes.
- The `s_Hm` arm is left stopped at 175.5k: `resid_nc_hc` answers its
  question.
- The control and the `s_hcossim` arm were resumed from their checkpoints
  at 14:17 so they can be scored at 370k as the README describes.

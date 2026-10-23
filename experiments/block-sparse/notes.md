# Block-sparse featurizer baseline: notes

## 2026-10-22: smoke run and launch

**Smoke run.** 2,000 steps at blocks of 4, 8 active, written to a scratch
directory:
- FVU_w was 0.791 at step 500 and 0.745 at step 2,000;
- L0 was 32, and no latent was dead;
- `pareto.py` reads it as 32 coefficients and 67 support bits, against
  TopK-32's 276.

At step 2,000 the converged TopK-32 run's log reads 0.652. Its `meta.json`
does not record a learning rate, but the sent16 replication of it ran at
the default, 1e-4.

**Launch.** The three runs of the README started together on the RTX 3090
as `sae-conv-bsf2-k16`, `sae-conv-bsf4-k8` and `sae-conv-bsf8-k4`, logging
to `experiments/logs/sae_conv_m5120_bsf*.log`. Each runs at about 16.7
steps a second beside the other two, about 2.4 hours for the 144,300 steps.

At the first eval (step 1,000), with L0 32 and no dead latents in each:

| run | FVU_w |
|---|---|
| `m5120_bsf2_k16` | 0.724 |
| `m5120_bsf4_k8` | 0.766 |
| `m5120_bsf8_k4` | 0.799 |
| TopK-32 (`m5120_k32`, from its log) | 0.684 |

`m5120_bsf4_k8` matches the smoke run's step-1,000 eval to every digit,
as the same seed and schedule should.

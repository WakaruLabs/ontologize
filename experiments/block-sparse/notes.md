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

## 2026-10-23: the converged runs

All three ran the 144,300 steps without a restart and finished at 00:56.
`pareto.py` scored them on the cache tail, on the CPU, beside TopK-32, its
seed-43 replica and the converged L1 sweep. The table is
`data/out/sonar/pareto/pareto_bsf/pareto.csv`, which also has the default
Ontologizer's points.

| run | coefficients | support bits | FVU_w | above TopK-32 |
|---|---|---|---|---|
| `m5120_k32` (TopK-32) | 32 | 276 | 0.4994 | |
| `m5120_k32_s43` | 32 | 276 | 0.4967 | |
| `m5120_bsf2_k16` | 32 | 137 | 0.5893 | 0.090 |
| `m5120_bsf4_k8` | 32 | 67 | 0.6459 | 0.146 |
| `m5120_bsf8_k4` | 32 | 33 | 0.6906 | 0.191 |
| `m5120_l10.0001` | 60 | 467 | 0.5520 | |
| `m5120_l10.0003` | 4 | 45 | 0.8393 | |

The other three L1 runs send 519 to 1,637 coefficients.

**Reconstruction.**
- At 32 coefficients, FVU_w rises with the block size: each halving of the
  support bits costs 0.045 to 0.090. The gaps to TopK-32 are 30 to 70
  times the spread between its two seeds, 0.003.
- The gaps grew during training. At step 1,000 they were 0.040, 0.083 and
  0.115. Over the run TopK-32's FVU_w fell by 0.184, and the block runs'
  by 0.134, 0.120 and 0.108.
- All four were still falling at the end, at about the same rate (0.006 to
  0.008 over the last 44,000 steps), so the gaps are not closing.
- Each block run's held-out FVU_w is 0.003 to 0.006 above its training
  slice's.

**Usage.** A block's latents fire together, so the effective number of
blocks in use is the exp-entropy of the latents' firing frequencies
(`use_ent`) divided by the block size. On the cache tail:

| run | effective units in use | dense units (on >10% of rows) |
|---|---|---|
| TopK-32 | 2,668 of 5,120 latents (52%) | 3 |
| blocks of 2 | 1,234 of 2,560 blocks (48%) | 6 |
| blocks of 4 | 274 of 1,280 blocks (21%) | 8 |
| blocks of 8 | 45 of 640 blocks (7%) | 8 |

- The blocks-of-8 run fell to 31 effective blocks at step 17,000 and has
  recovered slowly since.
- No latent was ever dead (1,000 training steps without firing), so every
  block stays in use, but not evenly.
- `pareto.py`'s support count, log2 C(G, L), treats every block as equally
  likely, which the runs at blocks of 4 and 8 are furthest from.

**Not settled: whether the support saved pays for the reconstruction
lost.** Each block run has a TopK counterpart with 12 to 16 more support
bits and far fewer coefficients:
- TopK-16 sends 16 coefficients and 153 support bits;
- TopK-8 sends 8 and 83;
- TopK-4 sends 4 and 45.

At more than a bit per coefficient, each costs less in total than the
blocks of 2, 4 and 8 respectively. So the blocks win on rate only if
TopK-16, TopK-8 and TopK-4 reconstruct worse than 0.589, 0.646 and 0.691.
No converged run of any of them exists on `mc4_4M`. Pricing the
coefficients themselves needs a block kind in `quantrate.py`.

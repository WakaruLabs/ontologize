# recipe-ladder: notes

## 2026-10-25: queued

**Smoke runs** (CPU, before queuing; their outputs were not kept):
- **The k32 SAE with all three noises** (`--sd-pre 0.3 --p-drop 0.1
  --sd-out 0.1`): 100 steps on the cache's first 262,144 rows, scored on
  8,192 tail rows.
  - Its FVU went from 0.741 at step 50 to 0.702 at step 100, at L0 32 with
    no dead latents. The stock run on the same rows read 0.740 and 0.701.
  - `meta.json` records the three noises.
- **The direct Ontologizer** with `--sd-f 0 --s-l1f 0 --max-to-keep 1
  --checkpoint-each 50000`, 500 steps.
  - Its loss fell from 3.95e-5 to 2.23e-5, and its whitened MSE from
    9.4e-6 to 5.4e-6.
  - `log.jsonl` records `sd_F` 0 and `s_L1F` 0. One checkpoint remained,
    1.1 GB.

**The SAE arms.** Unit `sae-noise` was started 2026-10-25 22:51. It
waits for `sent16-med` to finish, then for an idle GPU, and writes
`data/out/sonar/sae_sent16/m5120_k32_lr4e-4_no837_whole_eps1e-15_{sdpre0.3,pdrop0.1,sdout0.1,allnoise,s43}`.

**The Ontologizer arms** are not started yet:
- `ladder.sh` takes the MSE weights, which are chosen once `sent16-med`
  is scored: `med837` if the median weight stops the drift without
  costing FVU over the free coordinates, else `no837`.
- The 20 arms need about 17 GB. 52 GB was free at the start, and
  `sent16-med` will take about 38 GB of it.

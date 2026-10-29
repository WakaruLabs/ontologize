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

## 2026-10-28: the SAE noises do not move the k32

- **Where:** `sae-noise` finished 2026-10-28 02:16, with no errors or NaN
  in any arm's log.
- **Scoring**, on the CPU: `coordfvu.py`, on the 32,765 tail rows under
  the masked weights (`data/out/sonar/coordfvu_sent16_saenoise/`), and
  `quantrate.py` (`data/out/sonar/quantrate_sent16_med837_saenoise/`, with
  `experiments/sentence-split/notes.md`'s command). Dead latents come from
  `sae.py`'s final eval.

| arm | FVU | dead | bits to 0.5934 | bits to 0.4581 |
|---|---|---|---|---|
| control | 0.4180 | 33 | 120 | 240 |
| seed 43 | 0.4183 | 33 | 120 | 240 |
| + selection noise 0.3 | 0.4195 | 2 | 124 | 264 |
| + winner dropout 0.1 | 0.4167 | 5 | 124 | 246 |
| + output noise 0.1 | 0.4179 | 24 | 119 | 239 |
| all three | 0.4206 | 1 | 131 | 268 |

The targets: 0.5934 is the direct `med837` run's pinned FVU, and 0.4581 its
best checkpoint's.

- **The fit barely moves.**
  - No noise changes the FVU by more than 0.003. The seed pair differs by
    0.0003, which is one pair, so that is a weak null.
  - Winner dropout is the only one that lowers it (−0.0013).
  - Selection noise and all three together raise it (+0.0015, +0.0026),
    and cost 10–12% more bits to reach 0.4581.
- **Selection noise and winner dropout keep latents alive.** They take
  the dead count from 33 to 2 and 5, and all three together to 1. Output
  noise leaves 24.
- **Noise does not explain the gap to the Ontologizers.** With all three
  noises the k32 reaches the direct run's 0.5934 at 131 bits, against
  that run's 1,875.

## 2026-10-28: the Ontologizer arms, under `med837`

`sent16-med` showed the median weight holding coordinate 837 at 0.1–0.4×
the median coordinate's error, and improving the FVU over the free
coordinates in both architectures:
- hard stack 0.3406 against 0.3508;
- direct 0.5685 against 0.6538 (`experiments/sentence-split/notes.md`,
  2026-10-28).

By the rule set at queuing, the arms train under it. Unit `recipe-ladder`
started 2026-10-28 22:15 with `ladder.sh med837`, with 332 GB of disk
free.

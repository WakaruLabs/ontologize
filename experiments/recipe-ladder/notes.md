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

## 2026-10-30: two auxiliary weights cost most of the fit

- **Where:** `recipe-ladder` finished 2026-10-30 05:29. All 20 arms ran
  50,000 steps, with no NaN in any `loss.csv`, and each kept only its final
  checkpoint (557 MB hard stack, 1.1 GB direct).
  - The hard stack's `nopdrop` arm hit a GPU out-of-memory error at step
    19,734 (2026-10-29 12:30), as `sent16-med` did. The unit restarted it,
    and it resumed from its rolling save at step 19,500.
- **Scoring**, on the CPU, on the 32,765 tail rows:
  - `coordfvu.py --coord 837` under the masked weights
    (`data/out/sonar/coordfvu_recipe_ladder/`);
  - `coordtrace.py --coord 837 --steps 50000`
    (`data/out/sonar/coordtrace/<arm>/`);
  - `auxpull.py --coord 837 --batches 8` under the training weights
    (`med837`), on each control and auxiliary-weight arm
    (`data/out/sonar/auxpull_recipe_ladder/`);
  - `quantrate.py`, with the 2026-10-24 command and the k32, k160 and L1
    1.39e-6 SAEs at ε 1e-15 (`data/out/sonar/quantrate_recipe_ladder/`).

  The logs are `experiments/logs/{coordfvu,coordtrace,auxpull,quantrate}_recipe_ladder.log`.
  The collapse statistics are means over each run's last 1,000 steps of
  `loss.csv`.

### Held-out FVU over the 1023 free coordinates

| arm | hard stack: FVU, Δ | direct: FVU, Δ |
|---|---|---|
| control | 0.5838 | 0.4608 |
| seed 43 | 0.5859, +0.0020 | 0.4590, −0.0018 |
| no logit noise (`--sd-k 0`) | 0.5778, −0.0060 | 0.4584, −0.0024 |
| no feature noise (`--sd-f 0`) | 0.5871, +0.0033 | 0.4615, +0.0007 |
| no winner dropout (`--p-drop 0`) | 0.5839, +0.0000 | 0.4610, +0.0002 |
| no noise | 0.5843, +0.0005 | 0.4601, −0.0007 |
| no sample similarity (`--s-bcossim 0`) | **0.2682, −0.3156** | 0.3369, −0.1239 |
| no support overlap (`--s-support 0`) | 0.4280, −0.1558 | **0.2003, −0.2605** |
| no output L1 (`--s-l1f 0`) | 0.5772, −0.0066 | 0.4421, −0.0187 |
| no mean-entropy bonus (`--s-hm 0`) | 0.6510, +0.0672 | 0.4197, −0.0411 |

- **The seed pairs differ by 0.0020 and 0.0018.** Each arm is one seed.
- **The noises do not matter.**
  - No noise moves either architecture by more than 3× the seed pair's
    difference, and removing all three moves neither by more than 0.0007.
  - This matches the SAE arms (2026-10-28).
- **`s_bcossim` and `s_support` cost most of the fit.**
  - Without `s_bcossim` the hard stack reads 0.2682 at 50,000 steps. The
    full 344,250-step run under `med837` read 0.3406
    (`experiments/sentence-split/notes.md`, 2026-10-28).
  - Without `s_support` the direct stack reads 0.2003, the lowest
    held-out FVU recorded for any Ontologizer on `mc4_sent16`. The k160 SAE
    reads 0.1872.
  - Each costs both architectures. Which costs more depends on the
    architecture.
- **`s_L1F` costs a little.** Without it the direct stack's FVU falls by
  0.0187, 10× the seed pair's difference; the hard stack's falls by 0.0066.
- **`s_Hm` helps the hard stack and costs the direct one** (+0.0672 and
  −0.0411 without it). Its effect on the rate is below.

### Collapse

| arm | hard stack: `cossim_k_max`, `KL_m` | direct: `cossim_k_max`, `KL_m` |
|---|---|---|
| control | 0.61, 0.68 | 0.995, 2.05 |
| seed 43 | 0.60, 0.69 | 0.997, 2.05 |
| no noise | 0.61, 0.68 | 0.90, 2.05 |
| no sample similarity | 0.998, 0.48 | 0.96, 2.05 |
| no support overlap | 0.48, 0.66 | **0.046**, 2.05 |
| no output L1 | 0.62, 0.69 | 0.90, 2.05 |
| no mean-entropy bonus | 0.58, 17.5 | 0.26, 11.8 |

- **The direct run's head collapse comes from `s_support`.** Without it,
  `cossim_k_max` is 0.046 against the control's 0.995.
- **Without `s_bcossim` the hard stack has a head collapse, and its best
  fit.** `cossim_k_max` reads a single pair of heads, so one collapsed
  pair does not show in the FVU.
- **The noise arms' `cossim_k_max`** stays within 0.1 of the control's in
  both architectures (the direct run's lowest is `nonoise`'s 0.90).
- **Without `s_Hm`, entry usage concentrates** (`KL_m` 17.5 and 11.8,
  against 0.68 and 2.05). The rate falls with it (below).

### Gradient pulls at step 50,000 (`auxpull.py`)

Each term's configured weight × its gradient norm over the MSE's, on each
parameter group of the control, under the training weights:

| term (weight) | hard stack: classifier, dictionary | direct: classifier, dictionary |
|---|---|---|
| `cossim_b` (1e-5) | 4.71, 2.28 | 0.29, 0.13 |
| support (1e-6) | 0.89, 1.16 | 0.08, 0.67 |
| `KL_m` (2e-6) | 2.78, — | 9.83, — |
| `L1_F` (1e-9) | 0.04, 0.03 | 0.03, 0.01 |

- **In each architecture, the costliest weight has the largest pull of
  the terms other than `KL_m`:**
  - `cossim_b` on the hard stack's classifier, at 4.7× the MSE;
  - the support term on the direct stack's dictionary, at 0.67×.
- **`KL_m` pulls on the classifiers at 2.8× and 9.8× the MSE**, yet
  removing it moves the FVU by 0.067 and 0.041, less than removing either
  of the two. So the size of a pull at the end of training does not rank a
  term's cost.
- **Without `s_Hm`, the hard stack's dictionary pulls blow up:**
  `cossim_b`'s reaches 564× the MSE's and the support term's 148×.
- **Coordinate 837**, in the direct run. With 837 held:
  - the support term still carries 95% of the auxiliary pull on 837's atom
    components (0.60× the MSE's on the median coordinate);
  - without the support term, the net auxiliary pull on 837 is 13-fold
    smaller (4.8e-9 against 6.2e-8).

  The hard stack's decoder row for 837 again gets nothing from any
  auxiliary term.
- **837 is held in every arm.** Its error is 0.1–0.4× the median
  coordinate's, and its residual share is at most 0.001. The direct
  `nobcos` arm's atom components on 837 spread to 41× the median
  coordinate's (0.5× in the control), without an error above 0.2×.

### Operational rate

| code | float FVU | pinned: bits, FVU | k32: bits to the pinned FVU | k160 | L1 1.39e-6 |
|---|---|---|---|---|---|
| hard stack, control | 0.5838 | 1878, 0.6065 | 111 | 142 | 125 |
| seed 43 | 0.5859 | 1877, 0.6083 | 110 | 141 | 124 |
| no sample similarity | 0.2682 | 1896, **0.2968** | — | 606 | 736 |
| no support overlap | 0.4280 | 1882, 0.4527 | 247 | 280 | 286 |
| no mean-entropy bonus | 0.6510 | **496**, 0.6797 | 74 | 101 | 82 |
| direct, control | 0.4608 | 1888, 0.4874 | 207 | 240 | 237 |
| seed 43 | 0.4590 | 1888, 0.4857 | 208 | 241 | 240 |
| no sample similarity | 0.3369 | 1888, 0.3649 | — | 430 | 469 |
| no support overlap | 0.2003 | 1889, **0.2308** | — | 867 | 1469 |
| no output L1 | 0.4421 | 1889, 0.4686 | 227 | 261 | 262 |
| no mean-entropy bonus | 0.4197 | **1306**, 0.4459 | 257 | 290 | 297 |

The noise arms' pinned rates are all 1878–1889 bits. "—" is an SAE whose
curve never reaches that FVU (the k32's floor is 0.418).

- **The gap narrows, but the SAEs still lead.**
  - The direct stack without `s_support` sends 1,889 bits for 0.2308. The
    k160 reaches that at 867 bits, which is 46% of the stack's bits, and
    the L1 SAE at 1,469 (78%).
  - The hard stack without `s_bcossim`: the SAEs need 32–39% of its bits.
  - Under the full recipe at 50,000 steps, the SAEs need 6–8% of the hard
    control's bits and 11–13% of the direct control's. They need 22–31% of
    the full 344,250-step hard stack's (2026-10-28).
- **Without `s_Hm`, the hard stack uses few entries.** It realizes 496 of
  its 1,900 nominal bits, and the direct stack 1,306 of 1,890.
- **What this leaves open.**
  - The arms train for 50,000 steps on the cache's first 12.8M rows. The
    SAEs train for 144,300 steps over the whole cache.
  - Each arm drops one weight. Whether the two costliest weights compound
    is not tested; nor is a longer run of either arm.

## 2026-10-30: queued, both weights off and both rescaled

- **What.** Unit `recipe-ladder` was started again 2026-10-30 22:24 with
  `ladder.sh med837 nobcos_nosupp bcos2e-7_supp2e-8 bcos2e-7 supp2e-8`.
  Each arm runs in both architectures, so 8 runs, about 12 GPU-hours at the
  first queue's pace.
- **Why these.**
  - `nobcos_nosupp` tests whether the two weights' costs add.
  - The rescaled arms divide each weight by 50, `s_Hm`'s factor (README).
    Against the control and the matching off arm, they tell a term that
    costs fit at any weight from one that is only too heavy for this
    cache's loss scale.
- **Disk.** 283 GB free at the start; the 8 runs take about 7 GB.

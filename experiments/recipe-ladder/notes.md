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
  - The arms train for 50,000 steps of 256 on the cache's first 12.8M rows.
    The SAEs train over the whole cache, for 144,300 steps of 4,096 (the
    k32 and L1 SAEs) or 336,000 (the k160).
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

## 2026-11-02: both weights off is best; `s_bcossim` costs even rescaled

- **Where:** the queue ran 2026-10-30 22:24 to 2026-10-31 10:10, with no
  restarts. All 8 runs reached 50,000 steps with no NaN, and each
  `log.jsonl` records the intended `s_bcossim` and `s_support`.
- **Scoring**, as in the 2026-10-30 section. The outputs:
  - `data/out/sonar/coordfvu_recipe_ladder2/`;
  - `coordtrace/<arm>/`;
  - `auxpull_recipe_ladder/`;
  - `quantrate_recipe_ladder2/`.

  The logs are `experiments/logs/*_recipe_ladder2.log`.
- **The first `quantrate` run stopped partway** (2026-10-31 20:15), with no
  out-of-memory kill in the journal. It was rerun from the start as
  systemd unit `score-ladder2` (2026-11-02), and its log holds both
  attempts.
- The control and single-removal rows below are the 2026-10-30 section's.

### Held-out FVU over the 1023 free coordinates

| arm | hard stack | direct |
|---|---|---|
| control | 0.5838 | 0.4608 |
| `s_bcossim` off | 0.2682 | 0.3369 |
| `s_bcossim` 2e-7 (÷50) | 0.3429 | 0.3957 |
| `s_support` off | 0.4280 | 0.2003 |
| `s_support` 2e-8 (÷50) | 0.4328 | 0.2024 |
| both off | **0.2373** | **0.1959** |
| both ÷50 | 0.2585 | 0.2002 |

The seed pairs differ by 0.0020 and 0.0018.

- **`s_support` at ÷50 reads as if it were removed.** The rescaled arm is
  within 0.005 of the off arm in both architectures.
- **`s_bcossim` costs even at ÷50.**
  - Against removing it, the rescaled arm is +0.075 in the hard stack
    (0.3429 against 0.2682) and +0.059 in the direct one.
  - So it is not only too heavy for this cache's loss scale.
- **The two removals add in the hard stack.** With both off it reads
  0.2373, against 0.2682 with `s_bcossim` alone off.
- **In the direct stack, `s_support` carries nearly all of it.** Both off
  is 0.0044 below `s_support` off, 2.4× the seed pair's difference.
- **Both ÷50** is 0.021 (hard) and 0.0043 (direct) above both off.
- **Against the SAEs:** the direct stack with both off is 0.0087 above
  the k160 SAE's 0.1872, and below the L1 1.39e-6 SAE's 0.2138.
- **Coordinate 837 is held in every arm,** at 0.1–0.3× the median
  coordinate's error. Where `s_bcossim` is lowered but `s_support` stays,
  the direct atoms' 837 components spread out:
  - 15.8× the median coordinate's at ÷50, and 41× with `s_bcossim` off;
  - 0.14× once `s_support` is also off or rescaled;
  - in the `s_bcossim` ÷50 arm, the support term carries 98% of the
    auxiliary pull on them.

### Collapse

Means over the last 1,000 steps of `loss.csv`.

| arm | hard stack: `cossim_k_max`, `KL_m` | direct: `cossim_k_max`, `KL_m` |
|---|---|---|
| control | 0.61, 0.68 | 0.995, 2.05 |
| `s_bcossim` off | 0.998, 0.48 | 0.96, 2.05 |
| `s_bcossim` ÷50 | 0.89, 0.48 | 0.84, 2.05 |
| `s_support` off | 0.48, 0.66 | 0.046, 2.05 |
| `s_support` ÷50 | 0.49, 0.67 | 0.18, 2.05 |
| both off | 0.45, 0.48 | 0.002, 2.05 |
| both ÷50 | 0.74, 0.48 | 0.17, 2.05 |

- **The direct run's head collapse follows `s_support`.** `cossim_k_max`
  is 0.995 at full weight, 0.18 at ÷50, 0.046 off, and 0.002 with both
  weights off.
- **In the hard stack, `KL_m` falls from 0.68 to 0.48** whenever
  `s_bcossim` is off or rescaled, so entry usage is more even.

### Gradient pulls at step 50,000 (`auxpull.py`)

Each term's configured weight × its gradient norm over the MSE's, on the
classifier / the dictionary:

| arm | hard stack: `cossim_b`, support | direct: `cossim_b`, support |
|---|---|---|
| control | 4.71 / 2.28, 0.89 / 1.16 | 0.29 / 0.13, 0.08 / 0.67 |
| `s_bcossim` ÷50 | 0.08 / 0.03, 0.09 / 0.50 | 0.06 / 0.007, 0.10 / 0.16 |
| `s_support` ÷50 | 3.81 / 1.52, 0.05 / 0.04 | 0.21 / 0.09, 0.001 / 0.003 |
| both ÷50 | 0.06 / 0.03, 0.01 / 0.02 | 0.006 / 0.003, 0.001 / 0.004 |

- **At ÷50, `s_bcossim`'s pull is at most 0.08 of the MSE's.** It still
  costs the hard stack 0.075 against removing it. A small pull at the end
  of training does not make a term harmless.
- **Rescaling `s_bcossim` lowers the support term's pull too.** In the hard
  stack's classifier it falls from 0.89 to 0.09, at an unchanged weight.

### Operational rate

| code | float FVU | pinned: bits, FVU | k32: bits to the pinned FVU | k160 | L1 1.39e-6 |
|---|---|---|---|---|---|
| hard stack, both off | 0.2373 | 1896, **0.2650** | — | 715 | 959 |
| hard stack, both ÷50 | 0.2585 | 1896, 0.2859 | — | 641 | 801 |
| hard stack, `s_bcossim` ÷50 | 0.3429 | 1897, 0.3708 | — | 417 | 453 |
| hard stack, `s_support` ÷50 | 0.4328 | 1881, 0.4573 | 241 | 274 | 279 |
| direct, both off | 0.1959 | 1889, **0.2265** | — | 891 | 1599 |
| direct, both ÷50 | 0.2002 | 1889, 0.2307 | — | 867 | 1472 |
| direct, `s_bcossim` ÷50 | 0.3957 | 1889, 0.4217 | 325 | 324 | 339 |
| direct, `s_support` ÷50 | 0.2024 | 1889, 0.2329 | — | 856 | 1415 |

- **The closest any Ontologizer has come is the direct stack with both
  weights off:** 0.2265 at 1,889 bits.
  - The k160 SAE reaches that at 891 bits (47% of the stack's bits), and
    the L1 SAE at 1,599 (85%).
  - The 50,000-step direct control is at 11–13%, and `s_support` off at
    46–78% (2026-10-30).
- **The hard stack with both off:** the SAEs need 38–51% of its bits.
- **What this leaves open.**
  - One seed per arm.
  - 50,000 steps of 256 on the cache's first 12.8M rows, against the SAEs'
    144,300 or 336,000 steps of 4,096 over the whole cache. Whether the
    both-off stacks keep improving with more steps is not tested.
  - `s_L1F` is also unrescaled. Removing it took 0.019 off the direct
    stack's FVU (2026-10-30), and it has not been removed together with
    these two.

## 2026-11-02: queued, `s_L1F` off as well

Unit `recipe-ladder` was started 2026-11-02 15:56 with
`ladder.sh med837 nobcos_nosupp_nol1f`: `s_bcossim`, `s_support` and
`s_L1F` all at 0, in both architectures, against `nobcos_nosupp`. The
noises and `s_Hm` 2e-6 are as in the control. Two runs, about 3 GPU-hours;
277 GB free at the start.

## 2026-11-03: with `s_L1F` off as well, the direct stack's FVU is below the k160 SAE's

- **Where:** the queue ran 2026-11-02 15:56 to 18:36, with no restarts.
  Both runs reached 50,000 steps with no NaN in the live columns. Each
  `log.jsonl` records `s_bcossim`, `s_support` and `s_L1F` at 0, `s_Hm` at
  2e-6 and ε at 1e-15.
- **Scoring**, as in the 2026-10-30 section, ran as systemd unit
  `score-ladder3` on 2026-11-03. The outputs:
  - `data/out/sonar/coordfvu_recipe_ladder3/`;
  - `coordtrace/<arm>/`;
  - `auxpull_recipe_ladder/`;
  - `quantrate_recipe_ladder3/`.

  The logs are `experiments/logs/*_recipe_ladder3.log`.
- The other rows below come from the 2026-10-30 and 2026-11-02 sections.

### Held-out FVU over the 1023 free coordinates

| arm | hard stack | direct |
|---|---|---|
| control | 0.5838 | 0.4608 |
| `s_L1F` off | 0.5772 | 0.4421 |
| `s_bcossim`, `s_support` off | 0.2373 | 0.1959 |
| all three off | **0.1869** | **0.1806** |

The seed pairs differ by 0.0020 and 0.0018. On the same rows, the k160
SAE reads 0.1872 and the L1 1.39e-6 SAE 0.2138.

- **With the other two weights off, removing `s_L1F` helps both
  architectures.**
  - The hard stack improves by 0.050 (0.2373 to 0.1869), 25× its seed
    pair's difference.
  - The direct stack improves by 0.015 (0.1959 to 0.1806), 8.4× its seed
    pair's difference.
- **In the hard stack, `s_L1F`'s cost shows only once the other two weights
  are off.** Removing it alone improves the hard stack by 0.0067, and the
  direct stack by 0.019.
- **Against the SAEs:**
  - The direct stack's FVU is 0.0066 below the k160's, 3.6× its seed pair's
    difference. That makes it the first Ontologizer on `mc4_sent16` with a
    lower float FVU than the k160.
  - The hard stack's is 0.0002 below the k160's, well within its seed
    spread.
  - Both are below the L1 1.39e-6 SAE's. The L1 4.64e-7 SAE, left out of
    this scoring, reads 0.0526 (2026-11-04).
  - Each arm has one seed. The arms train for 50,000 steps of 256 on the
    cache's first 12.8M rows, the SAEs for 144,300 or 336,000 steps of
    4,096 over the whole cache.
- **`s_L1F` pulled little at the end of training, but removing it still
  helped.** In the both-off arms at step 50,000, its pull relative to the
  MSE's was:
  - 0.024 on the classifier and 0.027 on the dictionary in the hard stack;
  - 0.074 and 0.022 in the direct stack.

  This repeats the case of `s_bcossim` at ÷50 (2026-11-02): a small pull at
  the end of training does not make a term harmless.
- **Coordinate 837 is held in both stacks,** at 0.13× (hard) and 0.12×
  (direct) the median coordinate's error.
- **In the hard stack, coordinate 652 gets worse as weights come off.**
  - It has the largest MSE weight (13.0, 60× the next largest), and its
    standard deviation is 0.12× the median coordinate's.
  - Its FVU over the median coordinate's is:
    - 2.4× in the control;
    - 12.9× with both weights off;
    - 20.4× with all three off.
  - With all three off it holds 2.0% of the residual, and its own FVU is
    3.8: the stack predicts it worse than its mean does.
  - The direct stack holds it at 6.0×.

### Collapse and output L1

Means over the last 1,000 steps of `loss.csv`.

| arm | hard stack: `cossim_k_max`, `KL_m`, `L1_F` | direct: `cossim_k_max`, `KL_m`, `L1_F` |
|---|---|---|
| control | 0.61, 0.68, 86 | 0.995, 2.05, 166 |
| both off | 0.45, 0.48, 543 | 0.002, 2.05, 338 |
| all three off | 0.57, 0.50, 1327 | 0.006, 2.05, 425 |

- **Output L1 grows** 2.4× in the hard stack and 1.26× in the direct one.
- **The hard stack's worst head pair** rises from 0.45 to 0.57. That is
  still below the control's 0.61.
- **The direct stack's heads stay apart.**

### Gradient pulls at step 50,000 (`auxpull.py`)

With all three off, `s_Hm` is the only auxiliary weight left. Its pull on
the classifier, relative to the MSE's, is:

| arm | hard stack | direct |
|---|---|---|
| control | 2.78 | 9.83 |
| `s_L1F` on (both off) | 2.85 | 15.3 |
| all three off | 2.85 | 14.4 |

### Operational rate

| code | float FVU | pinned: bits, FVU | k32: bits to the pinned FVU | k160 | L1 1.39e-6 |
|---|---|---|---|---|---|
| hard stack, all three off | 0.1869 | 1894, **0.2096** | — | 1009 | — |
| direct, all three off | 0.1806 | 1889, **0.2073** | — | 1030 | — |

"—" marks an SAE whose curve never reaches that FVU.

- **To reach the pinned FVUs, the k160 needs about half the stacks' bits.**
  - It reaches them at 1,009 bits, 53% of the hard stack's, and at 1,030,
    55% of the direct stack's.
  - With `s_L1F` on, those were 38% and 47% (2026-11-02).
- **At 1,900 bits, the direct stack is ahead of the k160.** The pinned code
  sends no gains. Sending them costs the direct stack 11 more bits (1,899.7
  bits, FVU 0.1829). At 1,900 bits:
  - the direct stack reads 0.1828;
  - the k160 reads 0.1872;
  - the hard stack reads 0.1971;
  - the L1 1.39e-6 SAE reads 0.2202.

  The L1 4.64e-7 SAE, left out of this scoring, reads 0.1421 and is ahead
  of them all (2026-11-04).
- **Neither pinned FVU is in the L1 1.39e-6 SAE's reach.** Its curve gets
  no lower than 0.2138, at 3,631 bits, the highest rate its sweep tries.
- **What this leaves open.**
  - One seed per arm. A seed-43 replica would put a spread under the k160
    comparisons.
  - Whether the all-three-off stacks keep improving past 50,000 steps.
  - `s_Hm`, the one weight left, pulls hardest. Removed alone, it cost the
    hard stack 0.067, which then realized 496 of its 1,900 nominal bits.
    It helped the direct stack by 0.041 (2026-10-30). It has not been
    removed with the other three.

## 2026-11-03: queued, `s_Hm` off as well

Unit `recipe-ladder` was started 2026-11-03 03:13 with
`ladder.sh med837 nobcos_nosupp_nol1f_nohm`. With `s_bcossim`, `s_support`,
`s_L1F` and `s_Hm` all at 0, no auxiliary weight is left, and the MSE is
the whole loss. The three training noises are as in the control. The arm
runs in both architectures, against `nobcos_nosupp_nol1f`: two runs, about
2.5 GPU-hours. 274 GB was free at the start.

## 2026-11-04: with no auxiliary weight, both stacks' FVUs are below the k160 SAE's

- **Where:** the queue ran 2026-11-03 03:13 to 06:07. Both runs reached
  50,000 steps with no NaN in the live columns. Each `log.jsonl` records
  all four weights at 0, and `loss` equals `MSE` on every row of `loss.csv`.
- **The hard stack's run stopped once, at step 7,681, and resumed.** Its log
  has no traceback, and the journal has no entry for it, so the cause is
  not known. It resumed from the rolling save at 7,500, and its loss at
  step 7,681 matches the first attempt's (3.08e-6). The direct run went
  straight through.
- **Scoring**, as in the 2026-10-30 section, ran as systemd unit
  `score-ladder4` on 2026-11-04. The outputs:
  - `data/out/sonar/coordfvu_recipe_ladder4/`;
  - `coordtrace/<arm>/`;
  - `auxpull_recipe_ladder/`;
  - `quantrate_recipe_ladder4/`.

  The logs are `experiments/logs/*_recipe_ladder4.log`.
- **A manual restart of the machine cut off the first `quantrate` run.** It
  was rerun alone, from the start, and its log holds both attempts.
- **`auxpull.py` has no pull to report:** with every weight at 0, each
  term's applied pull is 0.
- The other rows below come from the earlier sections.

### Held-out FVU over the 1023 free coordinates

| arm | hard stack | direct |
|---|---|---|
| control | 0.5838 | 0.4608 |
| `s_Hm` off | 0.6510 | 0.4197 |
| `s_bcossim`, `s_support`, `s_L1F` off | 0.1869 | 0.1806 |
| all four off | **0.1681** | **0.1611** |

The seed pairs differ by 0.0020 and 0.0018. On the same rows, the k160
SAE reads 0.1872, the L1 1.39e-6 SAE 0.2138 and the L1 4.64e-7 SAE 0.0526.

- **With the other three weights off, removing `s_Hm` helps both
  architectures.**
  - The hard stack improves by 0.019, 9.4× its seed pair's difference.
    Removed alone, `s_Hm` cost it 0.067.
  - The direct stack improves by 0.019, 10.6× its seed pair's difference.
- **Both stacks' FVUs are below the k160's:** the hard stack's by 0.019
  (9.5× its seed pair's difference), and the direct stack's by 0.026 (14×).
- **Against the control,** the MSE alone leaves 0.29× the hard stack's FVU
  and 0.35× the direct stack's.
- **Coordinate 837 is held,** at 0.16× (hard) and 0.10× (direct) the median
  coordinate's error.
- **Coordinate 652 partly recovers.** Its FVU over the median coordinate's
  falls from 20.4× to 7.8× in the hard stack and from 6.0× to 3.9× in the
  direct one. Its own FVU is 1.32 and 0.64, so the hard stack still
  predicts it worse than its mean does.

### Collapse and output L1

Means over the last 1,000 steps of `loss.csv`.

| arm | hard stack: `cossim_k_max`, `KL_m`, `L1_F` | direct: `cossim_k_max`, `KL_m`, `L1_F` |
|---|---|---|
| `s_Hm` off | 0.58, 17.5, 67 | 0.26, 11.8, 155 |
| `s_bcossim`, `s_support`, `s_L1F` off | 0.57, 0.50, 1327 | 0.006, 2.05, 425 |
| all four off | 0.56, 0.55, 1356 | 0.010, 2.23, 431 |

- **Entry usage stays even without `s_Hm`.**
  - Removed alone, it sends `KL_m` to 17.5 and 11.8. With all four off,
    `KL_m` is 0.55 and 2.23.
  - So the uneven usage of the `s_Hm`-off arm needs at least one of the
    other three weights on.
  - The rates agree. With all four off, the stacks realize 1,891 of 1,900
    and 1,877 of 1,890 nominal bits. With `s_Hm` off alone, they realize
    496 and 1,306.

### Operational rate

| code | float FVU | pinned: bits, FVU | FVU at 1,900 bits | at 1,900 bits, subset rate |
|---|---|---|---|---|
| hard stack, all four off | 0.1681 | 1891, 0.1893 | 0.1722 | 0.1893 |
| direct, all four off | 0.1611 | 1877, 0.1865 | **0.1612** | 0.1638 |
| L1 4.64e-7 | 0.0526 | — | **0.1421** | **0.1423** |
| k160 | 0.1872 | — | 0.1872 | 0.1874 |
| L1 1.39e-6 | 0.2138 | — | 0.2202 | 0.2201 |

The k32's curve stops at 491 bits, at FVU 0.4180, so it has no reading at
1,900 bits. The L1 4.64e-7 row comes from
`data/out/sonar/quantrate_sent16_eps1e-15/`, under the same settings: the
k160's curve there matches this run's to 4e-8. It was left out of this
ladder's scoring, which passed `quantrate.py` only the k32, the k160 and
the L1 1.39e-6.

- **At 1,900 bits on average per row, the L1 4.64e-7 SAE leads.**
  - It reads 0.1421, against the direct stack's 0.1612 and the hard
    stack's 0.1722.
  - It reaches the direct stack's 0.1612 at 1,682 bits and the hard stack's
    0.1722 at 1,575.
- **Under a fixed budget of 1,900 bits per row, the direct stack leads
  these SAEs.** The k256 SAE, trained later, does not leave that lead
  standing (next results section).
  - Its code has a fixed length, and it reads 0.1638 at 1,900 nominal bits.
  - Held to that budget on every row (`rowbudget.csv`), the L1 4.64e-7 SAE
    reads 0.1742 and the k160 0.1867. The L1 SAE needs a budget of 2,010
    bits to reach 0.1638. (Exact budgets, from `rowbudget.py`: see the next
    results section.)
  - The hard stack's gains start at 1,906 nominal bits, so at 1,900 it
    reads its pinned 0.1893, behind both SAEs.
- **Both stacks are below the k160 at the mean rate:** the direct stack by
  0.026 and the hard stack by 0.015.
- **The gains cost few bits.**
  - The direct stack goes from 0.1865 at 1,877 bits (pinned) to 0.1633 at
    1,888 and 0.1612 at 1,898.
  - The hard stack goes from 0.1893 at 1,891 bits to 0.1699 at 1,902.
- **To reach the pinned FVUs:**
  - The k160 reaches the hard stack's 0.1893 at 1,355 bits, 72% of the
    stack's (1,646 at the subset rate, 87%).
  - It does not reach the direct stack's 0.1865. Its floor is 0.0007
    above that, less than the seed pair's difference, so the two are level
    at that point.
  - The L1 4.64e-7 SAE reaches them at 1,427 and 1,450 bits, 75% and 77%
    of the stacks' bits.
  - The L1 1.39e-6 SAE reaches neither.
- **What this comparison does not show.**
  - The k160 is saturated at this rate: it reads 0.1872 from about 1,710
    bits up, so it has bits it cannot use. A larger k has not been trained
    on `mc4_sent16`, so the fixed-budget lead is over a top-k code that
    cannot spend the budget.
  - The stacks have no code below about 1,880 bits, so only the SAEs have
    a reading at lower rates.
  - The rates leave out the coders' tables.
  - One seed per arm. The arms train for 50,000 steps of 256 on the cache's
    first 12.8M rows, the SAEs for 144,300 or 336,000 steps of 4,096 over
    the whole cache.
- **What this leaves open.**
  - The three noises were tested only under the full recipe, where the
    weights dominated the fit. Whether they help or hurt the MSE-only runs
    is not tested.
  - A seed replica of this arm, and a longer run of it.
  - An SAE of larger k at this rate.

## 2026-11-04: queued, a seed replica and a k256 SAE

Three systemd units, in series, so nothing shares the GPU:
- **`recipe-ladder`:** `ladder.sh med837 nobcos_nosupp_nol1f_nohm_s43`, the
  all-four-off arm at seed 43 in both architectures. Two runs, about 2.5
  GPU-hours. It puts a spread under this arm's comparisons, which so far
  lean on the controls' seed pairs.
- **`sae-k256`:** `sae_k256_run.sh`, the k160 with `--topk 256`, once
  `recipe-ladder` has finished. A 30-step smoke run went at 13.4 steps per
  second, so about 7 GPU-hours.
- **`score-ladder5`:** `coordfvu.py` and `coordtrace.py` on the replicas,
  then `quantrate.py` on both seeds against the k32, k160, k256, L1 1.39e-6
  and L1 4.64e-7 SAEs. It waits for the two units above.

343 GB was free at the start.

## 2026-11-04: the seeds agree, and the k256 SAE leads the stacks at 1,900 bits

- **Where:** the three units ran 2026-11-04 01:59 to 12:18.
  - Both replicas reached 50,000 steps with no resume and no NaN in the live
    columns, and `loss` equals `MSE` on every row.
  - The k256 ran 04:49 to 11:28 and reached 336,000 steps, with 47 of its
    11,264 latents dead.
- **The outputs:**
  - `data/out/sonar/coordfvu_recipe_ladder5/`;
  - `coordtrace/<arm>/`;
  - `quantrate_recipe_ladder5/`, which now has all five SAEs and both seeds
    in one run.

  The logs are `experiments/logs/*_recipe_ladder5.log`. `auxpull.py` was
  not run, since no weight is left to pull.

### The seed replica

Held-out FVU over the 1023 free coordinates:

| | hard stack | direct |
|---|---|---|
| seed 42 | 0.1681 | 0.1611 |
| seed 43 | 0.1679 | 0.1609 |
| difference | 0.0003 | 0.0002 |
| the controls' seed pairs | 0.0020 | 0.0018 |

- **The arm reproduces.** Its seed differences are about a tenth of the
  controls', so the multiples of the seed difference in the earlier
  sections are conservative for this arm.
- **The rest reproduces too.**
  - Pinned rates: 1,891 and 1,877 bits in both seeds.
  - `cossim_k_max`: 0.56 and 0.010. `KL_m`: 0.55 and 2.23.
  - Coordinate 837: 0.16× and 0.10× the median coordinate's error.
- **Coordinate 652 varies most.** In the hard stack it reads 8.9× the median
  coordinate's FVU at seed 43, against 7.8× at seed 42. Its own FVU is 1.49
  and 1.32, so both seeds predict it worse than its mean does. The direct
  stack reads 3.8× and 3.9×.

### The k256 SAE

All at 1,900 bits, from the one `quantrate.py` run, with the stacks at
seed 42. The SAEs' fixed per-row readings are `rowbudget.py`'s exact ones
(next subsection). A stack's fixed per-row reading is its nominal reading,
since its code has a fixed length.

| code | float FVU | mean rate | subset rate | fixed per-row budget |
|---|---|---|---|---|
| **k256** | 0.1260 | **0.1282** | **0.1538** | **0.1288** |
| L1 4.64e-7 | 0.0526 | 0.1421 | 0.1423 | 0.1742 |
| direct stack | 0.1611 | 0.1612 | 0.1638 | 0.1638 |
| hard stack | 0.1681 | 0.1722 | 0.1893 | 0.1893 |
| k160 | 0.1872 | 0.1872 | 0.1874 | 0.1867 |

- **The k256 is below both stacks in all three columns.** It is below the
  direct stack by 0.033 at the mean rate, 0.010 at the subset rate and
  0.035 under the fixed budget.
- **So the direct stack's fixed-budget lead does not hold, as long as an
  SAE can change its step row by row.** It was a lead over SAEs that could
  not spend the budget: the k160 is saturated there, and the L1 SAE's rows
  vary in length. The next subsection has what happens without that freedom.
- **Bits for the k256 to match the stacks,** each as a share of 1,900:

  | target | mean rate | subset rate | per-row budget |
  |---|---|---|---|
  | direct, 0.1612 | 1,262 (66%) | 1,813 (95%) | 1,392 (73%) |
  | hard, 0.1722 | 1,182 (62%) | 1,705 (90%) | 1,361 (72%) |

  Under the full recipe at ε 1e-15, the SAEs needed about a fifth of the
  hard stack's bits (`experiments/sentence-split/notes.md`, 2026-10-25).
- **The k256 is not the SAE frontier either.** Going from k = 160 to 256
  took the float FVU from 0.1872 to 0.1260, and at 1,900 bits it is still
  short of its floor.
- **The training budgets are far apart.**
  - The k256 saw 1.38 billion rows: 336,000 steps of 4,096, about 31 passes
    over the cache.
  - Each stack saw 12.8 million: 50,000 steps of 256, one pass over 29% of
    the cache. That is 107× fewer.
  - So this compares a finished SAE with a short Ontologizer run. How much
    of the gap training closes is not tested.

### What the fixed per-row budget measures

A top-k SAE's budget reading differs from its mean-rate reading, and for
the k32 and k160 it is below the unquantized FVU. Both come from variable
quantization.
- **Where these come from:** `rowbudget.py`, which reads `quantrate.sae_rows`'s
  per-row lengths and errors at exactly 1,900 bits.
  - Output in `data/out/sonar/quantrate_recipe_ladder5/rowbudget_exact.json`,
    log in `experiments/logs/rowbudget_recipe_ladder5.log`.
  - The one-step-on-average row is `quantrate.py`'s mean-rate reading
    instead, interpolated between steps to 1,900 bits.
  - "—" is a reading the code cannot give.

| at a 1,900-bit budget | k32 | k160 | k256 | L1 4.64e-7 |
|---|---|---|---|---|
| float FVU | 0.4180 | 0.1872 | 0.1260 | 0.0526 |
| one step for all rows, 1,900 bits on average | — | 0.1872 | 0.1282 | 0.1421 |
| each row at its longest code that fits | 0.4180 | 0.1872 | 0.1291 | 0.1742 |
| each row at its lowest-error step that fits | 0.4163 | 0.1867 | 0.1288 | 0.1742 |
| one step that every row fits | 0.4180 | 0.1879 | 0.1846 | 0.4985 |

- **A fixed k fixes how many latents a row sends, not how many bits they
  cost.** Rare latents and large values cost more.
  - At a mean of 1,887 bits, the k256's rows run from 1,734 to 2,133 (5th
    to 95th percentile), up to 2,562, and 37% are over 1,900.
  - The L1 4.64e-7 SAE's, at a mean of 1,969, run from 974 to 3,359.
- **The budget column is variable quantization.** Each row picks one of 67
  steps, named in 6.07 bits that count against the budget. The mean-rate
  column uses one step for every row.
- **Picking steps by error is what takes the k32 and k160 below float.**
  - No single step beats float, and neither does each row's longest code.
  - Row error is not monotonic in step size, so the search finds a lucky
    rounding for 100% of the k32's rows and 97.5% of the k160's.
  - The rows it picks are also shorter: 363 and 1,615 bits on average.
- **For the k256, the budget binds, but it costs little.** Forcing long rows
  onto coarser steps adds 0.0008, and the search takes back 0.0003.
- **For the L1 SAE, the budget is all of it.** The search moves 0.1% of its
  rows off their longest code, so its 0.1742 is what the cap costs its
  long rows.
- **The single-step reading is the one that models lowering the
  coefficients' precision.**
  - One step is a property of the model: the same quantization for every
    input. A per-row step makes precision depend on the row, which makes it
    a codec. The stacks' gains also use one step for all rows.
  - That reading is the mean-rate column, where the k256 leads (0.1282,
    against the direct stack's 0.1612).
  - "One step" is one scale: each latent's step is that scale over its
    whitened decoder norm, so latents differ, but rows do not.
- **A hard cap on top of one step changes the ranking.** The two single-step
  rows differ only in how the bits are charged. One that every row fits gives,
  with the stacks' fixed-length readings (no side bits are charged, since
  every row shares the step; charging them changes none of these):
  - direct stack, 0.1638;
  - k256, 0.1846;
  - k160, 0.1879;
  - hard stack, 0.1893;
  - L1 4.64e-7, 0.4985.

  This is a worst case. The longest of the 32,765 rows sets the step, and
  the k256's rows then average 1,107 bits, so about 800 go unspent. Those
  long rows exist because the lengths come from an entropy coder.
- **That reading is still not a bound on the SAE.** It is the longest row
  seen, under tables fit on those same rows, and a symbol the tables have
  not seen has no code at all (`row_bits` raises on one).
- **The stacks' strict bound is the pinned reading:** exactly 1,900 and
  1,890 bits, for 0.1893 and 0.1865. The direct stack's 0.1638 adds gains
  whose 10 or so bits are entropy-coded, so it is not a strict bound either.
- **A literal fixed-width format is not measured.** It is the reading that
  would bound an SAE's rows. With no entropy coder,
  naming the support alone costs 1,757 bits for the k256 (log2 of 11,264
  choose 256), which leaves 0.56 bits per coefficient at 1,900 bits. The
  k160 has 4.3 and the k32 51.
- **`quantrate.py`'s reading at 1,900 bits is interpolated.** `rowbudget.csv`
  has budgets on a quarter-octave grid, and the reading interpolates between
  1,722 and 2,048. For the k256 it gives 0.1300, against the exact 0.1288.
  The per-row readings in this section are `rowbudget.py`'s, which are exact.

### Between-head support

`support` from `loss.csv`, summed over the 5 layers, so 5 means identical
coordinate profiles in every layer. Means over the last 1,000 steps.

| arm | hard stack | direct |
|---|---|---|
| control | 0.26 | 1.18 |
| `s_support` off | 2.62 | 4.95 |
| all four off | 4.91 | 4.97 |

- **These stacks do not have the findings' disjoint support.**
  - All four off reads 0.98 and 0.99 per layer, against 0.05 and 0.24 in
    the controls.
  - They start at 4.87 and 4.86 (the first 100 steps) and stay there, since
    nothing pulls the heads apart.
- **The per-row bit bound does not depend on it.** That bound comes from
  each head choosing one entry, which every arm has by construction.
- **The ladder cannot separate the penalty from the property.** It shows
  `s_support` costing the fit. It does not show whether disjoint heads
  themselves do, since heads disjoint by construction (`--concat`) are not
  in this ladder.

### What this leaves open

- A full-length run of this arm.
- The three noises under the MSE alone.
- A fixed-width reading of the SAEs, the comparison for a strict bound.
- `--concat` under the MSE alone, to separate disjoint heads from the
  penalty.

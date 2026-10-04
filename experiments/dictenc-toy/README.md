# dictenc-toy

A single `DictEnc` layer on a planted factorial code. The layer's forward
pass is itself a generative model: one categorical choice per head, one
non-negative dictionary row per choice, summed over heads and pushed
through the shared linear decoder. This experiment plants exactly that
structure and asks whether the stock training stack recovers it. On
SONAR data none of the interesting questions have ground truth (is this
head a factor? is this entry a level? is the code actually hard?); here
every one of them is exact, and the toy is small enough that a seed
runs in half a minute on CPU beside a live GPU run.

One file: `toy_dictenc.py`. It imports the real `Hyperparams`,
`Ontologizer`, `update` and `schedules`, and touches nothing in the
package.

## The planted code

- `h_true` independent factors, `k_true` uniform levels each, one atom
  per (factor, level) drawn N(0, 1/h_true) per coordinate, so the clean
  signal has about unit variance per coordinate.
  `x = sum_h M[h, z_h] + sigma * eps`.
- Defaults: `d=64, h_true=4, k_true=8, sigma=0.3`, which puts the noise
  floor at FVU ≈ 0.09. 32768 training rows, 8192 held out.
- The model is `hyper.ontologizer(d, d, e_dec, k, h, 1, forward="resid",
  resid_const=True, resid_gain=False, ...)`: one DictEnc with a
  passthrough encoder; `l=1` makes residual forwarding and deep
  supervision vacuous. With `e_dec >= h_true * k_true` an exact solution
  exists (decoder columns = atoms, one-hot dictionary rows), so a failure
  belongs to the training stack, not to the model class. Default
  `e_dec=128`, the live 2x ratio.
- `--antipodal` plants each factor's levels in `(+m, -m)` pairs. Two
  paired samples differ only in the sign of that factor's component,
  and an unbiased bilinear logit is even in its input, so without the
  constant coordinate the pair's logits differ by a cross-term with the
  other factors' sum, which is zero-mean: no consistent separation is
  possible. Since a confused pair decodes to the pair's mean, which is
  zero, telling pairs apart is worth nothing to the reconstruction
  either, so the expected failure is total. The constant coordinate
  adds a linear term that separates the pair. Every factor is then
  exactly zero-mean, so there is no corpus mean to lean on.
- Training mirrors `ontostate.train` step for step (`schedules` +
  `update`, no checkpoint manager): the live mechanism set of
  temperature anneal 1 → 0.03, winner-dropout ramp to 0.1, logit noise
  0.02 → 0.006 and featvar noise 0.1, with `lr=1e-3` and 4000 steps. The
  auxiliary loss weights default to 0: their live values are calibrated
  against a whitened MSE of ~1e-6 and would be inert at this scale.
  Each is a flag.

## Symmetries the scoring has to respect

The generative model is invariant to permuting heads, permuting entries
within a head, and shifting every atom of one head by a vector as long
as the shifts sum to zero over heads. So:

- heads are matched to factors by Hungarian assignment on normalized
  mutual information between the head's argmax entry and the factor's
  level (defined for `h != h_true` and `k != k_true`, and it does not
  punish a head that splits one level over two entries);
- within a matched pair, entries are matched to levels by Hungarian
  assignment on the contingency table, which gives the accuracy;
- decoded dictionary entries (`Ontologizer.decodeEntries`) are compared
  to the planted atoms by cosine after centering both within the head,
  since only centered atoms are identifiable.

## Metrics

| column | meaning |
|---|---|
| `fvu`, `fvu_hard`, `fvu_oracle` | held-out MSE / Var(x) for the soft decode at the eval temperature, for the one-hot argmax code, and for the planted decomposition itself (the noise floor) |
| `acc`, `nmi`, `purity` | per true factor, for its matched head; chance for `acc` is ~1/k_true; `purity` is the headline when `k > k_true`, since a level split over two entries costs `acc` but not `purity` |
| `atom_cos` | mean centered cosine between matched entries and atoms |
| `H_bits` | per-head mean per-sample entropy of the classification at the eval temperature; 0 is a hard code, log2(k) is uniform |
| `logit spread/T` | within-head std of the raw classifier logits in units of the eval temperature; the softmax is hard only when this is large |
| `extras` | model heads left unmatched: best NMI to any factor (redundant if high) and normalized argmax-usage entropy (frozen if ~0) |
| `dead`, `KL_m`, `cossim_k_max` | entries never selected on the eval set, usage KL to uniform, and the within-head row cosine, as the training stats define them |

Every score is also reported for the untrained model and under a
shuffled-label null (the level rows permuted, matching redone), which
is the chance level of the Hungarian statistics at this sample size
(`acc` ≈ 0.14, `nmi` ≈ 0 at the defaults).

Note that `fvu` below `fvu_oracle` is not a better dictionary: the
reconstruction is a function of the noisy input, and a soft code can
pass part of the noise through. The hard code cannot, which is why
`fvu_hard` is the number to compare against the floor.

## Run

From the repo root (CPU by default; set `JAX_PLATFORMS` to override).
Run the first two as a pair: the default is the live selection rule,
and `--select ste` is the positive control that defines what a working
layer looks like here. `--seeds N` advances the model seed and the data
seed together and prints a tally; single seeds mislead on this toy, so
quote tallies.

```bash
T=experiments/dictenc-toy/toy_dictenc.py
uv run python $T --seeds 5                                 # live config
uv run python $T --seeds 5 --select ste                    # positive control
uv run python $T --seeds 5 --antipodal --select ste        # constant coordinate:
uv run python $T --seeds 5 --antipodal --select ste --no-const   # with / without
uv run python $T --seeds 5 --h 8 --select ste              # extra heads
uv run python $T --seeds 5 --k 16 --select ste             # extra entries
uv run python $T --seeds 5 --unit                          # SONAR input regime
uv run python $T --seeds 5 --s-h 0.03 --name sh0.03_x5     # entropy penalty
uv run python $T --seeds 5 --select ste --p-drop 0 --name ste_pd0_x5      # dropout off
uv run python $T --seeds 5 --h 8 --select ste --s-hcossim 0.1 --name h8_hc_x5  # head cosine
uv run python $T --seeds 5 --entropy-target 0.05           # entropy setpoint
```

The entropy setpoint runs the package's `DualLoop` on the code
entropy. `update` threads only two per-step coefficients into the
jitted loss, so the harness borrows the `s_kcossim` slot, which this
toy leaves unidentified (a free decoder makes dense and one-hot
dictionary rows equally exact, and the recovered solutions sit at row
cosine 0.45 to 0.6, so a row-cosine target would penalize a correct
dictionary). The two therefore cannot both be applied in one run, and
the harness refuses it.

`--select top1` is not an arm: a softmax over a single survivor is the
constant 1, so the classifier receives no gradient at all. `top2` is
the smallest top-k rule that trains the classifier.

## Outputs (`experiments/dictenc-toy/out/<name>/`)

- `summary.json`: config; per seed, the full readout before and after
  training including the null; and the tally (per-seed recovered
  counts, how many seeds recovered every factor, mean and std of each
  scalar);
- `heads.csv`: the per-factor rows of the table for every seed, with
  `seed` and `data_seed` columns;
- `loss_s<seed>.csv`: the 16-column training stats row per step, in
  the `Hyperparams.loss` layout, with a header. Under
  `--entropy-target` the borrowed column is written as `s_H_applied`
  rather than `s_kcossim`, and carries the controller's trajectory.

## Reading it

Pass: every factor at `acc` ≥ 0.9, `atom_cos` > 0.99, `H_bits` near 0,
and `fvu_hard` within a few points of `fvu_oracle`, at every seed.

The failure signatures are distinct:

- **Soft code.** `fvu` at the floor but `fvu_hard` far above it,
  `H_bits` above 1, `logit spread/T` of order 1. The reconstruction
  lives in the softmax coefficients, not in tag identity. The
  interventions and the tag decoding both assume the opposite.
- **Wrong partition, hard code.** `H_bits` ≈ 0 with some factors well
  below 0.9: hardening locked in an early mistake. Seed-dependent.
- **Blind classifier.** `H_bits` ≈ 0 and a large logit spread, but
  `nmi` near 0 and `fvu_hard` ≈ 1: the classifier is confident about a
  partition that carries no information, and the decode is the mean.
- **Redundant heads.** With `h > h_true`, extras with high `max_nmi`
  and high usage entropy: the layer spread a factor over two heads
  instead of leaving one idle.

## Pilot readout

Five seeds per arm at the defaults above, about 30 s per seed on CPU.
"recovered" lists, per seed, the factors at `acc` ≥ 0.9 out of 4;
"exact" counts the seeds that recovered all four. The other columns
are means over seeds. The floor is `fvu_oracle` ≈ 0.09, or 0.08 for
the antipodal codes.

| arm | recovered per seed | exact | H_bits | fvu_hard | also |
|---|---|---|---|---|---|
| softmax, live mechanisms (default) | 1 0 0 0 1 | 0/5 | 1.8 | 1.02 | soft `fvu` 0.098; logit spread 1.4 T |
| `ste` | 4 4 4 4 4 | 5/5 | 0 | 0.106 | atom_cos 0.999 |
| softmax, unit-norm inputs | 0 0 0 0 0 | 0/5 | 2.2 | 2.05 | spread starts at 0.9 T |
| `ste`, unit-norm inputs | 2 4 4 4 4 | 4/5 | 0 | 0.14 | |
| softmax, antipodal | 0 0 0 0 0 | 0/5 | 2.2 | 2.10 | acc 0.33 |
| softmax, antipodal, no constant coordinate | 0 0 0 0 0 | 0/5 | 0.08 | 1.006 | acc 0.18, nmi 0.03: the mean predictor, confidently |
| `ste`, antipodal | 0 0 1 1 0 | 0/5 | 0 | 0.31 | acc 0.52 ± 0.15; per head 0.25 to 0.98 |
| `ste`, antipodal, no constant coordinate | 0 0 0 0 0 | 0/5 | 0 | 1.006 | acc 0.19, nmi 0.04 |
| `ste`, antipodal, 3x steps | 1 0 2 4 0 | 1/5 | 0 | 0.25 | acc 0.64 ± 0.23 |
| `ste`, antipodal, lr 3e-3 | 4 0 1 4 1 | 2/5 | 0 | 0.23 | acc 0.69 ± 0.28 |
| `ste`, antipodal, unit-norm inputs | 4 1 2 1 0 | 1/5 | 0 | 0.26 | acc 0.65 ± 0.23 |
| `ste`, h = 8 heads | 2 2 2 2 1 | 0/5 | 0 | 0.112 | purity 0.88; no extra head idle |
| `ste`, k = 16 entries | 0 0 0 0 0 | 0/5 | 0 | 0.106 | purity 0.86, atom_cos 0.96, 0.6 dead of 64 |
| softmax + `s_H` 0.03 | 4 3 1 1 2 | 1/5 | 0.3 ± 0.3 | 0.25 ± 0.17 | |
| `top2` + `p_revive` 0.05 | 0 0 0 0 0 | 0/5 | 0.7 | 0.41 | KL_m 0.27 |

Winner dropout and the head-cosine penalty, all under `ste` at five
seeds. `p_drop` 0.1 is the live ramp and the reference; "extras" gives
the unmatched heads' best NMI to any factor and their usage entropy.

| arm | recovered per seed | exact | atom_cos | fvu_hard | also |
|---|---|---|---|---|---|
| `p_drop` 0 | 4 4 4 4 4 | 5/5 | 1.000 | 0.103 | |
| `p_drop` 0.1 | 4 4 4 4 4 | 5/5 | 0.999 | 0.106 | |
| `p_drop` 0.3 | 4 3 4 3 3 | 2/5 | 0.949 | 0.205 | |
| k = 16, `p_drop` 0 | 2 0 0 0 1 | 0/5 | 0.934 | 0.114 | purity 0.81, 7.4 dead of 64, KL_m 0.40 |
| k = 16, `p_drop` 0.1 | 0 0 0 0 0 | 0/5 | 0.958 | 0.106 | purity 0.86, 0.6 dead |
| k = 16, `p_drop` 0.3 | 0 0 0 0 0 | 0/5 | 1.000 | 0.094 | purity 1.00, 1.2 dead |
| antipodal, `p_drop` 0 | 3 1 1 1 0 | 0/5 | 0.73 | 0.27 | acc 0.65 |
| antipodal, `p_drop` 0.1 | 0 0 1 1 0 | 0/5 | 0.59 | 0.31 | acc 0.52 |
| antipodal, `p_drop` 0.3 | 4 0 0 0 0 | 1/5 | 0.64 | 0.30 | acc 0.58 |
| h = 8, `p_drop` 0 | 2 2 2 1 1 | 0/5 | 0.87 | 0.106 | extras 0.35, usage 0.99 |
| h = 8, `p_drop` 0.1 | 2 2 2 2 1 | 0/5 | 0.82 | 0.112 | extras 0.63, usage 0.96 |
| h = 8, `p_drop` 0.3 | 1 2 2 0 2 | 0/5 | 0.73 | 0.160 | extras 0.73, usage 0.95 |
| h = 8, `s_hcossim` 0.03 | 2 1 1 1 1 | 0/5 | 0.80 | 0.113 | cossim_h 0.65 → 0.35; extras 0.58, usage 0.95 |
| h = 8, `s_hcossim` 0.1 | 1 1 1 1 2 | 0/5 | 0.75 | 0.109 | cossim_h → 0.21; extras 0.48, usage 0.91 |
| h = 4, `s_hcossim` 0.1 | 4 3 2 4 4 | 3/5 | 0.97 | 0.128 | cossim_h 0.62 → 0.31 |

The entropy penalty under softmax with the live mechanisms at five
seeds, over weight and step budget; the long runs anneal over 9000 of
their 12000 steps.

| `s_H` | steps | recovered per seed | exact | acc | H_bits | fvu_hard |
|---|---|---|---|---|---|---|
| 0.03 | 4000 | 4 3 1 1 2 | 1/5 | 0.90 | 0.29 | 0.25 |
| 0.05 | 4000 | 1 1 1 1 0 | 0/5 | 0.75 | 0.00 | 0.23 |
| 0.1 | 4000 | 1 0 0 1 0 | 0/5 | 0.75 | 0.00 | 0.25 |
| 0.3 | 4000 | 0 2 0 0 0 | 0/5 | 0.66 | 0.00 | 0.28 |
| 0.01 | 12000 | 1 2 0 0 0 | 0/5 | 0.66 | 1.70 | 1.04 |
| 0.03 | 12000 | 2 2 1 2 2 | 0/5 | 0.88 | 0.42 | 0.29 |
| 0.1 | 12000 | 1 1 1 2 0 | 0/5 | 0.79 | 0.04 | 0.24 |

The entropy setpoint (`--entropy-target`), the remedy that sweep
suggests: the package's own `DualLoop` holding the code entropy at a
setpoint ramped down from wherever the run starts, so the coefficient
is whatever sustains the target rather than a constant. Softmax, five
seeds, `s_H` starting at 1e-3. "s_H held" is the mean applied
coefficient over the steps where it was nonzero.

| arm | recovered per seed | exact | acc | H_bits reached | s_H held |
|---|---|---|---|---|---|
| target 0.01 | 1 1 1 0 1 | 0/5 | 0.73 | 0.005 ± 0.003 | 0.20 |
| target 0.05 | 1 0 0 0 1 | 0/5 | 0.69 | 0.050 ± 0.002 | 0.21 |
| target 0.2 | 1 0 0 0 1 | 0/5 | 0.68 | 0.164 ± 0.023 | |
| target 0.05, gain 3e-3 | 1 0 0 0 1 | 0/5 | 0.66 | 0.049 | |
| target 0.05, gain 3e-2 | 2 2 2 0 1 | 0/5 | 0.76 | 0.092 | 0.18 |
| target 0.05, gain 1e-1 | 1 1 1 0 1 | 0/5 | 0.72 | 0.043 | |
| target 0.05, gain 3e-1 | 1 1 1 0 1 | 0/5 | 0.71 | 0.024 | 0.30 |
| target 0.01, gain 1e-1 | 1 1 1 0 1 | 0/5 | 0.71 | 0.002 | |
| target 0.05, ramp 1000 | 1 0 0 0 1 | 0/5 | 0.67 | 0.002 | 0.32 |
| target 0.05, ramp 3800 | 1 0 0 0 1 | 0/5 | 0.68 | 0.051 | |
| target 0.05, 12000 steps | 1 0 0 0 1 | 0/5 | 0.66 | 0.054 | 0.31 |
| target 0.05, capped at 0.1 | 1 0 0 0 1 | 0/5 | 0.69 | 0.212 | 0.08 |
| target 0.05, capped at 0.05 | 1 0 0 0 1 | 0/5 | 0.71 | 0.680 | 0.04 |
| target 0.05, unprojected | 1 1 1 0 1 | 0/5 | 0.73 | 0.007 | 0.05 |
| target 0.01, unprojected | 1 2 2 0 1 | 0/5 | 0.74 | 0.002 | 0.14 |
| target 0.05, unit-norm | 0 0 0 0 0 | 0/5 | 0.64 | | |

Single-seed variants of the default arm, all of which stayed soft
(`H_bits` 1.4 to 2.2, `fvu_hard` 0.75 to 2.2, 0 or 1 of 4 recovered):
no winner dropout; all noise and dropout off; no anneal; 3x steps;
`T_end` = 0.003; centered atoms (three seeds: 4, 0, 0 recovered);
unit-norm inputs at 3x steps and at lr 3e-3. And `s_H` at 0.01 was too
weak at 4000 steps (1 of 4); at sigma = 1 (floor 0.53) `s_H` 0.03 was
too weak again, while `ste` recovered 4 of 4 at acc 0.92 to 0.95.

What the pilot shows:

1. **The metric discriminates.** Both nulls sit at chance, the
   untrained model near it, and the positive control passes.
2. **Softmax selection does not produce a hard code, and the anneal
   cannot make it.** At every seed and in every variant the
   classifier's logit spread ends at 1.2 to 1.8 temperature units,
   having started at 30: the bilinear classifier is scale-free, so it
   shrinks its logits as fast as the temperature falls and settles at
   the softness the MSE prefers. A 10x colder endpoint moves the
   spread from 300 T to 8 T and leaves the code at 1.4 bits. The soft
   decode reconstructs at the floor, or below it by passing noise
   through, while its argmax reconstructs at FVU 0.5 to 2.
3. **Straight-through argmax (`select="ste"`, already implemented)
   recovers the planted code.** 5 of 5 seeds exact at the defaults, 4
   of 5 in the unit-norm regime, hard FVU within 0.01 to 0.05 of the
   floor. The single-seed sweep had one `ste` seed stop at 2 of 4 with
   a hard code on a partly wrong partition, so that failure exists but
   is rare.
4. **The constant coordinate is necessary, and not sufficient.**
   Without it the antipodal code collapses to the mean predictor at
   every seed under both selection rules, with the hard classifier
   fully confident about a partition that carries nothing. With it,
   `ste` recovers the antipodal code at 0 to 2 of 5 seeds, and the
   per-seed results are bimodal: a seed lands at 4 of 4 or at 0 to 1
   of 4. Neither 3x steps, 3x learning rate nor unit-norm inputs
   (where the coordinate is eight times the size of a data coordinate)
   changes that, so it is not a budget problem but a race: the sign
   information reaches the logits only through the linear terms the
   coordinate supplies, and hard selection locks in a sign-confused
   partition before they win it. Softmax with the coordinate does no
   better (acc 0.33, soft).
5. **An entropy penalty (`s_H`) is a race, not a fix, and neither a
   stronger weight nor longer training changes that.** At the best
   weight found, 0.03, the tally is 4 3 1 1 2. Above it every seed
   ends fully hard and accuracy falls monotonically, 0.90 to 0.75 to
   0.66 at 0.05, 0.1 and 0.3, with the hard code at 0.23 to 0.28
   against the 0.09 floor: a stronger weight forces commitment
   earlier, before the partition is right. Three times the steps
   leaves 0.01 still soft, 0.03 unchanged in mean, and 0.1 at 0.79.
   The cause is the non-stationarity the codebase already meets in
   `s_L1F` and `s_kcossim`: the MSE gradient shrinks toward the floor
   while the entropy gradient does not, so no constant weight is right
   both early and late, and the working value also scales with the
   MSE. The consistent remedy would be a ramped setpoint, which
   finding 11 builds and rules out.
6. **Extra heads duplicate rather than die.** With 8 heads for 4
   factors no extra head went idle (usage entropy 0.86 to 0.99); they
   took over parts of factors, and exact recovery fell to 2 of 4 at
   every seed. Extra entries split levels: the split entries decode to
   the same atom, and fewer than one of 64 died on average.
7. **The unit-norm regime is harder for softmax but not for `ste`.**
   Unit inputs start the logit spread below the temperature and
   softmax never leaves the soft attractor at any step budget or
   learning rate tried; `ste` recovered 4 of 5.
8. **Top-k selection is worse than both.** `top2` recovered nothing at
   any seed, with usage skew (KL_m 0.27) on top.
9. **Winner dropout is not part of what makes `ste` work, and too much
   of it hurts.** Without it `ste` is still exact at 5 of 5 with atom
   cosine 1.000; at 0.3 it falls to 2 of 5 and blurs the atoms, since
   the promoted runner-up is trained toward samples it should not own.
   It is not the lever in the antipodal race either: 0, 0.1 and 0.3
   all land inside the seed noise. Where it acts is on surplus
   capacity. With extra heads, the extras' best NMI to a factor rises
   from 0.35 to 0.63 to 0.73 as dropout goes from 0 to 0.1 to 0.3, and
   at 0.3 the true heads' atoms degrade too; but the extras are fully
   used even at 0, so dropout amplifies duplication rather than
   causing it. With extra entries the split of levels over entries
   happens without dropout, because `ste`'s backward pass is the
   softmax Jacobian and every entry receives gradient; dropout makes
   the split clean, purity 0.81 → 0.86 → 1.00 and atom cosine
   0.93 → 0.96 → 1.00, and at 0.3 the hard code sits exactly at the
   floor.
10. **`s_hcossim` lowers its own statistic and nothing else improves.**
    It is not inert at the solution: the recovered code has `cossim_h`
    about 0.62, because per-head reconstructions are dense and
    non-negative in code space even when the planted atoms are
    orthogonal, so the penalty selects a different solution rather
    than confirming this one. At about a quarter of the loss it halves
    `cossim_h` and costs recovery at `h = h_true`, 3 of 5 exact against
    5 of 5. With extra heads it reduces the extras' overlap with the
    true heads but does not idle them, usage entropy stays above 0.9,
    and recovery falls rather than rises. Head decorrelation in code
    space is not informational independence, which is the premise of
    `experiments/hsic-bottleneck`.
11. **The entropy setpoint controls perfectly and recovers nothing.**
    It is a good controller: target 0.05 lands at 0.050 ± 0.002 across
    seeds, where the best fixed weight wandered over 0.29 ± 0.30. It is
    also the right shape in principle, since entropy is measured after
    the classifier has rescaled its logits, so unlike the temperature
    anneal it cannot be evaded by shrinking them. It still recovers
    nothing: 0 of 5 exact in all sixteen arms above, at accuracy 0.66
    to 0.76, below the best fixed weight's 0.90 and far below the
    positive control's 1.00.

    The reason is the coefficient the setpoint demands, not the
    schedule. Holding a low entropy target costs a weight of 0.2 to
    0.3, which is exactly where the fixed sweep already measured the
    damage (0.1 gives accuracy 0.75, 0.3 gives 0.66), and the setpoint
    arms land on those same numbers. Capping the coefficient below that
    band does not buy the target: at a ceiling of 0.1 the loop
    saturates and entropy stalls at 0.21 bits, at 0.05 it stalls at
    0.68. So the controller does not escape the tradeoff, it steers to
    the wrong side of it. Three further explanations are ruled out.
    Not the projection to zero: `dual_apply` leaves the penalty on for
    about 60% of steps including the first third, and applying the
    unprojected dual variable instead changes accuracy by 0.04.
    Not the gain or the ramp: two decades of gain and ramps from 1000
    to 3800 steps span 0.66 to 0.76. Not the budget: three times the
    steps gives 0.66.

    What the best fixed weight had was not a better schedule but a
    weight too weak to reliably harden, leaving the partition free to
    settle on its own; its good seeds are the ones where it did. That
    is luck, not control. Hardness bought through the objective costs
    partition accuracy however it is bought, and the way out is to take
    it out of the objective: under `ste` the code is exactly hard, the
    penalty is inert, and reconstruction sits at the floor.

What it predicts about the live run, to be checked rather than
assumed: the live `entropy` column at T = 0.03 says directly whether
the SONAR model's content is in tag identity or in soft coefficients,
and `pareto.py`'s hard-code capacity curve says how much it costs.
`ONTO_SELECT=ste` is the one-variable arm; the toy says it is the one
worth training before any of the softer remedies.

## Cost

About 30 s per seed on CPU, JAX startup included, so a five-seed arm is
2.5 min; the arms in the table above ran in parallel in about ten
minutes on 24 cores. No data, checkpoint or GPU is needed.

## Caveats

- One scale (d = 64, 4 x 8) and one learning rate. The table rows are
  five seeds each; the variants paragraph is single seeds and says what
  happened, not how often.
- Levels are uniform and independent, so `s_Hm` is inert and the
  frozen-head pathology of the live run (which needs skewed usage) is
  not reproduced here. Skewed marginals and correlated factors are the
  natural next arms for that question.
- `--unit` normalizes after adding noise, so the model class is only
  approximate there (the norm of a factorial sum varies by a few
  percent across samples). Atom cosines in that regime are against the
  raw atoms.
- The aux loss weights are off by default. The live values were tuned
  against a whitened MSE ~1e-6; a comparable strength here is roughly
  1e4 to 1e5 times larger.

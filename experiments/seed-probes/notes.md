# seed-probes: notes

## 2026-11-11: translation probes and agreement by band

Two questions the one-to-one seed comparisons leave open, each on the
32-way stack pair (`ste_h76_init01`, `ste_h76_i01_s43`; 5 × 76 heads) and
the flat pair (`ste_l1_h380_i01`, `ste_l1_h380_i01_s43`; 1 × 380), at step
369,500. One seed pair per shape.

### Translation (`translate.py`)

131,072 rows of `mc4_fresh` (no model trained on them): 81,920 to fit,
16,384 to choose the penalty, the last 32,768 to score. Chance is 1/32 =
0.031, and a head's most used entry covers 0.037–0.045 of rows. `earlier`
is A's own checkpoint 19,500 steps before (step 350,000).

```bash
M=data/out/sonar/multilingual
uv run python experiments/seed-probes/translate.py --mlp --earlier 350000 \
    --a $M/ste_h76_init01 --b $M/ste_h76_i01_s43
uv run python experiments/seed-probes/translate.py --mlp --earlier 350000 \
    --a $M/ste_l1_h380_i01 --b $M/ste_l1_h380_i01_s43
```

**Stack, mean held-out accuracy over seed 42's heads, by their layer.**
`all` is a ridge from the source's whole code, `mlp` a one-hidden-layer
probe from it, `same` a ridge from the source's heads of the target's
layer.

| source | probe | all layers | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|---|
| input | ridge | 0.303 | 0.568 | 0.337 | 0.248 | 0.199 | 0.162 |
| seed 43 | all | 0.166 | 0.378 | 0.167 | 0.117 | 0.093 | 0.075 |
| random-projection heads | all | 0.152 | 0.345 | 0.152 | 0.108 | 0.086 | 0.070 |
| untrained stack | all | 0.143 | 0.312 | 0.144 | 0.104 | 0.085 | 0.070 |
| seed 42, earlier | all | 0.485 | 0.890 | 0.634 | 0.432 | 0.293 | 0.179 |
| seed 43 | mlp | 0.191 | 0.437 | 0.196 | 0.135 | 0.104 | 0.080 |
| random-projection heads | mlp | 0.160 | 0.370 | 0.160 | 0.112 | 0.088 | 0.070 |
| untrained stack | mlp | 0.139 | 0.322 | 0.136 | 0.097 | 0.078 | 0.064 |
| seed 42, earlier | mlp | 0.327 | 0.718 | 0.437 | 0.248 | 0.144 | 0.089 |
| seed 43 | same | 0.109 | 0.293 | 0.088 | 0.062 | 0.053 | 0.047 |
| random-projection heads | same | 0.105 | 0.230 | 0.098 | 0.075 | 0.064 | 0.056 |
| untrained stack | same | 0.091 | 0.192 | 0.084 | 0.067 | 0.059 | 0.054 |
| seed 42, earlier | same | 0.492 | 0.890 | 0.634 | 0.432 | 0.298 | 0.208 |

Seed 43's margin over the better of the two controls, head by head:

| probe | all layers | L0 | L1 | L2 | L3 | L4 | heads above the control |
|---|---|---|---|---|---|---|---|
| all | +0.014 | +0.033 | +0.015 | +0.010 | +0.006 | +0.004 | 99% |
| mlp | +0.030 | +0.067 | +0.036 | +0.023 | +0.016 | +0.010 | 100% |
| same | +0.004 | +0.063 | −0.010 | −0.013 | −0.012 | −0.009 | 20% |

With the two seeds' roles swapped the table is the same to the third
decimal (margins +0.014, +0.030, +0.004).

**Flat pair.**

| source | ridge | mlp |
|---|---|---|
| input | 0.575 | 0.581 |
| seed 43 | 0.293 | 0.318 |
| random-projection heads | 0.266 | 0.287 |
| untrained | 0.241 | 0.245 |
| seed 42, earlier | 0.897 | 0.737 |

Margin over the better control: +0.027 (ridge), +0.031 (mlp), above it
for 99% of heads.

**Reading.**

- **Any code of the input predicts these heads several times over
  chance.** Random-projection heads reach 0.152 on the stack and 0.266 on
  the flat arm, against a chance of 0.031. A translation score without
  this control would have read as substantial agreement.
- **The other seed's code is barely better than that.** Its margin is
  consistent (nearly every head) and small: +0.014 by the ridge, against
  the +0.33 by which the same run's earlier checkpoint clears the control.
  As a share of that, head by head, the median is 3.5% on the stack (6.4%
  in layer 0, 3.1–3.7% above it) and 4.3% on the flat arm. No head's
  margin exceeds 0.05.
- **A hidden layer doubles the margin and does not change the picture**
  (+0.030). So a little of what the seeds share is non-additive across
  heads. The probe itself is weak: it reads the earlier checkpoint worse
  than the ridge does (0.327 against 0.485), so it bounds nonlinear
  re-factoring from below only.
- **Layer for layer, seed 43's layers 1–4 predict seed 42's worse than
  random partitions of the raw input do** (−0.009 to −0.013). A layer
  above the first partitions its own residual, from which that seed's
  earlier codes have been removed; a random partition of the input has
  not lost them.
- **One run's hold on its own code falls with depth.** From 19,500 steps
  earlier, a layer-0 head is predicted at 0.89 and a layer-4 head at
  0.18–0.21. The flat arm's heads read 0.90.
- **The topic head** (layer 0, head 54 in seed 42) is the most
  predictable head from every source: 0.605 from seed 43's code, 0.631
  from random-projection heads, 0.596 from the untrained stack. So a whole
  code of the other seed knows no more about it than an arbitrary code of
  the same shape. Its single-head match is another matter: its best NMI
  with one head of seed 43 is 0.416 (head 14), against 0.097 for the best
  of 380 random-projection heads and 0.092 for the best untrained layer-0
  head (32,768 fresh rows). What recurs across seeds is that one head
  holds this variable, not that the code holds it.

**Limits.** One seed pair per shape. The probes read what a ridge and a
small network can; a re-factoring neither reads is not excluded. The
`same` comparison gives the controls the raw input at every layer, which
favors them above layer 0.

### Agreement by band (`bandagree.py`)

`headcontrib.py`'s frame (whitened, centered contributions, 4,096 tail
rows of `mc4_4M`), split by rank in the spectrum of the whitened data
covariance (32,768 tail rows). Matched centered cosine, mean over A's
heads; the null shuffles B's rows.

```bash
uv run python experiments/seed-probes/bandagree.py --earlier 350000 \
    --a $M/ste_h76_init01 --b $M/ste_h76_i01_s43
uv run python experiments/seed-probes/bandagree.py --earlier 350000 \
    --a $M/ste_l1_h380_i01 --b $M/ste_l1_h380_i01_s43
```

**Stack.**

| ranks | data variance | head energy | seeds | null | earlier | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0–8 | 0.113 | 0.080 | 0.047 | 0.016 | 0.486 | 0.165 | 0.019 | 0.017 | 0.017 | 0.016 |
| 8–32 | 0.100 | 0.083 | 0.035 | 0.009 | 0.485 | 0.129 | 0.013 | 0.011 | 0.010 | 0.010 |
| 32–128 | 0.198 | 0.198 | 0.024 | 0.005 | 0.483 | 0.093 | 0.008 | 0.007 | 0.006 | 0.005 |
| 128–512 | 0.400 | 0.443 | 0.014 | 0.002 | 0.479 | 0.054 | 0.006 | 0.005 | 0.004 | 0.004 |
| 512–1024 | 0.189 | 0.196 | 0.012 | 0.002 | 0.470 | 0.046 | 0.004 | 0.004 | 0.004 | 0.003 |

The layer columns are the seeds' matched cosine by seed 42's layer.

**Flat pair.**

| ranks | data variance | head energy | seeds | null | earlier |
|---|---|---|---|---|---|
| 0–8 | 0.113 | 0.073 | 0.100 | 0.016 | 0.842 |
| 8–32 | 0.100 | 0.073 | 0.080 | 0.009 | 0.860 |
| 32–128 | 0.198 | 0.175 | 0.050 | 0.005 | 0.855 |
| 128–512 | 0.400 | 0.433 | 0.024 | 0.002 | 0.841 |
| 512–1024 | 0.189 | 0.246 | 0.016 | 0.002 | 0.807 |

**Reading.** The null differs between bands, since a band of fewer
dimensions has a higher chance cosine, so the bands are compared by the
seeds' excess over the null as a share of the earlier checkpoint's:
(seeds − null) / (earlier − null).

| ranks | stack | flat |
|---|---|---|
| 0–8 | 6.6% | 10.2% |
| 8–32 | 5.3% | 8.3% |
| 32–128 | 4.0% | 5.3% |
| 128–512 | 2.5% | 2.6% |
| 512–1024 | 2.2% | 1.7% |

- **Agreement does concentrate where the data vary most**: three times
  the share in the top eight directions as in the bottom half for the
  stack, six times for the flat arm.
- **It is small in every band.** The best is a tenth of what one run
  keeps with itself. Restricting attention to the data's main directions
  does not make heads reproducible.
- **A run agrees with itself about equally in every band** (0.47–0.49 on
  the stack, 0.81–0.86 flat), so the low-variance directions are not
  where a code is unstable in training. They are where two seeds differ
  most.
- **The stack's layers 1–4 are at the null in every band** (0.016–0.019
  against 0.016 in the top band; 0.003–0.004 against 0.002 in the bottom
  half).
- The whitened spectrum is flat enough that the bottom half of the
  directions holds 19% of the variance and 20–25% of what heads write.

**Limits.** One seed pair per shape; the layer columns are read against
the all-head null, not a per-layer one. The whitening is the objective's
per-coordinate weighting, so "low variance" here means low after it.

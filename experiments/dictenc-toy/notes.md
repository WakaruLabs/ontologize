# dictenc-toy: notes

Earlier results are in [`README.md`](README.md), which predates this
file.

## 2026-11-11: a feedback anneal hardens the code and does not find the partition

**Why.** The stock schedule lowers the temperature by step count, and on
this toy the softmax model stays soft under it (README, "Reading it").
Agustsson et al. (2017) anneal a softmax relaxation of quantization by
feedback instead, raising the hardness so that the gap between hard and
soft reconstruction error follows a shrinking target, and warn that too
slow an anneal lets the network invert the soft assignments.
`--gap-target` does this with the package's `DualLoop` on 1/T. The
question was whether the soft arm's failure to quantize is the schedule
or the relaxation.

**Arms.** Defaults otherwise (4 factors of 8 levels, d = 64, sigma 0.3,
4,000 steps), five seeds each, `out/anneal_*`:

| arm | flags |
|---|---|
| stock | none: T from 1 to 0.03 over 3,000 steps |
| gap | `--gap-target 0.01`, at `--gap-eta` 1e-2 (the default), 3e-3 and 1e-3 |
| stock, matched | `--temperature-end 0.01` and `0.003`: the stock schedule run to where the controller ends |
| straight-through | `--select ste`, the positive control |

**Results** (mean ± s.d. over seeds; the noise floor `fvu_oracle` is
0.093):

| arm | soft FVU | hard FVU | factors recovered, by seed | every factor | code entropy (bits) | final T |
|---|---|---|---|---|---|---|
| stock | 0.098 ± 0.001 | 1.018 ± 0.149 | 1 0 0 0 1 | 0/5 | 1.79 | 0.03 |
| stock to T 0.01 | 0.103 ± 0.002 | 1.000 ± 0.114 | 1 0 0 0 1 | 0/5 | 1.74 | 0.01 |
| stock to T 0.003 | 0.116 ± 0.005 | 0.737 ± 0.191 | 1 1 0 0 1 | 0/5 | 1.42 | 0.003 |
| gap, gain 1e-2 | 0.214 ± 0.035 | 0.218 ± 0.033 | 2 2 2 0 1 | 0/5 | 0.03 | 0.020 ± 0.017 |
| gap, gain 3e-3 | 0.227 ± 0.029 | 0.241 ± 0.027 | 2 0 2 1 1 | 0/5 | 0.08 | 0.008 ± 0.005 |
| gap, gain 1e-3 | 0.236 ± 0.023 | 0.263 ± 0.050 | 1 2 0 1 1 | 0/5 | 0.08 | 0.009 ± 0.005 |
| straight-through | 0.113 ± 0.013 | 0.113 ± 0.013 | 4 3 4 4 4 | 4/5 | 0.00 | 0.03 |

**Readings.**
- **The feedback does harden the code, and the end temperature alone
  does not.** Under the controller the hard and soft errors meet (0.218
  against 0.214) and the code's entropy falls to 0.03 bits. The stock
  schedule run to the same or a lower temperature leaves the hard error
  at 0.74–1.00: lowering T on a fixed timetable lets the classifier
  shrink its logits as fast as T falls (logit spread over T stays near 1
  at T 0.01), which is the inversion Agustsson et al. warn of. Holding
  the measured gap to a setpoint denies it that.
- **It hardens onto a wrong partition.** The gap closes because the soft
  error rises to meet the hard one, from 0.098 to 0.21–0.24, more than
  twice the noise floor, and no seed recovers every factor at any gain.
  This is the README's "wrong partition, hard code" signature: the code
  is committed while the partition is still forming.
- **Straight-through is better on every count**: half the error, and
  every factor in four seeds of five. A hard forward pass from the first
  step finds the partition; a soft code driven hard afterwards keeps the
  one it had.
- **The gain changes little.** A slower controller ends slightly softer
  and slightly worse. At gain 1e-2 the temperature overshoots and
  recovers (0.06, 0.10, 0.005, 0.006, 0.10, 0.014 at 500-step intervals
  in seed 42); the result is the same without that.

**So for the question asked:** on this toy the soft arm's failure to
quantize is partly the schedule, since feedback removes the gap, and the
remedy is still worse than not relaxing at all. The plan's SONAR stage
was conditional on the toy reaching a good hard code, and it did not, so
no SONAR run was made.

**Caveats.** One planted scale and one target (0.01). The warmup is
1,000 soft steps, during which the partition the controller later
commits to is formed under a soft objective; a shorter warmup, or a
controller on from the first step with a looser start, was not tried
and is the obvious variant. The controller measures the gap on the
training batch without the training noise. `logit spread/T` is not
comparable across arms that end at different temperatures.

```bash
T=experiments/dictenc-toy/toy_dictenc.py
uv run python $T --seeds 5 --name anneal_fixed_x5
uv run python $T --seeds 5 --name anneal_ste_x5 --select ste
uv run python $T --seeds 5 --name anneal_gap0.01_eta1e-2_x5 --gap-target 0.01
uv run python $T --seeds 5 --name anneal_gap0.01_eta3e-3_x5 --gap-target 0.01 --gap-eta 3e-3
uv run python $T --seeds 5 --name anneal_gap0.01_eta1e-3_x5 --gap-target 0.01 --gap-eta 1e-3
uv run python $T --seeds 5 --name anneal_fixed_Te0.01_x5 --temperature-end 0.01
uv run python $T --seeds 5 --name anneal_fixed_Te0.003_x5 --temperature-end 0.003
```

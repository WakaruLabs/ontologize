# layer4-freeze: notes

## 2026-10-06: audit of results computed with the old `freeze_diag.py` probe

`freeze_diag.py`'s probe predated the constant coordinate, the gain-shape
split and the router: it classified each layer's raw next-layer input and
added `combine(hfwd(P))` without the layer gain. It now calls
`autointerp.onto_probe`, which applies all of them as the forward does
(pinned to the forward by `tests/test_autointerp.py`). Which recorded
results the old probe could have changed:

**`resid_nc` and `resid_nc_hm`: unaffected.** Both predate `constinput`
and `resid_gain` (no layer-0 constant, no gain-shape, no router), and on
both the old probe is bit-identical to `onto_probe` (max |difference| 0.0
over 1024 tail rows at the final checkpoint, and over 4096 at
`resid_nc` step 10000). A rerun of `resid_nc` reproduces the saved
`diag_nc_summary.csv` frozen counts (0, 5, 18, 4, 0 at step 10000); its
means differ from the saved ones by ~1e-4, the same between old and new
probe on this machine, so that is run-to-run numerics, not the probe.

**The four retrains (`retrain_variants.py`; Figure `fig-variants`):
unaffected.** Their checkpoints no longer exist, but their saved
summaries match `resid_nc_hm`'s per (step, layer) almost exactly: frozen
counts agree on 100% of 185 rows for the control and 98%, 100%, 93% for
`drop_ramp`, `sdk_floor`, `slow_anneal` (median |top-share difference|
0.010-0.026). Only a model with `resid_nc_hm`'s configuration -- no
gain-shape -- can track it that closely, and on that configuration the
old probe is exact.

**The HSIC-bottleneck arms (`diag_hsic`, `diag_both`; the appendix's
"HSIC bottleneck in place of the auxiliary losses" and Figure `fig-hsic`):
not verifiable.** `train_hsic.py` built its model through
`hyper.ontologizer(...)` without passing `resid_gain`, so the arms took
the dataclass default at training time, which became True on 2026-09-10.
Their run directories are gone and nothing recorded which side of that
date they trained on. They do predate `constinput` (the old probe would
have failed on a layer-0 classifier one coordinate wider). If they had
gain-shape on, the old probe measured them wrongly past layer 0, and the
error has the reported shape. On `bl_ctl_full`, a gain-shape model with no
frozen heads under the correct probe, the old probe's logic (raw residual
plus constant into the classifier, no gain) collapses argmax usage in the
deep layers (4.41, 4.38, 2.63, 0.92, 0.75 bits by layer, against 4.41,
4.62, 4.27, 4.11, 3.84) and flags 10 and 11 heads in layers 3 and 4 at a
modal-entry share >= 0.95 (`freeze_diag`'s share threshold), against 0
and 0: as the residual shrinks, the constant coordinate dominates the
classifier's input and every sample gets the same entry. The HSIC arms
froze late and deep (layer 3 32/32 from step 80k; 77 of 160 heads by the
end), which a real HSIC collapse would also produce -- a constant head has
zero dependence on the others. Settling it needs either the arms'
`log.jsonl`/spec from wherever they ran, or a retrain of one arm to ~80k
steps diagnosed with the fixed script. Their reconstruction numbers
(training MSE from `loss.csv`) do not depend on the probe. (Resolved
2026-10-08, below: gain-shape off, the freeze real, the estimator its
cause.)

### 2026-10-07: retrain of the `hsic` arm with gain-shape on

`train_hsic.py --arm hsic --s-hsic-heads 1e-4 --epochs 6 --resid-gain 1
--const0 0` (seed 42; `data/out/sonar/hsic_check/hsic`), the
configuration under which the old probe would be wrong. The harness
needed repairs to run at all on current code (see the hsic-bottleneck
README). Optimizer and schedules do not depend on the epoch count (constant
Adam rate; anneals on the absolute 50k-step horizon), so the first 93k
steps follow the 24-epoch run's path. Training was healthy to step
~92,400 (1000-step mean whitened MSE 5.2e-5) and then diverged in its last
~700 steps (final rows: MSE 1.7e-2, KL_m 4.3), so only checkpoints
through 90k are read below.

Frozen heads per layer 0-4, both probes on the same checkpoints
(`freeze_diag.py` and `old_probe_diag.py`, outputs in
`data/out/sonar/hsic_check/diag_fixed` and `diag_old`):

| step | fixed probe | old probe |
|---|---|---|
| 50k | 0 0 0 0 0 | 0 0 0 0 0 |
| 60k | 0 0 0 0 0 | 0 0 0 0 18 |
| 70k | 0 0 0 0 3 | 0 0 0 0 28 |
| 80k | 0 0 0 0 2 | 0 0 0 0 29 |
| 90k | 0 0 0 0 3 | 0 0 0 6 28 |

On a gain-shape HSIC model the old probe manufactures a large late, deep
freeze that the correct probe does not see.

But this retrain is not the original arm. Layer 0, which no probe
difference touches, diverges from the saved original from the start:
original layer-0 usage 3.1-3.5 bits (mean top share ~0.5) through 90k,
retrain 2.7 -> 4.8 bits (top share 0.45 -> 0.09); the original froze
layer 3 first (32/32 by 80k), the retrain's old-probe freeze is in
layer 4. So the original arms most likely trained with gain-shape off, on
which the old probe is exact and their freeze would be real. The
no-gain retrain (`--resid-gain 0 --const0 0`,
`data/out/sonar/hsic_check_nogain`) tests that directly: if it
reproduces the original's trajectory, the reported freeze stands.

### 2026-10-07: retrain with gain-shape off

`--resid-gain 0 --const0 0`, otherwise as above
(`data/out/sonar/hsic_check_nogain/hsic`). It crashed once on a GPU OOM at
step 13,600 and resumed from that checkpoint (the batch stream continues;
the training RNG restarts from the seed, so noise draws after 13.6k differ
from an uninterrupted run); it then ran to 93,150 without failing, final
loss 7.8e-5, no late divergence. With gain-shape off the old and fixed
probes are identical, so one diagnosis serves both
(`diag_fixed`): no frozen heads in layers 0-3 at any checkpoint, 2-5 in
layer 4 from 80k.

It does not reproduce the original either. Layer 0, by usage bits /
mean top share:

| step | original | gain-shape on | gain-shape off |
|---|---|---|---|
| 10k | 3.46 / 0.376 | 2.74 / 0.452 | 2.74 / 0.452 |
| 40k | 3.09 / 0.487 | 4.17 / 0.177 | 4.02 / 0.189 |
| 90k | 3.07 / 0.566 | 4.80 / 0.087 | 4.75 / 0.111 |

The two retrains agree with each other and both leave the original
within 10k steps, so something other than `resid_gain` differs from the
original arms': the HSIC estimator (next section). With the unbiased
estimator both retrains stay healthy to 93k steps (at most 5 frozen
heads, all in layer 4).

### 2026-10-08: the original arms used the biased estimator

Both retrains above ran the unbiased head-CKA estimator, `train_hsic.py`'s
current default. The original arms ran the biased one:
`--s-hsic-heads`'s help sizes 1e-4 for "per-layer CKA O(0.01-0.1)", the
biased estimator's range on trained heads (unbiased: ~0.001), and
`pairwise_head_cka` changed its default when the straight-through arm
found the bias (`experiments/ste-arm/notes.md`, "Head-independence
pressure"). Retrained with `--hsic-estimator biased --resid-gain 0
--const0 0`, otherwise as above (one epoch, 15,525 steps;
`data/out/sonar/hsic_check_biased`), step 10k reproduces the original
`hsic` arm. Usage bits / mean top share:

| step 10k | layer 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| original `hsic` | 3.459 / 0.376 | 4.364 / 0.198 | 4.592 / 0.156 | 4.804 / 0.100 | 4.812 / 0.095 |
| original `both` | 3.459 / 0.376 | 4.362 / 0.199 | 4.591 / 0.157 | 4.801 / 0.100 | 4.810 / 0.096 |
| retrain, biased | 3.459 / 0.376 | 4.365 / 0.197 | 4.590 / 0.158 | 4.806 / 0.100 | 4.811 / 0.097 |
| retrain, unbiased | 2.744 / 0.452 | 2.963 / 0.339 | 3.152 / 0.309 | 3.165 / 0.308 | 3.366 / 0.282 |

The biased retrain is as close to the original `hsic` arm as the original
`both` arm is (`KL_m` agrees to four decimals as well). Gain-shape alone
moves the 10k row by up to 0.13 bits (the unbiased pair), so this also
pins the configuration: gain-shape off and no layer-0 constant, as the
arms' author recalls. On it the old probe is exact (`old_probe_diag.py`
and `freeze_diag.py` agree to 0.0 over every head and column of this
checkpoint), so the original arms' frozen counts are correct
measurements. **The freeze was real, and the biased estimator caused
it**: with everything else held fixed, the unbiased estimator leaves the
original's trajectory by the first checkpoint and freezes at most 5
layer-4 heads through 93k.

The mechanism. On heads independent by construction, the biased
estimator still reads roughly (k_eff - 1)/(b - 1), k_eff = 1/sum p^2 a
head's effective entry count, and scores a frozen head 0 against every
other (`experiments/hsic-bottleneck/cka_floor.py`; b=256, h=32, k=32):

| usage | frozen heads | biased | unbiased |
|---|---|---|---|
| uniform (k_eff 32) | 0 | 0.108 | -2e-4 |
| Dirichlet(0.1) (k_eff 4) | 0 | 0.016 | -4e-5 |
| uniform | 16 | 0.026 | 7e-5 |
| uniform | 28 | 0.001 | -1e-5 |

With the heads already near-independent, the only way the weights can
lower the biased penalty is to concentrate or freeze heads. In this run
it is worth 1.0-1.3e-4
weighted through 15.5k steps (raw 1.04 rising to 1.27, summed over
layers; the unbiased run's reads 0.000-0.014), a fifth of the MSE at
that stage (6.4e-4) and above the MSE the model reaches later (~5e-5),
which is when the original froze (layer 2 near-collapsed by 40k, layer
3 from 80k). The auxiliary terms do not resist it: `both` froze in step.

So the appendix's HSIC-bottleneck result is the straight-through arm's
estimator-bias failure (writeup `sec:hsic`) in a configuration where it
could freeze heads, not evidence about head-independence pressure. The
six-epoch retrain below reproduces the freeze itself.

### 2026-10-08: the biased retrain reproduces the freeze

The same configuration for six epochs (93,150 steps, one uninterrupted
run as a systemd unit; `data/out/sonar/hsic_check_biased_e6`),
diagnosed at every checkpoint (`diag_fixed`; gain-shape off, so the old
probe would read the same). Frozen heads per layer 0-4:

| step | original `hsic` | original `both` | retrain, biased |
|---|---|---|---|
| 40k | 0 0 1 0 0 | 0 0 1 0 0 | 0 0 2 0 0 |
| 60k | 0 0 0 7 0 | 0 0 0 7 0 | 0 0 0 6 0 |
| 70k | 0 0 0 29 0 | 0 0 0 31 0 | 0 0 0 29 0 |
| 80k | 0 0 1 32 0 | 0 0 0 32 0 | 0 0 0 32 0 |
| 90k | 0 0 4 32 10 | 0 0 1 31 5 | 0 0 0 31 14 |

Usage bits and top share agree with the original `hsic` arm to within
0.04 bits and 0.01 through 40k. From 50k to 70k the largest per-layer
gap (0.12-0.28 bits) is smaller than the gap between the two original
arms (0.24-0.39); at 80-90k it widens to 0.71-0.95 bits, two to three
times theirs, as individual heads freeze at different checkpoints.
Layer 3 freezes in the same window and completely, as in the original.

The penalty is what the freeze lowers. Raw (summed over layers) it peaks
at 1.29 at 20k and falls to 0.76 at 40k, 0.43 at 80k and 0.27 at 93k,
an 80% drop, while the unbiased retrain's stays at 0.02. Reconstruction
pays during the freeze and mostly recovers: training MSE 2.7e-4 against
the unbiased retrain's 1.8e-4 at 40k, 9.0e-5 against 7.3e-5 at 80k,
7.4e-5 against 7.0e-5 at 93k (one seed each; the unbiased run resumed
once at 13.6k). `KL_m` rises to 1.56 by 93k as frozen heads unbalance
usage.

Verdict: the recorded HSIC-bottleneck freeze is real, reproduces on
current code, and is the biased estimator's. Under the unbiased
estimator the same arm freezes at most 5 layer-4 heads by 93k.

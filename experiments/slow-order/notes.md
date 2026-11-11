# slow-order: notes

## 2026-11-11: one epoch, one seed per arm: the order forms inside a run

**Runs.** `train_order.py --dec 1 --epochs 1 --keep-every 2500`, ordered at
seed 43 and symmetric at seed 42 (`data/out/sonar/slow-order/
{ord,sym}_dec1_sp0.001_ph38_s{43,42}`), 15,420 steps of 256 pairs. The
other seed of each arm is training. The baseline is the long Fano pair
(`data/out/sonar/next-sentence/long/doc2_fano_sp1e-3_ph38{,_s43}`), which
differs in `--dec` alone. Everything below is `order.py --step 15000` on
the cache's last 32,768 rows; κ is on same-document pairs two segments
apart (30,345 pairs). Outputs in `data/out/sonar/slow-order/eval/`.

**One epoch is early for the block.** The baseline's slow heads match
across seeds at NMI 0.145 here, against 0.41 at step 46,250
(`experiments/next-sentence/notes.md`, 2026-11-10). What follows compares
arms at the same step; it does not say where any of them ends up.

**Per run, the 38 slow heads:**

| | Fano 42 | Fano 43 | symmetric 42 | ordered 43 |
|---|---|---|---|---|
| held-out FVU_w (whole model) | 0.5229 | 0.5223 | 0.5203 | 0.5201 |
| κ, mean | 0.195 | 0.194 | 0.105 | 0.124 |
| κ, range over heads | 0.153–0.247 | | 0.084–0.144 | 0.076–0.197 |
| κ against head index (rank correlation) | +0.11 | +0.10 | −0.27 | −0.92 |
| NMI with language, mean | 0.152 | 0.152 | 0.083 | 0.099 |
| NMI with language against head index | +0.03 | +0.14 | −0.27 | −0.92 |
| NMI with earlier heads, first half of the block | 0.135 | 0.136 | 0.029 | 0.087 |
| NMI with earlier heads, second half | 0.140 | 0.141 | 0.029 | 0.037 |
| NMI between two slow heads | 0.139 | 0.139 | 0.029 | 0.044 |

Ordered seed 43 by head: κ 0.178, 0.197, 0.172, 0.158, 0.166 for heads
0–4, 0.145 at head 9, 0.144 at head 18, 0.090 at head 28, 0.087 at
head 37.

**The baseline pair, as a block** (Fano 42 against 43): matched NMI
0.1453, any pairing 0.1392, best partner over runner-up 0.0014 (median).
Head i against head i: 0.1394. The match keeps the index for 5.3% of
heads and the best partner is the own index for 2.6% (chance 2.6%).

**Readings.**
- **The term does what it was built to do within a run.** Both arms cut
  the NMI between slow heads to a fifth to a third of the baseline's.
  Under the ordered term the first heads stay as slow as a baseline head
  and slowness falls with the index; under the symmetric term every head
  gives up about the same amount.
- **Neither arm pays in reconstruction** at this step (FVU within 0.003
  of the baseline, on the lower side).
- **Decorrelated heads are less slow on average** (κ 0.105 and 0.124
  against 0.195), as they must be if the baseline's heads all sat on the
  one or two slowest directions.
- **The baseline has no order**, as expected: head i of one seed is no
  closer to head i of the other than to any head.
- **Not yet measured:** whether head i is the same head in two seeds of
  the ordered arm, which is what the experiment is for. The two finished
  runs are different arms; scored against each other they match at 0.040
  against 0.036 for any pairing, which tests nothing.

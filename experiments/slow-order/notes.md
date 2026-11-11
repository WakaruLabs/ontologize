# slow-order: notes

## 2026-11-11: one epoch: the order reproduces as a gradient; no head is identified

**Runs.** `train_order.py --dec 1 --epochs 1 --keep-every 2500`, ordered
and symmetric, seeds 42 and 43 (`data/out/sonar/slow-order/
{ord,sym}_dec1_sp0.001_ph38_s{42,43}`), 15,420 steps of 256 pairs. The
baseline is the long Fano pair
(`data/out/sonar/next-sentence/long/doc2_fano_sp1e-3_ph38{,_s43}`), which
differs in `--dec` alone. Ordered seed 42 and symmetric seed 43 were
interrupted by a machine crash and resumed from steps 7,750 and 8,000;
each lands on its uninterrupted sibling's figures below. Everything is
`order.py --step 15000` on the cache's last 32,768 rows
(`eval/order_{fano,sym,ord}.json`); κ is on same-document pairs two
segments apart (30,345 pairs).

**One epoch is early for the block.** The baseline's slow heads match
across seeds at NMI 0.145 here, against 0.41 at step 46,250
(`experiments/next-sentence/notes.md`, 2026-11-10). The arms are compared
at one step; this does not say where any of them ends up.

**Per run, the 38 slow heads** (seed 42 / seed 43):

| | Fano | symmetric | ordered |
|---|---|---|---|
| held-out FVU_w (whole model) | 0.5229 / 0.5223 | 0.5203 / 0.5198 | 0.5206 / 0.5201 |
| κ, mean | 0.195 / 0.194 | 0.105 / 0.102 | 0.125 / 0.124 |
| κ against head index (rank correlation) | +0.11 / +0.10 | −0.27 / +0.30 | −0.92 / −0.92 |
| NMI with language, mean | 0.152 / 0.152 | 0.083 / 0.082 | 0.098 / 0.099 |
| NMI with language against head index | +0.03 / +0.14 | −0.27 / +0.35 | −0.93 / −0.92 |
| NMI with earlier heads, first half of the block | 0.135 / 0.136 | 0.029 / 0.025 | 0.088 / 0.087 |
| NMI with earlier heads, second half | 0.140 / 0.141 | 0.029 / 0.029 | 0.035 / 0.037 |

Ordered seed 43 by head: κ 0.178, 0.197, 0.172, 0.158, 0.166 for heads
0–4, 0.145 at head 9, 0.144 at head 18, 0.090 at head 28, 0.087 at
head 37. The other arms' κ ranges over heads: Fano 0.153–0.247,
symmetric 0.084–0.144 (seed 42).

**Seed 42 against seed 43, the slow block:**

| | Fano | symmetric | ordered |
|---|---|---|---|
| matched NMI | 0.1453 | 0.0318 | 0.0550 |
| any pairing | 0.1392 | 0.0288 | 0.0446 |
| two slow heads of one seed | 0.139 / 0.139 | 0.029 / 0.028 | 0.044 / 0.044 |
| best partner over runner-up (median) | 0.0014 | 0.0015 | 0.0037 |
| head i against head i | 0.1394 | 0.0287 | 0.0521 |
| match keeps the index (chance 2.6%) | 5.3% | 2.6% | 5.3% |
| best partner is the own index (chance 2.6%) | 2.6% | 2.6% | 2.6% |
| index against matched partner's index (rank correlation) | −0.15 | −0.12 | +0.89 |

**NMI by how far apart two heads' indices are**, ordered arm:

| offset | 0 | 1 | 2 | 3–5 | 6+ |
|---|---|---|---|---|---|
| across seeds | 0.0521 | 0.0520 | 0.0510 | 0.0498 | 0.0423 |
| within one seed | | 0.0518 | 0.0512 | 0.0494 | 0.0420 |

The Fano and symmetric arms are flat in the offset, across seeds and
within (0.139–0.140 and 0.028–0.029).

**Readings.**
- **The term does what it was built to do within a run.** Both arms cut
  the NMI between slow heads to a fifth to a third of the baseline's.
  Under the ordered term the first heads stay as slow as a baseline head
  and slowness falls with the index, alike in both seeds; under the
  symmetric term every head gives up about the same amount and the index
  means nothing (the trend's sign differs between seeds).
- **The order reproduces, as a gradient.** In the ordered arm a head's
  matched partner in the other seed sits near its own index (rank
  correlation +0.89), and head i against head i scores 0.052 where any
  pairing scores 0.045.
- **It does not identify heads.** The other seed's head i is no closer
  than the other seed's head i ± 1 (0.0521 against 0.0520), and no closer
  than a neighbouring head of the same seed (0.0518), which the term has
  pushed away. The match keeps the exact index at chance. What two heads
  share depends on how early both are, the same across seeds as within
  one: the early heads still share the slowest content (NMI 0.088 among
  the first half), and a pairing that follows the index pairs heads with
  like amounts of it.
- **In every arm the seeds share what the heads share with each other
  and no more.** Any pairing across seeds equals two heads of one seed:
  0.139 for both, 0.029 for both, 0.045 against 0.044. Decorrelation
  shrinks the common part; it does not put anything reproducible in its
  place. The best match's gain over any pairing is 0.006 (Fano), 0.003
  (symmetric) and 0.010 (ordered), and the ordered arm's is the gradient.
- **Neither arm pays in reconstruction** at this step (FVU within 0.003
  of the baseline, on the lower side). Decorrelated heads are less slow
  on average (κ 0.10 and 0.12 against 0.195).

**Caveats.** One epoch, one weight (`--dec 1`), two seeds. The slow
heads are far from independent at this weight (NMI 0.03–0.09 where
shuffled rows give 0.005), so "decorrelated" overstates both arms. The
three epochs the baseline block takes to form are below; a larger
`--dec` was not run.

## 2026-11-11: the ordered arm at three epochs: a sharper gradient, still no head identified

**Runs.** Both ordered runs continued to three epochs
(`train_order.py --dec 1 --epochs 3` on the same directories; step
46,261, checkpoints every 2,500), as the long Fano runs were. `order.py`
at steps 30,000 and 45,000 for both arms (`eval/order_{ord,fano}_{30000,
45000}.json`). The symmetric arm was not continued.

**Seed 42 against seed 43, the slow block:**

| | Fano 15k | 30k | 45k | ordered 15k | 30k | 45k |
|---|---|---|---|---|---|---|
| matched NMI | 0.145 | 0.309 | 0.408 | 0.055 | 0.112 | 0.156 |
| any pairing | 0.139 | 0.298 | 0.393 | 0.045 | 0.081 | 0.110 |
| head i against head i | 0.139 | 0.298 | 0.394 | 0.052 | 0.107 | 0.151 |
| best partner over runner-up (median) | 0.0014 | 0.0037 | 0.0045 | 0.0037 | 0.0022 | 0.0026 |
| match keeps the index (chance 2.6%) | 5.3% | 2.6% | 2.6% | 5.3% | 2.6% | 7.9% |
| index against matched partner's (rank correlation) | −0.15 | +0.37 | +0.25 | +0.89 | +0.94 | +0.94 |
| head i over its index neighbours (± s.e.) | +0.0004 ± 0.0016 | +0.0009 ± 0.0017 | +0.0013 ± 0.0020 | −0.0003 ± 0.0009 | −0.0000 ± 0.0014 | +0.0013 ± 0.0016 |

The last row is `neighbour_gap`: a head's NMI with the other seed's head
of its index, minus the mean of the four entries one index away.

**Ordered arm at step 45,000, NMI by index offset:**

| offset | 0 | 1 | 2 | 3–5 | 6+ |
|---|---|---|---|---|---|
| across seeds | 0.1508 | 0.1483 | 0.1459 | 0.1425 | 0.0972 |
| within one seed | | 0.1473 | 0.1466 | 0.1408 | 0.0962 |

The Fano arm is flat at 0.393 at every offset, across seeds and within.

**Per run at step 45,000** (seed 42 / seed 43):

| | Fano | ordered |
|---|---|---|
| held-out FVU_w (whole model) | 0.4674 / 0.4662 | 0.4616 / 0.4604 |
| κ, mean over slow heads | 0.451 / 0.445 | 0.251 / 0.252 |
| κ against head index (rank correlation) | −0.15 / +0.07 | −0.98 / −0.98 |
| NMI with language, mean | 0.322 / 0.318 | 0.186 / 0.189 |
| NMI with earlier heads, first / second half | 0.39 / 0.39 | 0.27 / 0.07 |

Ordered seed 42 by head: κ 0.466 at head 0, 0.419 at head 1, 0.232 at
head 18, 0.088 at head 37. Its head 0 matches seed 43's head 0 at NMI
0.373; the Fano arm's head 0 pair is at 0.378.

**Readings.**
- **Longer training sharpens the gradient and identifies nothing.** The
  matched partner's index tracks a head's own at rank correlation 0.94
  and slowness falls with the index at −0.98, but a head is 0.0013 ±
  0.0016 closer to the other seed's head of its index than to the heads
  one index away. The Fano arm, which has no order, reads the same
  +0.0013 ± 0.0020. The exact index is kept for 3 of 38 heads.
- **The first heads are the baseline's block; the rest thin out.** Head 0
  is as slow as a Fano head (κ 0.47) and matches across seeds as well
  (0.37 against 0.38). The first half of the block is still redundant
  (NMI 0.27 with earlier heads against the baseline's 0.39), and the
  second half is nearly decorrelated (0.07) and barely slow (κ 0.09 at
  head 37, where the 15,000-step value was 0.08: the late heads have not
  found slower content to take).
- **So the term at this weight rations the one slow thing, it does not
  find a second.** Each head gets as much of the block's shared content
  as the heads before it leave, and the amounts reproduce across seeds;
  no head carries content of its own that the other seed's counterpart
  has and its neighbours lack. This is the linear picture of 2026-11-03
  (`experiments/next-sentence/notes.md`) again: every slow head on linear
  slow direction 1 or 2.
- **It costs no reconstruction.** FVU 0.461 against the Fano arm's 0.467
  at the same step: the ordered block takes less from the rest of the
  model than the Fano block does. For scale, seed 44's control reads
  0.460 and its Fano arm 0.469 at step 46,250
  (`experiments/next-sentence/notes.md`, 2026-11-10); the control was not
  scored at step 45,000 here.

**Caveats.** Two seeds, one weight. Weights large enough to decorrelate
the first half as well are in the next entry, at one epoch. The test for
an identified head compares against index neighbours; a head identified
only up to a small shift in index would be missed by the exact-index rows
and partly by the neighbour gap, though the one-seed neighbours (offset 1
within, 0.1473) bound it: the other seed's counterpart is 0.0035 above
them.

## 2026-11-11: larger weights, one epoch: head 0 is identified, and it is the only slow head left

**Why larger.** The term is a mean over all 703 pairs of slow heads, so
head i is pushed off its predecessors with weight `dec` · i / 703 while
its Fano reward has weight 1/38. At `--dec 1` head 1 feels the term at a
eighteenth of its reward, which is why the first half of the block stayed
redundant.

**Runs.** `train_order.py --dec 10` and `--dec 100`, ordered, seeds 42
and 43, one epoch (`ord_dec{10,100}_sp0.001_ph38_s{42,43}`). The
`--dec 100` runs were interrupted near step 12,000 by a second machine
crash and resumed from step 11,500, one seed at a time. `order.py --step
15000` as before (`eval/order_ord_dec{10,100}.json`), and on the pair
objective's control, which has no pair term
(`long/doc2_sp0_ph38{,_s43}`, `eval/order_control.json`), as the
reference for a head nothing has made slow.

**Per run, the first 38 heads** (seed 42 / seed 43):

| | control | Fano | `--dec 1` | `--dec 10` | `--dec 100` |
|---|---|---|---|---|---|
| held-out FVU_w | 0.5199 / 0.5194 | 0.5229 / 0.5223 | 0.5206 / 0.5201 | 0.5234 / 0.5234 | 0.5239 / 0.5241 |
| κ, mean | 0.067 / 0.065 | 0.195 / 0.194 | 0.125 / 0.124 | 0.061 / 0.059 | 0.051 / 0.049 |
| κ, head 0 | 0.068 / 0.059 | 0.185 / 0.183 | 0.187 / 0.178 | 0.184 / 0.181 | 0.186 / 0.183 |
| κ, mean of heads 1–37 | 0.067 / 0.065 | 0.196 / 0.195 | 0.124 / 0.122 | 0.057 / 0.055 | 0.047 / 0.045 |
| NMI with language, mean | 0.046 / 0.046 | 0.152 / 0.152 | 0.098 / 0.099 | 0.041 / 0.040 | 0.032 / 0.031 |
| NMI with earlier heads, first / second half | 0.018 / 0.018 | 0.135 / 0.140 | 0.088 / 0.035 | 0.019 / 0.007 | 0.009 / 0.007 |

At `--dec 10` κ still falls over the first heads (seed 42: 0.184, 0.130,
0.111 for heads 0–2, 0.080 at head 4, 0.063 at head 9); at `--dec 100`
head 1 is already at 0.052 and heads 1–37 range over 0.038–0.060.

**Seed 42 against seed 43, the first 38 heads:**

| | control | Fano | `--dec 1` | `--dec 10` | `--dec 100` |
|---|---|---|---|---|---|
| matched NMI | 0.0186 | 0.1453 | 0.0550 | 0.0134 | 0.0105 |
| any pairing | 0.0172 | 0.1392 | 0.0446 | 0.0087 | 0.0071 |
| head 0 against head 0 | 0.0178 | 0.1147 | 0.1147 | 0.1142 | 0.1156 |
| head 0 against the other seed's heads 1 and up | 0.0176 | 0.1289 | 0.0695 | 0.0183 | 0.0126 |
| heads 1 and up against each other | 0.0172 | 0.1398 | 0.0432 | 0.0081 | 0.0067 |
| head i over its index neighbours, heads 1–36 (± s.e.) | −0.0000 ± 0.0002 | +0.0004 ± 0.0016 | −0.0003 ± 0.0009 | −0.0003 ± 0.0002 | −0.0002 ± 0.0002 |
| seed 42's heads whose best partner in seed 43 is head 0 | 0 of 38 | 0 | 1 | 37 | 38 |

The three head-0 rows are read from the saved `cross` matrices (row and
column 0); `order.py`'s own statistics leave the block's two end heads
out of the neighbour gap.

**Readings.**
- **A large weight identifies one head: head 0.** At `--dec 100` seed
  42's head 0 shares 0.116 with seed 43's head 0 and 0.013 with seed 43's
  other slow heads; at `--dec 1` those were 0.115 and 0.070, and in the
  Fano arm head 0 matched other heads better than its counterpart. Head 0
  is the one head the ordered term never touches, and its figures are the
  Fano arm's head 0 at every weight (κ 0.18–0.19, 0.115 across seeds, NMI
  with language 0.152). The term does not change it. It makes it the only
  head of its kind.
- **No second slow variable appears.** The heads after it do not find
  other slow content when barred from head 0's. They stop being slow:
  heads 1–37 read κ 0.047 at `--dec 100`, below the 0.067 of the control,
  which has no pair term at all, and they share less across seeds (0.0067)
  and with language (0.03) than control heads do (0.0172, 0.046). Being
  independent of head 0 costs a head the little document-level content an
  untrained-for head carries anyway.
- **In between, the gradient of the earlier entries.** At `--dec 10` the
  first few heads keep decreasing fractions of head 0's content: nearly
  every head's best partner in the other seed is head 0, and the cross-seed
  matrix's corner is close to a product (0.114, 0.083, 0.054 along row 0;
  0.083, 0.057, 0.039 along row 1). Past head 8 the block is at the level
  of `--dec 100`.
- **The Fano bound loses 37 of its 38 heads for this.** The block's mean
  κ is the control's or below at both weights. Reconstruction is 0.001
  worse than the Fano arm's and 0.004 worse than the control's.

So the answer to the last entry's open question, at one epoch: with
these pairs and this bound there is one slow thing, and an ordered
decorrelation at full strength gives it to one head and leaves the rest
with nothing. That is an identified head, obtained by removing the
redundancy rather than by finding more variables.

**Caveats.** One epoch and two seeds, where the Fano block takes three
epochs to form (head 0's κ is 0.47 at step 45,000 at `--dec 1`, and the
Fano arm's heads average 0.45); whether later heads find slow content of
their own with
longer training at these weights is not tested. Head 0's identification
rests on one head in two seeds. The term scores dependence between
entries, so a second variable carried jointly with head 0's content would
have to be separated from it to count; nothing here rules that out.


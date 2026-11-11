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
shuffled rows give 0.005), so "decorrelated" overstates both arms. A
larger `--dec` or the three epochs the baseline block takes to form
could change the picture; neither was run.

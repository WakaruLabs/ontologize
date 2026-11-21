# planted-circle: notes

## 2026-11-20: the circle's cuts wander, a tilt pins them partly, and a third of seeds mix the factors

**Short version.** In seeds whose two heads each carry one factor, the circle's cuts are as far
apart between seeds as cuts placed independently (cut distance 0.52, against 0.5), while the
partition NMI still reads 0.72. Equal arcs at independent offsets would read 0.76, so NMI near
0.7 is compatible with seeds that agree on no cut at all. A density that is not uniform around the
circle (tilt 0.9) brings the cuts closer (0.40). The segment's cuts, which its ends should pin,
are only partly pinned (0.35 and 0.30), and its NMI is no higher than the circle's. A third to
half of the seeds never separate the factors: both heads carry about a quarter of each, and those
seeds fit worse (FVU 0.094–0.098 against 0.064–0.065). In this toy, a worse optimum explains more
of the disagreement between seeds than the circle's symmetry does.

**Setup.** `circle.py`, d = 16, sigma 0.05, circle radius 1, segment sd 1; one layer of two heads
at k = 8, planted-hierarchy's builder and dictenc-toy's schedule (4,000 steps, annealed to
straight-through selection); the same data for every seed, 32 model seeds per arm, 16,384 held-out
rows. A seed is clean when one head carries at least 0.35 of each factor's entropy (32 bins; at
most 0.6 at k = 8) and no other head more than 0.1 of it. Cuts are read on 64 bins.

| | uniform circle | tilt 0.9 |
|---|---|---|
| clean seeds | 21 / 32 | 15 / 32 |
| FVU, clean / mixed | 0.064 / 0.098 | 0.065 / 0.094 |
| circle cut distance, clean pairs | 0.521 | 0.398 |
| segment cut distance, clean pairs | 0.349 | 0.296 |
| circle NMI, clean pairs (equal arcs at random offsets: 0.760) | 0.722 | 0.735 |
| segment NMI, clean pairs | 0.688 | 0.700 |
| circle / segment NMI, all pairs | 0.589 / 0.559 | 0.575 / 0.563 |

The shuffled-rows NMI is 0.001 throughout. Cut distance is the mean distance from each cut of one
seed to the nearest cut of the other, over half the mean gap between cuts, both ways: 0 for the
same cuts, about 0.5 for independent ones, 1 for cuts offset by half a gap. Clean pairs number 210
and 105.

- **The circle behaves as its symmetry predicts.** Its cuts are placed independently across
  seeds, and its NMI sits just under what rotation alone gives. A tilt of 0.9 (density from 0.1 to
  1.9 around the circle) pins them part of the way.
- **NMI cannot see this.** Free rotation of eight equal arcs costs NMI only 0.24, so a partition
  NMI of 0.7 between two seeds does not mean they cut in the same places.
- **Against SONAR.** The stack's layer-0 heads match across seeds at NMI 0.025
  (`recipe-ladder/notes.md`), far below the 0.72 of seeds that share a factor but cut it at
  different places. So SONAR's heads do not share a factor and differ only in where they cut it;
  they differ in what they cut, as this toy's mixed seeds do, or there is no shared factor to
  cut.
- **Open.** Whether longer training, or more heads than factors, lowers the share of mixed
  seeds; and the plane's recurrence, which this run does not measure (it needs each head's
  decoded atoms).

```bash
uv run python experiments/planted-circle/circle.py --seeds 32
uv run python experiments/planted-circle/circle.py --seeds 32 --tilt 0.9
```

Logs: `experiments/planted-circle/out/run_tilts32.log` (and `run_tilts.log`, the 8-seed runs at
tilts 0, 0.5 and 0.9, scored before the clean-seed rule).

## 2026-11-20: mixing is not under-training, and spare heads all mix

**Short version.** Four times the training leaves the share of seeds that keep each head to one
factor where it was (19/32 against 21/32), so the mixed seeds are a stable outcome, not runs that
stopped early. With four heads for the two factors, no seed keeps any head to one factor (about
two heads per seed carry more than 0.1 of each), and those seeds fit better than the two-head
ones. Once heads outnumber the factors, the solutions training reaches do not align heads with
factors at all.

A head is mixed when it carries more than 0.1 of each factor's entropy; a seed with no mixed head
has separated the factors however many heads it spends on each. The two-head counts are read from
the earlier runs' saved shares. Uniform circle, 32 seeds:

| arm | seeds with no mixed head | mixed heads per seed | FVU, unmixed / mixed |
|---|---|---|---|
| h = 2, 4,000 steps | 21 / 32 | 0.69 | 0.064 / 0.098 |
| h = 2, 16,000 steps (anneal 12,000) | 19 / 32 | 0.78 | 0.062 / 0.094 |
| h = 4, 4,000 steps | 0 / 32 | 1.97 | — / 0.040 |
| h = 2, tilt 0.9 | 15 / 32 | 0.88 | 0.065 / 0.094 |

- **Longer training pins the segment a little more and the circle not at all.** Clean pairs: cut
  distance 0.487 for the circle and 0.286 for the segment (0.521 and 0.349 at 4,000 steps); NMI
  0.727 and 0.743.
- **Four heads fit better and align with nothing.** Their FVU of 0.040 spends 12 bits against two
  heads' 6, so the fit is no surprise; that none of 32 seeds puts a head on one factor is the
  result. The cut statistics are not defined there (a factor's cuts are shared between heads), so
  only the all-pairs line is in the log.
- **Against SONAR.** Its layers have 76 heads at k = 32, far more heads than any factor count the
  data could plausibly have, which this toy says is the regime where heads stop aligning with
  factors. That fits the stack's head agreement sitting near the null while a single k-means
  partition of the same rows reproduces (`seed-stability/notes.md`, the k-means baseline).

```bash
uv run python experiments/planted-circle/circle.py --seeds 32 --steps 16000 --anneal-steps 12000 \
    --name l1h2k8_tilt0_x32_steps16k
uv run python experiments/planted-circle/circle.py --seeds 32 --h 4
```

Log: `experiments/planted-circle/out/run_followup.log`.

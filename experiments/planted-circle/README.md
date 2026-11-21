# planted-circle

Head disagreement across seeds where no structure is missing. A circle
has no preferred place to cut it, so a partition of its angle into arcs is
fixed only up to rotation: two seeds should cut the same circle at
different places while carrying the same information about the angle. A
segment has ends, which pin its cuts. `circle.py` plants both in one data
set and asks whether the heads behave that way, and whether a density
that is not uniform around the circle (`--tilt`) pins the cuts.

    x = R (cos θ u₁ + sin θ u₂) + t u₃ + sigma * eps

u₁, u₂ and u₃ are orthonormal in R^d; θ is uniform, or of density
proportional to 1 + tilt · cos θ; t is uniform with standard deviation
`--segment-sd`. The data are the same for every seed and only the model
seed changes, as in a seed pair on a cache. The model, its builder and its
training schedule are `experiments/planted-hierarchy/tree.py`'s
(dictenc-toy's: an anneal from soft to straight-through selection), by
default one layer of two heads with k = 8. It runs on CPU in about a
minute for eight seeds.

## What it reports

Per seed, on held-out rows:
- which head carries θ and which t: the share of each factor's entropy
  (32 bins) that the head's entry carries, at most log 8 / log 32 = 0.6;
- each head's cuts along its factor: the bin edges (64 bins) at which the
  modal entry changes, with runs shorter than two bins folded into their
  neighbour so a flickering boundary counts once.

A seed is clean when one head carries at least `--clean-share` (0.35) of
each factor's entropy and no other head more than `--cross-share` (0.1)
of it. The pair statistics are reported over the clean seeds, where a
head's cuts along its factor mean something, and over all seeds.

Between every pair of seeds:
- the cut distance: from each cut of one seed to the nearest cut of the
  other, over half the mean gap between cuts, averaged both ways. 0 is
  the same cuts, about 0.5 cuts placed independently, 1 cuts offset by
  half a gap;
- the partition NMI of the two circle heads and of the two segment heads,
  beside one with rows shuffled. For reference, two partitions of the
  circle into m equal arcs at independent offsets have expected NMI
  1 − 1/(2 ln m), 0.76 at m = 8: NMI barely sees cuts that wander.

## Run

```bash
uv run python experiments/planted-circle/circle.py --seeds 8
uv run python experiments/planted-circle/circle.py --seeds 8 --tilt 0.5
```

Output: `experiments/planted-circle/out/<name>/summary.json` with every
seed's shares, cuts and FVU, and every pair's cut distances and NMIs.
`tests/test_planted_circle.py` checks the tilted sampler, the cut finder on
planted arcs, intervals and a flickering boundary, the cut distance, the
NMI and its rotated-arcs reference.

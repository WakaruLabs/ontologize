# modularity: are heads communities of what they classify?

A head partitions the samples into k cells. `headstruct.py` already asks
whether SAE latents can be grouped into units that *behave* like heads
(exhaustive, exclusive, constant sum). This experiment asks a graph-side
question of both model families: is the partition a head induces a
community structure of the input it classifies, i.e. do its cells hold
more affinity than a degree-preserving null graph would put there?

The measure is soft modularity (Newman 2006; the fuzzy form of Zhang et
al. 2007). Each scored unit is a head (an Ontologizer head, a
`sae.py --groups` head) or a group discovered post hoc among SAE latents
(`headstruct.py`'s default); its soft cells are its members' activations
normalized within the unit:

    Q = (1/2m) [ tr(P^T D P) - gamma * sum_c (k^T P_c)^2 / 2m ]

on a per-batch affinity graph D. It is the graph-clustering objective of
Leiden-on-kNN pipelines (Traag et al. 2019) and of DMoN (Tsitsulin et al.
2023), which trains it directly on softmax assignments. Soft modularity
is the within-cell affinity that `fns/pwak.py`'s `PP^T` gate already
builds, minus a null term that, under uniform degrees, is the collision
probability of cell usage: an `s_Hm`-like anti-collapse term with gamma
as the trade-off.

## Files

No code lives here; everything is a mode of `headstruct.py` at the repo
root, with tests in `tests/test_headstruct.py`.

- `run.sh` -- the runs recorded in `notes.md` (and, in a comment, the
  training command for the one SAE trained for this experiment).
- `notes.md` -- results.

`headstruct.py` flags used:

- `--modularity` scores each unit's sample partition; `--gamma` is the
  resolution, `--tau` the heat-kernel temperature of
  `ontologize.fns.pwak.affinity`.
- `--knn K` sparsifies each graph to every sample's K strongest edges
  (union, kernel weights kept). The dense heat kernel on unit-norm SONAR
  embeddings is nearly uniform, so every partition scores Q ~ 0 on it;
  `--knn 15` is what the recorded runs use.
- `--onto CKPT` scores an Ontologizer's heads, each (layer, head) a unit
  of k entries. Each layer is scored on the graph of its own input: X for layer 0,
  the residual it classifies above that
  (`autointerp.onto_acts_fn(..., inputs=True)`).
- `--prefix-layers` treats a Matryoshka SAE's (`sae.py --prefixes P`)
  prefix blocks as layers: groups are discovered within blocks and block L
  is scored on its prefix residual `X - decode(code on blocks < L)`. This
  is the SAE control for the per-layer reading.

## Design

- **The null is size-matched regrouping** of the same latents, permuted
  within a layer or block. Any partition of points by firing region is
  somewhat spatially coherent, so raw Q has no ideal value. Read the
  real/null ratio; the z-scores are inflated by the small spread of null
  means.
- **The graph is restricted to the samples a unit fires on**, so a
  sparse group is not penalized for coverage (exhaustiveness measures
  that). Per-block rows report coverage alongside Q.
- **Soft cells near uniform force Q toward 0** regardless of structure.
  A near-uniform softmax head scores ~0 by construction, so read Q
  alongside the assignment entropy.
- **What Q measures is alignment with the graph's dominant cluster
  structure, not quality as a variable.** A head encoding one of many
  independent factors cuts across the joint kNN graph of its input,
  since a sample's neighbours share most factors and not necessarily
  this one. Low Q on X is the expected reading for a factorial code.
- **Matryoshka asymmetry.** An Ontologizer layer classifies its residual;
  a Matryoshka block is encoded from X by the shared encoder and is only
  trained to reconstruct the prefix residual. The control matches the
  graph, not what the block sees.

## Usage

    bash experiments/modularity/run.sh

Each run writes `groups.csv` (a trailing `modularity` column),
`assignment.npy` and `run.log` to its output directory, listed in
`run.sh`.

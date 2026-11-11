# planted-hierarchy: does residual depth order a hierarchy?

The stack's `l` residual layers are often described as a hierarchy:
coarse structure first, refinements after. Nothing in the repo tests that
against a known answer. `experiments/dictenc-toy` has ground truth, for
one layer and independent factors. This experiment plants a two-level
tree and asks where a two-layer stack puts each level.

## The planted tree

After the synthetic benchmark of Costa et al. (2025), itself adapted
from Bussmann et al. (2025): 11 parent concepts, the first 3 of which
have 3 children each. A sample draws one parent uniformly and, if the
parent has children, adds one of them with probability 0.75:

    x = P[z_p] + child_scale * C[z_c] + sigma * eps

One atom per parent and per child, N(0, 1/2) per coordinate, in d = 64
with sigma = 0.3. The parent z_p has 11 levels. The child z_c has 10:
none (a zero atom) or one of nine, each of which occurs only under its
own parent. Coefficients are 1, the model class of a head (one entry, one
atom); the benchmark's real-valued magnitudes are not planted.

In a tree the parent alone carries much of the child: it says whether a
child is possible. So a head is scored on the child twice, as below.

## What could put a factor in a layer

| | option | removes |
|---|---|---|
| the hierarchy | `--shuffle-tree` permutes z_c across samples | the dependence of child on parent, keeping both marginals |
| variance | `--child-scale 3` | the parent's lead in variance: the child component then has more |
| depth | `--l 1 --h 2`, one layer of two heads | any order between the heads, at the same capacity |
| the prefix loss | `--no-deepsup` | the loss on layer 0's own reconstruction |
| spare entries | `--k 11` against the default 16 | a head's room to hold the parent and part of the child at once |

The model is `dictenc-toy`'s with `l` layers: `forward="resid"`, the
constant coordinate, no gain-shape split, a free linear decoder,
straight-through selection, and that experiment's schedule (4,000 steps).

## Run

```bash
uv run python experiments/planted-hierarchy/tree.py --seeds 5                  # the stack on the tree
uv run python experiments/planted-hierarchy/tree.py --seeds 5 --shuffle-tree
uv run python experiments/planted-hierarchy/tree.py --seeds 5 --l 1 --h 2      # flat
experiments/planted-hierarchy/grid.sh                                          # every condition and arm
```

CPU, about 16 s a seed. Each run writes `out/<name>/summary.json`;
`grid.sh` also keeps `out/<name>.log`.

## Scores

On 8,192 held-out samples, per head (`heads` in the summary):

| score | reads |
|---|---|
| `parent` | I(entry; z_p) / H(z_p): the share of the parent the head carries |
| `child` | the same for z_c. A head that carries the parent and nothing else reads about 0.6 on the tree |
| `child_new` | I(entry; z_c given z_p) / H(z_c given z_p): of what the child adds once the parent is known, the share the head carries. This is the score that says which head tells children apart |
| `*_null` | each with rows permuted: the chance level at this sample size |

Per run: `parent_layer` and `child_layer`, the layers of the heads with
the largest `parent` and the largest `child_new`; `fvu_prefix`, the FVU
of the reconstruction through each layer; and three references,
`fvu_oracle` (the planted decomposition: the noise floor),
`fvu_parent_only` and `fvu_child_only` (the best reconstruction from one
factor alone, each sample replaced by its cell's mean).

The tally counts over seeds how often the parent's head is in layer 0
and the child's in the last layer. For the flat arm the corresponding
count is how often the parent's head is head 0, where chance is one half.

Not built: the SAE arms of the plan (a plain and a nested-prefix SAE on
the same data, for the absorption the benchmark's authors report).

Tests: `tests/test_planted_hierarchy.py`.

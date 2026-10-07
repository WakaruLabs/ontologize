# modularity: notes

Common settings for every run below: `run.sh`, 1,048,576 rows of
`mc4_4M.npy` (the first half for discovery, the second for scoring),
batch 4096, 20 size-matched nulls, `--knn 15`, `--tau 0.2`,
`--gamma 1`, `--fire-thr 0`. One checkpoint per model and one seed.
"Ratio" is real Q / null Q. The z-scores are reported but inflated by the
small spread of null means.

## 2026-10-06: Ontologizer vs SAEs

### Ontologizer heads, each layer on its own input (`bl_ctl_full`, step 369500)

5 layers x 32 heads x 32 entries, `--temperature 0.03`: the run annealed
1.0 -> 0.03 over its first 50k steps (`log.jsonl` `hyper`), so this is its
final temperature. Nulls are permuted within each layer.

| layer | graph | real Q | null Q | ratio | z |
|---|---|---|---|---|---|
| 0 | X | 0.0026 | 0.0021 | 1.2 | 24 |
| 1 | residual | 0.0084 | 0.0036 | 2.3 | 63 |
| 2 | residual | 0.032 | 0.011 | 2.9 | 49 |
| 3 | residual | 0.117 | 0.043 | 2.7 | 53 |
| 4 | residual | 0.218 | 0.105 | 2.1 | 36 |
| all | | 0.076 | 0.033 | 2.3 | 58 |

Layers 1-4 have 2-3x the modularity of random within-layer regroupings
of their entries; layer 0 is barely above its null. Raw Q rises with
depth in both columns, partly because assignments sharpen with depth (on
2048 rows: 4.4, 4.4, 3.8, 3.0, 2.3 bits of 5 per head, layers 0-4) and
partly because the residual graphs are more clustered, so the ratio is
the comparable number.

Before per-layer graphs were added, the same heads scored on the graph
of X for every layer gave mean Q 0.001-0.006 per layer (16k rows,
3 nulls). With argmax cells instead of soft ones, layer 0 reached mean
0.021 (best head 0.079) and the later layers ~0.004. So on X the heads
cut across the input's kNN communities, as a factorial code should; the
community structure appears only on each layer's own input.

### Matched-shape SAE (`m5120_g160softmax`, `--trained-groups`)

160 trained softmax heads of 32, graph of X.

Real Q 3.6e-5 per head (range 3.6-4.0e-5); null ~3e-5. Sum CV 0.0001 and
full coverage: the heads' softmaxes are close to uniform over their 32
latents, which forces Q to ~0. Uninformative on this measure without
sharpened cells.

### Flat trained-head SAE (`m5120_g160top1`, `--trained-groups`)

160 trained top-1 heads of 32, graph of X. Exhaustiveness and
exclusivity are both exactly 1: every head assigns every sample to
exactly one latent, so the cells are hard and Q is informative here,
unlike the softmax-head run.

| | real Q | null Q | ratio | z |
|---|---|---|---|---|
| modularity | 0.024 | 0.027 | 0.90 | -2.3 |

Slightly below null. The nulls are not partitions (exhaustiveness 0.64,
exclusivity 1.5: a regrouped set of latents from different heads can
leave a sample uncovered or fire twice), so each null cell structure is
scored on its own fired subgraph with shared mass where members co-fire.
The comparison is still size-matched, but the real heads are the only
units here that are exact partitions.

### Top-k SAE, discovered groups (`m5120_k32`)

164 groups (median size 32) covering 4247/5120 latents, graph of X.

| | real Q | null Q | z |
|---|---|---|---|
| modularity | 0.099 | 0.131 | -10.9 |

Per group: median 0.076, max 0.55. The discovered groups are *less*
modular than random regroupings of the same latents. Discovery selects
paradigmatic groups (latents that fire in the same contexts but not
together), and mutually exclusive alternatives in one context are
neighbours in X, so such a group splits a neighbourhood. A random group
mostly collects latents firing in different regions, whose cells are
already separated.

### Matryoshka SAE, prefix blocks as layers (`m5120_k32_p5`)

5 prefix blocks of 1024, `--prefix-layers`: groups discovered within
blocks, nulls permuted within blocks, block L scored on its prefix
residual `X - decode(code on blocks < L)` (block 0: `X - b_dec`). 111
groups (median size 32) covering 3102/5120 latents.

| block | groups | coverage | real Q | null Q | ratio | z |
|---|---|---|---|---|---|---|
| 0 | 32 | 0.497 | 0.116 | 0.113 | 1.03 | 0.9 |
| 1 | 27 | 0.162 | 0.099 | 0.112 | 0.88 | -4.8 |
| 2 | 24 | 0.078 | 0.063 | 0.071 | 0.89 | -3.1 |
| 3 | 27 | 0.033 | 0.012 | 0.004 | 2.7 | 1.8 |
| 4 | 1 | 0.002 | -0.064 | -0.064 | -- | -- |
| all | 111 | | 0.073 | 0.076 | 0.97 | -1.8 |

Coverage is the groups' mean exhaustiveness: under the global top-k most
of a sample's active latents sit in the early blocks. Block 4 is
uninformative by construction: it holds one group, and permuting that
group's members among themselves reproduces it, so real equals null.
Block 3's ratio rests on Q ~0.01 at 3% coverage (a group fires on ~135
samples per 4096-row batch, a thin kNN subgraph) and does not clear the
null spread (z 1.8).

Blocks 0-2, which carry nearly all the coverage, sit at or below their
nulls (0.88-1.03), where Ontologizer layers 1-4 sit at 2.1-2.9 with full
coverage. A coarse-to-fine dictionary scored on matched prefix-residual
graphs does not reproduce the Ontologizer's per-layer result.

Confound: these are discovered groups, and discovery selects paradigmatic
groups, which already scored below null on X for the flat `m5120_k32`.
The comparison mixes "residual graphs vs model" with "trained heads vs
post hoc groups". The clean control is a Matryoshka SAE with trained
heads, `sae.py --m 5120 --topk 0 --groups 160 --prefixes 5` (top-1 heads;
groups must be a multiple of prefixes), scored with
`--trained-groups --prefix-layers`.
No such run exists yet.

### Trained-head Matryoshka SAE (`m5120_g160top1_p5`, `--trained-groups --prefix-layers`)

Trained for this experiment: `m5120_g160top1` plus 5 nested prefixes,
`sae.py --m 5120 --topk 0 --groups 160 --prefixes 5 --lr 4e-4 --epochs 150`
(144,300 steps; FVU_w 0.2935, 79/5120 dead latents). 160 top-1 heads of
32, 32 per prefix block; block L scored on its prefix residual.

| block | heads | coverage | real Q | null Q | ratio | z |
|---|---|---|---|---|---|---|
| 0 | 32 | 1.000 | 0.0683 | 0.0646 | 1.06 | 2.5 |
| 1 | 32 | 1.000 | 0.0378 | 0.0393 | 0.96 | -1.5 |
| 2 | 32 | 1.000 | 0.0316 | 0.0324 | 0.98 | -1.0 |
| 3 | 32 | 1.000 | 0.0246 | 0.0254 | 0.97 | -1.7 |
| 4 | 32 | 1.000 | 0.0154 | 0.0157 | 0.98 | -0.9 |
| all | 160 | | 0.0356 | 0.0354 | 1.01 | 0.2 |

Exact partitions (exhaustiveness and exclusivity 1.000) with full
coverage in every block -- none of the discovered-group control's thin
late blocks -- and at their nulls throughout (0.96-1.06), where
Ontologizer layers 1-4 sit at 2.1-2.9 on the same kind of graph. This
removes the confound in the discovered-group Matryoshka control: trained
heads in a coarse-to-fine SAE, scored on matched prefix-residual graphs,
do not show the effect either.

## Reading so far

On X, the one graph both families share, neither model's heads or groups
are communities of the input. The Ontologizer's positive result is
within-model and depends on scoring each layer on the residual it
classifies. Neither Matryoshka control reproduces it on matched
prefix-residual graphs: discovered groups sit at 0.88-1.03 of their
nulls in the blocks with coverage, and trained top-1 heads at 0.96-1.06
in every block. So it is not explained by residual graphs being more
clustered, nor by trained heads in a coarse-to-fine dictionary: what
differs is that an Ontologizer layer classifies the residual it is
scored on, while a Matryoshka block is encoded from X and only trained to
reconstruct that residual. One checkpoint per model, one seed throughout.

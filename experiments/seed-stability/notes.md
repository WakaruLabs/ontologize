# seed-stability: notes

## 2026-11-19: the SAE seed baseline, remeasured, and beside the field's own statistics

The SAE side of every seed comparison here is the `k32` pair, `sae_conv/m5120_k32` against
`m5120_k32_s43`. Its containment fractions (0.31 / 0.24 / 0.15 / 0.03) were typed into the
findings, `match_heads.py` and `make_figs.py` from a run that no longer had an output on disk.
`splitting.py` was rerun on the pair, and now also prints three statistics the literature reports
seed agreement in, so the containment fractions can be read against them.

**Setup.** `splitting.py`'s defaults: the cache's first 524,288 rows (training rows for both SAEs),
latents firing at least 50 times live (5,114 of 5,120 in A); the agreement statistics both ways,
A->B / B->A. Output in `data/out/sonar/splitting/m5120_k32__m5120_k32_s43/` (`agreement.json`).

**Containment reproduces.**

| τ | A matched | B matched |
|---|---|---|
| 0.3 | 0.307 | 0.313 |
| 0.5 | 0.235 | 0.234 |
| 0.7 | 0.147 | 0.147 |
| 0.9 | 0.030 | 0.030 |

These round to the numbers in the findings, the figure and the scripts, so none of them changes.

**The field's statistics are low too.**

| statistic | this pair | reported for language-model SAEs |
|---|---|---|
| decoder share, cosine > 0.7 (Leask et al.) | 0.331 / 0.334 | 0.94, same-size GPT-2 SAEs from two seeds |
| PW-MCC (Song et al.) | 0.511 (random directions 0.117) | 0.818 TopK on Pythia-160M, 0.790 on Gemma-2-2B |
| mean MPPC (Wang et al.) | 0.384 / 0.385 | 0.81 across SAE seeds on Pythia |

- **Decoder share over thresholds:** 0.489 / 0.500 above cosine 0.4, 0.419 / 0.430 above 0.5,
  0.241 / 0.241 above 0.9.
- **PW-MCC's matched pairs** above 0.4, 0.5, 0.7 and 0.9: 0.445, 0.400, 0.328 and 0.241.
- **MPPC above** 0.4, 0.5, 0.7 and 0.9: 0.392, 0.356, 0.298 and 0.178 (A->B; B->A within 0.004).
- **What it settles.** The findings explained this pair's 3-31% by the containment metric's
  strictness: under the decoder-cosine and activation-correlation metrics, language-model SAEs
  report agreement near 0.8. Under those metrics this pair reads 0.33-0.51. So the gap to
  language-model SAEs is not the metric. Under the same containment metric the pair still agrees
  30 to 300 times as often as the softmax Ontologizer's seed pair (at most 0.1%), which is the
  comparison the findings rest on, and that stands.
- **What it does not settle.** This SAE leaves half the variance unexplained (FVU_w 0.499), and
  the papers' SAEs were trained on language-model activations and, in Song et al., configured to
  maximize PW-MCC. Whether the low agreement comes from SONAR or from the fit is open. A seed pair
  of the GPT-2 `m12160_k32` SAE (master has one seed) would separate them, at one GPU run.
- **PW-MCC matches an earlier reading of the same checkpoints** (0.511, random 0.116), which used
  its own matching code.

```bash
JAX_PLATFORMS=cpu uv run python splitting.py \
    --a data/out/sonar/sae_conv/m5120_k32/params.npz \
    --b data/out/sonar/sae_conv/m5120_k32_s43/params.npz \
    --out data/out/sonar/splitting/m5120_k32__m5120_k32_s43
```

## 2026-11-20: subspaces, against the data's own (`subspace.py`)

**Short version.** The SAE pair's unmatched latents span a subspace that the other seed
reproduces almost to the optimum and about twice as well as the data's principal directions. The
hard stack's layers 0 and 3 agree with each other no more than each agrees with the data, and
layers 1–2 only slightly more in their first few directions. At layer 4 the seeds pick different
data-aligned subspaces. The old notes' rank-8 table (ste-arm, "Subspaces, and why their agreement
is not the architecture's") is φ(8)/chance and reproduces here within 20%, on 32,768 rows. Its
text's "25x to 33x" did not match it, and nor did the appendix's.

**SAE pair** (`sae_conv/m5120_k32`, `_s43`; whitened frame, 32,768 tail rows of `mc4_4M`). Of
5,120 and 5,118 live latents, 33.0% and 33.3% have a match at decoder cosine ≥ 0.7 in the other
seed (Gerasimov et al.: 0.746 on GPT-2). Energy share of seed 42's set inside each subspace, mean
over 10 size-matched draws (`seed_subspace/sae_k32/`):

| set | r | own (ceiling) | other seed's | data's top-r | data's bottom-r | chance |
|---|---|---|---|---|---|---|
| unmatched (3,413) | 8 | 0.078 | 0.075 | 0.031 | 0.0002 | 0.008 |
| | 64 | 0.355 | 0.342 | 0.172 | 0.002 | 0.063 |
| | 128 | 0.564 | 0.547 | 0.306 | 0.007 | 0.125 |
| | 512 | 0.951 | 0.931 | 0.877 | 0.123 | 0.500 |
| matched (1,690) | 8 | 0.081 | 0.079 | 0.061 | 0.0003 | 0.008 |
| | 64 | 0.301 | 0.291 | 0.254 | 0.007 | 0.063 |
| | 128 | 0.457 | 0.439 | 0.383 | 0.019 | 0.125 |

- **The other seed captures 96–97% of the optimum at every rank**, for both sets.
- **The data's own top-r captures about half as much for the unmatched set**, and nearly as much
  for the matched set: matched latents lie more along the data's principal directions.
- So Gerasimov et al.'s "reproducible subspaces" survives the control that removed the stack's
  layer agreement. The SAE's unstable latents re-express a subspace both seeds share beyond the
  data's second moments.

**Hard stack** (`ste_h76_init01`, `ste_h76_i01_s43`, step 369500; each layer's contribution
x̂_{ℓ+1} − x̂_ℓ, whitened and centered; `seed_subspace/ste_h76_init01/`). φ(r, r) over chance
(r/1024), seeds against each other | seed 42 against the data | seed 43 against the data:

| layer | r = 1 | 8 | 32 | 128 |
|---|---|---|---|---|
| 0 | 1023 \| 1021 \| 1021 | 126.9 \| 123.2 \| 120.5 | 31.4 \| 30.2 \| 30.1 | 7.5 \| 7.2 \| 7.2 |
| 1 | 979 \| 683 \| 553 | 121.8 \| 103.7 \| 104.5 | 27.5 \| 26.2 \| 26.0 | 6.2 \| 6.3 \| 6.4 |
| 2 | 842 \| 636 \| 463 | 106.9 \| 97.6 \| 97.1 | 22.6 \| 23.0 \| 23.5 | 4.8 \| 5.5 \| 5.5 |
| 3 | 779 \| 670 \| 645 | 82.1 \| 85.5 \| 86.0 | 13.0 \| 17.3 \| 17.3 | 3.2 \| 4.2 \| 4.2 |
| 4 | 204 \| 266 \| 306 | 10.6 \| 26.9 \| 25.7 | 3.1 \| 6.7 \| 6.8 | 1.7 \| 2.6 \| 2.6 |

The ceiling is d/r: 1024 at r = 1, 128 at 8, 32 at 32, 8 at 128. In raw φ, the top direction is
shared at 0.999 in layer 0, and at 0.96 and 0.82 between seeds in layers 1–2 against 0.45–0.67 with
the data.
- **Layer 0** is the data's principal subspace in both seeds (φ ≥ 0.93 to r = 128).
- **Layers 1–2** agree a little beyond the data in their first 1–16 directions, and not past rank
  32.
- **Layer 3** agrees with the other seed less than with the data from rank 8 on.
- **Layer 4's seeds share under half of what each shares with the data** from rank 2 to rank 32
  (0.39–0.47 of it), and 0.67–0.77 of it at rank 1.
- The full prefix grid φ(i, j) per layer is in `subspace_stack.npz`.

The two statistics differ (energy of a set of directions, against overlap of covariance
eigenbases), so the SAE and the stack are compared by reading, not number to number. The
findings' appendix now gives the stack's φ with its formula, and the SAE's transfer beside it.

```bash
uv run python experiments/seed-stability/subspace.py --mode sae \
    --a data/out/sonar/sae_conv/m5120_k32/params.npz \
    --b data/out/sonar/sae_conv/m5120_k32_s43/params.npz \
    --out data/out/sonar/seed_subspace/sae_k32
uv run python experiments/seed-stability/subspace.py --mode stack \
    --a data/out/sonar/multilingual/ste_h76_init01 \
    --b data/out/sonar/multilingual/ste_h76_i01_s43 --step 369500 \
    --out data/out/sonar/seed_subspace/ste_h76_init01
```

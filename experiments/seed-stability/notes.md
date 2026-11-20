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

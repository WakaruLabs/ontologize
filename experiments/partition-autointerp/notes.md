# partition-autointerp: results

## `resid_nc`, 20 heads (2026-10-07)

Rerun of the whole pipeline after the SONAR encoder fix and the SONAR_NORM
decode scale (`experiments/ste-arm/notes.md`, "The SONAR encoder ran with
dropout"); the first run's outputs were not kept. Defaults: step 372,600
at T=0.03, a stratified sample of 20 heads with 8 entries each, Haiku as
judge, every head also graded against a donor head's description
(`--null`).

| lens | median self accuracy | median null accuracy | mean self / null |
|---|---|---|---|
| entries (decoded dictionary parameters) | 0.000 | 0.000 | 0.031 / 0.029 |
| acts (activation-grounded description) | 0.083 | 0.125 | 0.104 / 0.113 |

Guessing one value per head gets about 0.111. Neither lens shows
above-chance head-level signal: parameter decodes fail at the partition
level as they do per latent, and activation-grounded head descriptions do
no better than a null pairing. The first run read nearly the same
(entries median 0.02; acts 0.083 against 0.125).

Against the per-latent campaign (`data/out/sonar/autointerp/onto/onto`,
cacts), a head's detection accuracy is uncorrelated with the mean
describability of its sampled latents: Spearman +0.12 over the 30
head--lens pairs with both, +0.25 under acts (15 heads, p = 0.37) and
+0.05 under entries (15 heads, p = 0.86). The first run reported +0.07;
`spearman` then ranked tied accuracies by row order, and many head
accuracies tie at 0, so it now averages tied ranks.

The describe stage's GPU decode once ran out of memory beside a
text-fidelity job and left empty outputs while exiting 0; it was rerun
from the finished harvest and texts.

Outputs: `out/` (`heads.npz`, `meta.json`, `descriptions.jsonl`,
`scores.csv`, `comparison.csv`).

```bash
uv run python experiments/partition-autointerp/headinterp.py harvest \
    --ckpt data/out/sonar/multilingual/resid_nc
uv run python autointerp.py texts \
    --features experiments/partition-autointerp/out/heads.npz \
    --out data/out/sonar/autointerp
uv run python experiments/partition-autointerp/headinterp.py describe \
    --mode both --device cuda
uv run python experiments/partition-autointerp/headinterp.py score --null
uv run python experiments/partition-autointerp/headinterp.py compare \
    --latent-scores data/out/sonar/autointerp/onto/onto/scores.csv \
    --latent-mode cacts
```

# task-naturalness: results

## The steering overlay's decodes on `resid_nc_hm` (2026-10-07)

Rerun on `experiments/steering-overlay/out/resid_nc_hm_4380000/steers.jsonl`
(960 base / steered pairs) after the SONAR encoder fix and the SONAR_NORM
decode scale, with the NLL pass. chrF between each steered decode and its
unsteered decode, by arm and magnitude:

| arm | 0.25 | 0.5 | 1.0 |
|---|---|---|---|
| difference of means | 0.365 | 0.242 | 0.149 |
| classification forcing | 0.368 | 0.227 | 0.148 |
| linear probe | 0.396 | 0.296 | 0.196 |
| random direction | 0.377 | 0.292 | 0.193 |

All four arms, the random direction included, damage the decode at
nearly the same rate; the supervised and forcing arms change slightly
more than probe and random at each magnitude. No arm buys naturalness at
matched magnitude. The first run (dropout encoder, step 4,383,100) read
the same to within 0.06 everywhere (dm 0.352 -> 0.151, forcing 0.316 ->
0.146, probe 0.389 -> 0.196, random 0.369 -> 0.201), the largest change
being forcing at 0.25.

The blinded rating sheet (`rating.csv`, 120 items) is regenerated with
the run and still unscored.

Outputs: `out/resid_nc_hm_4380000/` (`naturalness.csv`, `rating.csv`,
`key.csv`, `summary.json`); `summary.json` is copied to
`writeup/figures/data/naturalness-summary.json` for the figure.

```bash
uv run python experiments/task-naturalness/naturalness.py \
    --steers experiments/steering-overlay/out/resid_nc_hm_4380000/steers.jsonl \
    --device cuda
```

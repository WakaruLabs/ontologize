# steering-overlay: results

## `resid_nc_hm`, language steering (2026-10-07)

Rerun after the SONAR encoder fix and the SONAR_NORM decode scale (see
`experiments/ste-arm/notes.md`, "The SONAR encoder ran with dropout").
The original run's checkpoint, step 4,383,100, is no longer kept
(checkpoints are retained every 10k steps), so this runs at step
4,380,000. Defaults otherwise: 8 languages, magnitudes 0.25 / 0.5 / 1.0,
8 steered rows per language, 8 random directions.

Medians over each arm's 24 language--magnitude conditions; hit is the
shared-frame `hit_dm`:

| arm | effect (native) | effect (dm frame) | hit (dm frame) | collateral |
|---|---|---|---|---|
| difference of means | 0.474 | 0.474 | 0.062 | 0.752 |
| classification forcing | 0.399 | 0.302 | 0.000 | 0.776 |
| linear probe | 0.294 | 0.072 | 0.000 | 0.701 |
| random direction | -- | -- | -- | 0.705 |

On the shared difference-of-means frame, forcing a classification moves
decodes 0.64x as far as supervised steering (0.302 / 0.474) and 4.2x as
far as the probe direction (0.302 / 0.072). Every arm's collateral sits
at the random-direction floor (0.70--0.78).

The first run, with the encoder's dropout on and decodes rescaled to the
3-sentence reference norm (about 0.25), read dm 0.337 / 0.337 / 0.250 /
0.757, forcing 0.394 / 0.146 / 0.125 / 0.778, probe 0.293 / 0.028 /
0.000 / 0.720 and random 0.721: forcing at 0.43x of supervised steering
and 5x the probe. The re-encoding that every dm-frame effect is read off
carried dropout noise there, which is the likeliest reason the dm-frame
effects were lower.

Outputs: `out/resid_nc_hm_4380000/` (`steer_overlay.csv`, `steers.jsonl`,
`meta.json`).

```bash
uv run python experiments/steering-overlay/steer_overlay.py \
    --model data/out/sonar/multilingual/resid_nc_hm --step 4380000 --device cuda
```

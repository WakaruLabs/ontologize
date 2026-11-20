# Notes: forcing one entry

## 2026-11-20: the Burmese entry does not reproduce, and no entry writes a language alone

**Short version.** The claim was that forcing entry 17 of head 8 in layer 1 of `resid_nc`
turns decoded text into Burmese while keeping its meaning. Over held-out rows of other languages,
with the head's 31 other entries as the null, it does neither. On the recipe ladder's control arm,
trained on correctly tagged sentences, a scan of all 12,160 entries finds layer-0 entries that
Burmese inputs choose, but forcing any one of them moves an output at most a sixth of the way to
Burmese, and none yields a Burmese decode.

### `resid_nc` (step 372600, the only checkpoint the claim could have used)

The claim came from `decode.py`. It used temperature 0.1, English forced as the output language,
beam 4 and each input sentence's own norm. These runs use the model's final training temperature,
0.03, unless noted, and `textfid.SonarDecoder` (greedy, rescaled to SONAR's norm).

**The decoder has to be free to choose the language.** With English forced it translates. No
Burmese row decodes with a Myanmar letter: 0 of 64 tail rows of `mc4_4M`, which were encoded under the English
tag, and 0 of 64 sentences of `mc4_sent16`, which were encoded under `mya_Mymr`. With no forced first token
(`--free-lang`), Burmese rows come out mostly in Myanmar script:

| rows (`controls.json`) | decoder | mostly Myanmar | first token `mya_Mymr` |
|---|---|---|---|
| `mc4_4M` Burmese tail, input | free | 17.2% | 21.9% |
| `mc4_4M` Burmese tail, `resid_nc`'s reconstruction | free | 20.3% | 21.9% |
| `mc4_sent16` Burmese tail, input | free | 15.6% | 29.7% |
| `mc4_sent16` Burmese tail, `resid_nc`'s reconstruction (`--unit`) | free | 20.3% | 29.7% |
| `mc4_sent16` Burmese tail, the ladder arm's reconstruction (below) | free | 4.7% | 12.5% |
| either cache, input or reconstruction | English | 0 | 0 |

**Free decoder, 128 non-Burmese `mc4_4M` tail rows** (`resid_nc_372600_L1H8_free`):

| condition | mostly Myanmar | first token `mya_Mymr` | re-encoded cosine to the row |
|---|---|---|---|
| entry 17 | 0 / 128 | 1 / 128 | 0.328 (19th of 32) |
| its 31 siblings | median 0, max 3.1% (entry 13) | median 0.8%, max 4.7% | median 0.331, max 0.346 |
| unforced reconstruction | 0 | 0.8% | 0.342 |
| input | 0 | 0.8% | 0.339 |

An unrelated row reads 0.092.

**English decoder, 256 rows** (`resid_nc_372600_L1H8`). None of the 8,704 decodes (34 conditions)
has a Myanmar letter. Entry 17's cosine is 0.372, 20th of 32 (median 0.375, max 0.389). Unforced it
is 0.381, and an unrelated row reads 0.089. Entry 17's decodes are mild paraphrases of the
reconstruction, as its siblings' are.

**English sentences at the session's temperature** (`--cache mc4_sent16 --lang en --unit
--temperature 0.1`, 128 rows, English decoder; `resid_nc_372600_L1H8_sent16_en_t0.1`). Again no
Myanmar letter in 4,352 decodes. Forcing any entry of the head replaces most of a sentence's meaning:
- the median cosine is 0.303, against 0.914 for the input's own decode and 0.804 unforced;
- an unrelated sentence reads 0.066;
- entry 17 reads 0.294, 21st of 32;
- the best sibling keeps 0.537.

**Burmese inputs do not choose the entry.** Of `mc4_4M`'s 381 Burmese tail rows, 4.7% choose it, against 5.4% of other
rows and 3.1% uniform. Of `mc4_sent16`'s 388 Burmese tail sentences, 2.3% choose it, against 4.9%.

`resid_nc` was trained on `mc4_4M`, in which Burmese (like 79 other configurations) was encoded
under the English tag, so it never saw a Burmese-tagged embedding.

### The post-fix scan: `ste_h76_init01_med837_ctrl` (step 50000, t = 0.00015, raw `mc4_sent16` rows)

`scan.py`, `scan_ste_h76_init01_med837_ctrl_50000_my/`.

**The probe.** A class-balanced logistic probe for Burmese against the rest, fitted on the 200,000
rows before the tail (2,497 Burmese), reads the tail at AUROC 0.9999 on inputs and 0.9958 on the
model's reconstructions. It calls 90.5% of Burmese reconstructions Burmese and 0.6% of the rest.
The mean logits are +4.73 and −7.05, 11.8 apart.

**Input side.** Layer 0 has entries that Burmese rows choose. In five heads one entry takes 37–52%
of Burmese tail rows, 15–20× its rate among other languages (log₂ ratio 3.9–4.3): entry 7 of head 53, 6
of 64, 5 of 35, 25 of 38 and 5 of 13.

**Output side.** Each entry is forced over 256 non-Burmese tail rows, split into halves A and B:
- **Reproducible.** The shifts in probe logit correlate at 0.96 between the halves over all
  12,160 entries.
- **Selection held out.** The top 8 by A move B by +1.16 logits, against +0.011 ± 0.109 for random
  entries.
- **Largest shift:** +1.87 (entry 6 of head 64 in layer 0), a sixth of the 11.8-logit gap. Four
  entries exceed +1, and 44 exceed +0.5.
- **Rows tipped.** The probe calls at most 4.7% of an entry's forced rows (1.6% on B) Burmese, and
  0.07% on average.
- **By layer.** Mean shifts are 0.035, 0.020, 0.003, 0.002 and −0.001 for layers 0–4.
- **Link to the input side.** In layer 0 an entry's shift correlates with how strongly Burmese
  inputs choose it (r = 0.52; 0.29 over all layers). Six of the top 8 are among the entries Burmese
  rows choose.

**Decodes** (free decoder, B's 128 rows each). The positive controls read 15.6% (inputs) and 4.7%
(this model's reconstructions) mostly Myanmar, and 29.7% / 12.5% starting with `mya_Mymr`. Every
forced entry decoded reads 0 on both:
- the 8 largest shifts;
- the 8 entries Burmese rows choose most;
- 8 random entries.

The unforced reconstruction reads 0 too.

**Reading.** On both models a language is written by many heads together. Even on the correctly
tagged model, where layer 0 sorts Burmese inputs into a handful of entries, one entry carries only a
small part of what separates a Burmese output from the rest. The decode instrument is weak on the
hard stack, whose reconstructions of Burmese rows decode as Burmese only 4.7% of the time, so the
probe is the measure there.

```bash
JAX_PLATFORMS=cpu uv run python experiments/entry-force/force.py \
    --model data/out/sonar/multilingual/resid_nc --layer 1 --head 8 --entry 17 \
    --device cuda --free-lang --rows 128 --out data/out/sonar/entry_force/resid_nc_372600_L1H8_free
uv run python experiments/entry-force/scan.py \
    --model data/out/sonar/recipe_ladder/ste_h76_init01_med837_ctrl \
    --cache data/sonar_embeddings/mc4_sent16.npy --lang my --device cuda
```

# pair-diff: notes

## 2026-11-11: the data, and what sentence models give on it

**Data.** `diffs.py` with its defaults. The weights are fitted on 262,144
training pairs from the cache's first 4M rows; coordinate 837 is masked
as in the sentence weights. The median |x_a − x_b| is 0.262, against a
median sentence norm of 0.21. The held-out tail holds 32,375 pairs over
86 languages (`data/out/sonar/pair-diff/tail.npy`).

**Reference: the sentence models.** `evaldiff.py --step 15000` with the
pair objective's `s_pair = 0` controls (`doc2_sp0_ph38`, seeds 42 and 43)
in both roles, which gives what a code trained on sentences does on the
same rows before any difference model exists.

| | seed 42 | seed 43 |
|---|---|---|
| FVU_w of the sentence (x_a) | 0.5200 | 0.5196 |
| FVU of the difference from x̂_a − x̂_b (two codes, 760 heads) | 0.5832 | 0.5830 |
| FVU of the difference, fed to the model as a row | 0.8066 | 0.7945 |
| head NMI with language on sentences, mean / max | 0.046 / 0.056 | 0.046 / 0.054 |
| head NMI with language on differences, mean / max | 0.013 / 0.015 | 0.014 / 0.015 |

Seed match of the two sentence models, 380 heads:

| read on | matched NMI | rows shuffled | any pairing | within language (shuffled within) |
|---|---|---|---|---|
| sentences | 0.0194 | 0.0049 | 0.0173 | 0.321 (0.297) |
| differences | 0.0107 | 0.0055 | 0.0094 | 0.277 (0.266) |

**Readings.**
- **The bar for a difference model.** Two sentence codes already
  reconstruct the difference at FVU 0.583, so a 380-head difference model
  has that to beat at half the heads. Its seed match has 0.019 matched
  against 0.017 for any pairing to beat: sentence heads agree across seeds
  a little, and almost as well under any pairing as under the best.
- **Differences carry little language.** A sentence model's heads tell a
  third as much about the document's language from a difference as from
  a sentence (0.013 against 0.046), so most of what they read of language
  cancels between two sentences of a document.
- **Within-language NMI is dominated by its bias** at about 376 rows a
  language (0.30 where the shuffle gives 0.28). Only its margin over the
  shuffle means anything: 0.024 on sentences.

**Not yet run:** the difference models themselves (`diff_h380_l1_s42`,
`_s43`), which are queued.

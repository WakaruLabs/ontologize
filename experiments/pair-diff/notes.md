# pair-diff: notes

## 2026-11-11: difference codes reproduce across seeds no better than sentence codes

**Data.** `diffs.py` with its defaults. The weights are fitted on 262,144
training pairs from the cache's first 4M rows; coordinate 837 is masked
as in the sentence weights. The median |x_a − x_b| is 0.262, against a
median sentence norm of 0.21. The held-out tail holds 32,375 pairs over
86 languages (`data/out/sonar/pair-diff/tail.npy`).

**Runs.** `train_diff.py --epochs 1 --keep-every 2500`, seeds 42 and 43
(`data/out/sonar/pair-diff/diff_h380_l1_s{42,43}`): one layer of 380
heads at k 32, 15,420 steps of 256 differences. The sentence models
beside them are the pair objective's `s_pair = 0` controls
(`doc2_sp0_ph38`, seeds 42 and 43), the same shape, trained on the same
pairs' sentences. Everything is `evaldiff.py --step 15000`
(`eval/evaldiff_15000.json`; `eval/sentence_on_diffs_15000.json` is the
sentence models in the difference models' place).

**Seed match, 380 heads:**

| models, read on | matched NMI | rows shuffled | any pairing | best over runner-up | within language (shuffled within) |
|---|---|---|---|---|---|
| difference, on differences | 0.0119 | 0.0049 | 0.0105 | 0.0002 | 0.336 (0.326) |
| sentence, on sentences | 0.0194 | 0.0049 | 0.0173 | 0.0003 | 0.321 (0.297) |
| sentence, on differences | 0.0107 | 0.0055 | 0.0094 | 0.0002 | 0.277 (0.266) |

**Per model** (seed 42 / seed 43):

| | difference | sentence | sentence, on differences |
|---|---|---|---|
| FVU_w of its own input | 0.4523 / 0.4530 | 0.5200 / 0.5196 | 0.8066 / 0.7945 |
| FVU of the difference from x̂_a − x̂_b (760 heads) | | 0.5832 / 0.5830 | |
| head NMI with language, mean | 0.0129 | 0.0460 | 0.0134 / 0.0135 |
| the same with labels shuffled | 0.0105 | 0.0105 | 0.0112 |
| head entropy, bits | 5.00 | 5.00 | 4.51 / 4.50 |

**The negated difference** (difference models, both seeds alike). A head
gives x_b − x_a the entry it gave x_a − x_b on 3.4% of rows, where its
entry usage gives 3.1% by chance. The NMI between the two codes is 0.017
per head, against 0.004 with rows shuffled.

**Readings.**
- **The hypothesis fails at this scale.** Two seeds' difference codes
  match at 0.0070 above the shuffled-row null, where two seeds' sentence
  codes match at 0.0145. The margin of the best match over any pairing is
  0.0014 against 0.0021, and the within-language margin 0.010 against
  0.024. Training on what varies within a document does not make heads
  reproduce; it removes about half of the little agreement sentence codes
  have.
- **A reading of why, untested.** Joshi et al.'s result needs each pair to
  differ in few concepts. Two sentences of a document two or more
  segments apart differ in most of what a sentence says, so the shift is
  dense and their assumption does not hold. The agreement that is lost
  could be the document-level part that differences cancel; nothing here
  separates that from other causes.
- **Differences cancel language.** Above the shuffled-label bias, a
  difference model's head tells 0.0024 of the document's language and a
  sentence model's head 0.0355: a fifteenth. A sentence model reading
  differences gives the same 0.002, so this is the data, not the
  training.
- **The difference model is the better code of a difference.** It
  reconstructs x_a − x_b at FVU 0.452 with 380 heads, where the two
  sentence codes' reconstructions give 0.583 with 760. A difference is
  easier to code than a sentence at this budget (0.452 against 0.520 for
  a sentence), under each one's own whitening.
- **Heads read a signed direction, without opposite entries.** Negating
  the input changes the entry about as often as chance, so heads do not
  read an axis or a size alone. The code of −x is nearly unrelated to the
  code of x (NMI 0.017), so entries do not come in opposite pairs either:
  the model, which saw each pair in one order, has not learned the
  symmetry.
- **Within-language NMI is dominated by its bias** at about 376 rows a
  language (0.28 to 0.34 where the shuffle gives 0.27 to 0.33). Only its margin
  over the shuffle is read.

**Caveats.** One epoch, two seeds, one pair distance (at least two
segments). The seed match is NMI between hard codes; the weaker kinds of
agreement in `experiments/seed-probes` were not run here.

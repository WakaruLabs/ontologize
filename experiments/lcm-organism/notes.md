# lcm-organism: notes

## 2026-11-23, later: templates with word slots don't carry the lead either, against a random-deletion control

**In short.**
- **Two more flags** catch templates whose slots vary in words, which the entry below misses:
  - *skeleton:* a sentence with its names, numbers and URLs masked repeats another document's;
  - *shingle:* its masked word 3-grams overlap another tail document's row at Jaccard 0.5 or more.

  With them, 9.7% of the tail is flagged, against 6.0%.
- **Dropping every flagged row leaves the lead as it was.** The LCM's within-language excess over
  SONAR's is 0.0597, against 0.0589 on all rows.
- **The control** drops as many rows of each language at random, and the lead there is 0.0575.
  The targeted drop's difference from it (+0.0022) is about the size of the random drop's own
  (−0.0014), so neither moves the lead.
- **SONAR's agreement leans on the templated rows more than the LCM's does.** Dropping them takes
  SONAR's excess to 0.0157, against 0.0176 for the random drop, while the LCM's barely differs
  (0.0754 against 0.0751).

**The new flags** (`templates.py`, 2 min 28 s, most of it the pass over the sentence texts).
- **The mask** (`masked_tokens`). Words are casefolded. A URL, a word with a digit, and a name are
  masked, and a run of one mask counts as one. A name is a word after the first that starts with
  a capital, or a word with a capital after its first letter.
- **Skeleton.** The masked sentence is also a sentence of another document anywhere in mc4_sent16.
  It counts only when at least two words are left unmasked and they are at least half its tokens,
  so two sentences don't match on their masks alone.
  - It flags 7.0% of the rows, 3.3 points of them beyond the text and twin flags.
- **Shingle.** For a sentence of at least four tokens whose skeleton counts (83% of the rows), its
  masked word 3-grams have a Jaccard index of at least 0.5 with a row of another document in the
  tail.
  - MinHash signatures (60 hashes in 20 bands of 3) propose the pairs, and each is checked
    exactly.
  - It flags 2.6% of the rows, 1.6 points of them beyond the text and twin flags.
- **All four together** flag 9.7% of the rows overall and in the scored languages alike.
  - By kind: 16.4% of whole lines, 10.0% of cap-made pieces, 7.2% of sentences.
  - By language: from 1.5% (Malayalam) and 1.9% (English) to 62% (Cebuano, whose mC4 pages are
    largely bot-written), 36% (Maltese) and 30% (Scottish Gaelic).
- **The converter family** is now caught almost whole:
  - all 29 "fachtóir tiontaithe" sentences;
  - 94% of the 85 rows with "airgeadra";
  - all six "Aisghafa ó" footers;
  - 96% of the 16 converter pages' 168 tail rows.
- **What else they catch.** A sample of the rows only the new flags catch reads mostly as
  boilerplate: listing titles ("Apartament Bence 32 km"), directions to an address, credit ads,
  timestamps, GeoNames bot text, menus. A few are generic patterns, such as "Шта је Андроид
  Рецовери?" ("What is Android Recovery?") matching another "What is X?". German capitalizes its
  nouns, so there the mask reads them as names and the skeleton reaches further.

**The control** (`keep_random.npy`) drops as many rows of each language as the four flags do,
25,500 in all, chosen at random (seed 0). The two masks keep the same number of rows in every
language, 236,644 of the last 262,144 in all, so their within-language scores weigh the languages
alike. Among the
matching's last 32,768 rows they keep 29,545 and 29,612.

**The rescoring**, as in the entry below:

| Rows scored | Within language: LCM excess | SONAR excess | LCM's lead | Ratio | Readout (x_n and context) gap closed | Readout (x_n alone) |
|---|---|---|---|---|---|---|
| All | 0.0774 | 0.0186 | 0.0589 | 4.2 | 62% | 38% |
| Text and twin flags dropped (6.0%) | 0.0758 | 0.0163 | 0.0595 | 4.7 | 62% | 37% |
| **All four flags dropped (9.7%)** | **0.0754** | **0.0157** | **0.0597** | **4.8** | **63%** | **37%** |
| Random control (as many per language) | 0.0751 | 0.0176 | 0.0575 | 4.3 | 62% | 38% |
| Flagged rows alone | 0.1306 | 0.0649 | 0.0657 | 2.0 | | |

- **On all heads** the lead is 0.0189 with the flagged rows dropped and 0.0174 under the control
  (0.0172 on all rows). The readouts close 84% and 79% of the gap without the flagged rows, and 85%
  and 83% under the control (86% and 84% on all rows).
- **The flagged rows alone** are scored on 22,307 rows in 60 languages at 200 or more rows each,
  and on all heads on their 3,223 rows of the last 32,768.

**What it means.**
- **The lead sits in the ordinary rows,** now with templates that vary in words removed as well as
  verbatim repeats, and against a deletion of the same size per language.
- **Templated rows help SONAR's seeds agree more than the LCM's.** Removing them costs SONAR more
  than a random deletion does, and costs the LCM no more. On them alone the LCM leads by 2.0 times
  SONAR's excess, against 4.2 times on all rows.

**Caveats.**
- **One seed pair per arm.** The readout from x_n alone closing 79% of the all-heads gap without
  the flagged rows, against 83% under the control, rests on it.
- **The mask is rough.** It reads a capital as a name, which overreaches in German and misses names
  in scripts without capitals. The shingle flag compares tail rows only with other tail rows.
- **The control is one draw** of the random deletion.

```bash
uv run python experiments/lcm-organism/templates.py
# as in the entry below, with --row-mask $T/keep_none.npy and $T/keep_random.npy for the four arms,
# and $T/only_any.npy at --min-rows 200 for the LCM and SONAR
```

## 2026-11-23: dropping verbatim repeats leaves the LCM's within-language lead as it was

**In short.**
- **The question.** mC4 has many templated pages: footers, timestamps, shop carts, spam. Do the
  templated sentences in the scored rows carry the LCM's lead over SONAR within a language?
- **Dropping repeated text leaves the lead as it was.** 6.0% of the held-out tail's rows repeat
  text from another document, or have a near-twin in one. Without them, the LCM's within-language
  excess over SONAR's is 0.0595, against 0.0589 on all rows. The readouts close the same share of
  the gap as before.
- **On the flagged rows alone, both models agree much more** (excess 0.178 for the LCM and 0.105
  for SONAR). The LCM still leads there, by 1.7 times SONAR's excess, against 4.2 times on all rows.
- **This rules out verbatim repeats, not templates in general.** Templates whose slots vary in
  words largely survive the cut: about half of the Irish currency-converter sentences, which differ
  only in the currency's name, are flagged, and none by the twin flag.

**The flags** (`templates.py`, on the last 262,144 rows, which are the same sentences in every arm's
cache).
- **Text.** A row's sentence, casefolded with every digit read as 0 and whitespace collapsed, is also
  a sentence of another document anywhere in mc4_sent16. This flags 5.9% of the rows.
- **Twin.** A row's SONAR embedding, centered on the tail's mean, has cosine 0.95 or more with a row
  of another document in the tail. This flags 0.7% of the rows, and nine in ten of them repeat text
  too. The share flagged at a threshold of 0.9 is 0.9%, and at 0.99 it is 0.6%.
- **Either** flags 6.0% of all rows, and 5.9% of the rows in the 84 scored languages.
- **What they are.** A sample of flagged rows reads as boilerplate: "Vytvořeno 27. 2. 2017 19:01",
  "Winkelmand 0 producten €0.00", "コメントの読み込み中に問題が発生しました。", lists of spam URLs.
  Some unflagged rows are boilerplate too, so the flags are a lower bound.
- **Where they are.**
  - By language: from 1% (English, Tamil) to 32% (Maltese), with 24% in Scottish Gaelic and 20% in
    Cebuano.
  - By kind: whole lines are flagged most (8.6% repeat text), then cap-made pieces (6.6%), then
    sentences (4.7%).

**The rescoring.**
- **The arms.** `seedmatch.py --row-mask` drops the flagged rows from every score: the matching's
  last 32,768 rows keep 30,830, and the within-language score's last 262,144 keep 246,534. Four arms
  were rescored: the LCM's TPU pair, SONAR's GPU pair, and the two readouts.
- **The SONAR pair.** Unmasked, SONAR's GPU pair reads 0.0168 (0.0048) on all heads and 0.0692
  (0.0507) within a language. The TPU pair, which the gap is usually computed from, reads 0.0169 and
  0.0692; it isn't on this machine.

| Rows | Within language: LCM excess | SONAR excess | LCM's lead | Ratio | Readout (x_n and context) gap closed | Readout (x_n alone) |
|---|---|---|---|---|---|---|
| All | 0.0774 | 0.0186 | 0.0589 | 4.2 | 62% | 38% |
| **Flagged rows dropped** | **0.0758** | **0.0163** | **0.0595** | **4.7** | **62%** | **37%** |
| Flagged rows alone | 0.1776 | 0.1049 | 0.0727 | 1.7 | | |

- **On all heads.** The lead is 0.0172 on all rows, 0.0183 with the flagged rows dropped, and 0.0338
  on the flagged rows alone. The readouts close 86% and 84% of the gap on all rows, and 85% and 81%
  with the flagged rows dropped.
- **Scored rows and languages:**
  - with the flagged rows dropped, 245,025 rows in the same 84 languages;
  - on the flagged rows alone, 7,299 rows in 23 languages. That set needs at least 200 rows a
    language (`--min-rows 200`), so it is read beside all rows at 200, where the LCM's excess is
    0.0776 and SONAR's 0.0186, as at 1,000. Its all-heads score rests on the 1,938 flagged rows of
    the last 32,768.

**What it means.**
- **The lead isn't in the verbatim repeats.** Without them, both arms' within-language excess falls
  a little, SONAR's more (0.0186 to 0.0163) than the LCM's (0.0774 to 0.0758), so the lead grows
  slightly.
- **Repeated rows are easy to agree on.** Both models' seeds agree far more on them, and the LCM's
  margin there is smaller relative to SONAR's. They are 6% of the rows, so they add to both arms'
  scores and do little to the difference.
- **The most templated languages count little in this score.** It weighs each language by its rows,
  and the three most flagged (Maltese, Scottish Gaelic, Cebuano) hold 0.9–1.4% of the scored rows
  each.

**Caveats.**
- **The flags are a lower bound, and a loose one for templates with word slots.** One template
  family, checked by its text in the tail's Irish rows:
  - sentences with "fachtóir tiontaithe" (a converter page's conversion factor): 14 of 29 flagged,
    all by repeated text;
  - sentences with "airgeadra" (currency): 13% of 85 flagged;
  - Wikipedia's "Aisghafa ó" (retrieved from) footers, whose URL differs: 1 of 6 flagged.

  Their centered SONAR cosines to their nearest other-document row have medians of 0.67–0.79, far
  below the twin threshold. Of the 16 converter pages' 168 tail rows, 30% are flagged. (Later the
  same day, the skeleton and shingle flags in the entry above catch 96% of them, and the lead holds
  without them too.)
- **Few rows on the templated side.** The templated-only scores rest on 7,299 rows in 23 languages
  at 200 or more rows each, where NMI over 32 entries is biased upward (their nulls are 0.24–0.27).
  Read that row by its difference between the arms, not by its level.
- **One seed pair per arm,** as before.

```bash
uv run python experiments/lcm-organism/templates.py
A=data/activations/lcm-organism
T=data/out/lcm-organism/templates
TPU=data/out/lcm-organism/phase2_tpu
# each arm scored as phase2.sh scores it, with the flagged rows dropped
JAX_DEFAULT_MATMUL_PRECISION=highest uv run python experiments/next-sentence/seedmatch.py \
    --a $TPU/lcm_b4_doc2_sp0_ph38_s42 --b $TPU/lcm_b4_doc2_sp0_ph38_s43 \
    --cache $A/lcm_b4.npy --labels $A/lcm_b4.langs.npy --label-rows 262144 --min-rows 1000 \
    --row-mask $T/keep_neither.npy --out $T/seedmatch/lcm_keep_neither_min1000.json
# likewise SONAR (data/out/sonar/next-sentence/long/doc2_sp0_ph38 and _s43 on
# data/sonar_embeddings/mc4_sent16.npy) and the two readouts (data/out/lcm-organism/phase2,
# lcm_readout_b4 and lcm_readout_x_b4); then the LCM and SONAR with --row-mask $T/only_either.npy
# and with no mask, both at --min-rows 200
```

## 2026-11-22: 19 of block 4's coordinates kept on their axes, the rest rotated: keeping them undoes most of what mixing costs, and keeping 19 others undoes none of it

**In short.**
- **Three checks of what the readout's axes carry that a rotation removes:** each coordinate's marginal,
  the dependence between pairs of coordinates' magnitudes, and the tail of their correlations. Only the
  last tells the axes from random directions. It finds a group of 19 strongly correlated, heavy-tailed
  coordinates, inherited from block 4.
- **Keeping those 19 on their axes and rotating the other 1,005 among themselves** closes 69% of
  SONAR's gap to the LCM on all heads and 34% within a language. The rescaled readout closes 72% and
  35%, and the rotated one 41% and 26%. Unrounded, the partial rotation undoes 27 of the 30 points that
  mixing costs on all heads, and 8 of 9 within a language: about 90%.
- **It has to be these 19.** Keeping 19 coordinates drawn at random from outside the group, with the
  same rotation of the rest, closes 40% and 25%, as the full rotation does.

**The checks** (`axes.py`, output in `data/out/lcm-organism/axes`). Each arm is read on the same
200,000 training rows (sample seed 0, from the first 4,400,014), in the frame its loss weighs. Each
is compared with what a rotation would give: 1,024 random directions in the same frame, or five
random rotations of the same covariance.
- **Marginals.** On the readout from x_n alone, a typical coordinate has the shape of a random
  direction: median excess kurtosis 0.32 on the axes against 0.33, median |skew| 0.09 against 0.10.
  - **The same holds in every arm,** the LCM's heavy tails included (1.26 against 1.25).
  - **Eight of the readout's axes fall outside the random directions' range,** with kurtosis up to
    4.9 and skew up to 1.5. Among them are coordinates 532 and 709, block 4's most heavy-tailed. The
    rotation removes all eight, and the rescaling keeps them.
  - **Such axes don't track agreement across the arms.** The fixed mix has 8 and closes 17%, SONAR 6
    and closes 0%, and the LCM 3 and closes 100%.
  - So the candidate in the entry below, the shape of each coordinate's own distribution, is mostly
    ruled out.
- **Magnitude dependence.** Per pair of coordinates, the Spearman correlation of their absolute values
  less a Gaussian's with the same covariance (about 524,000 pairs per arm; noise about 0.003 a pair).
  - **On the readout, axes and random directions match from the mean to the maximum:** 0.029 against
    0.030 on average, 0.065 for both at the 99.9th percentile, and 0.092 against 0.088 at the most. One
    random scale per row would give this in any basis.
  - **Block 4 has a little such structure on its own axes,** 64 pairs above 0.1 against 2 along
    random directions. The readout doesn't inherit it.
  - **The fixed mix has the most on its axes** (mean 0.058 against 0.052) and agrees poorly.
- **The correlations' tail.** The readout's correlation matrix on its axes, against the correlations
  of five random rotations of its covariance, which keep the spectrum:

  | Readout, x_n alone | Axes | Rotated, mean [range] |
  |---|---|---|
  | Largest \|ρ\| | 0.93 | 0.48 [0.43, 0.56] |
  | Pairs above 0.5 | 44 | 0.4 [0, 2] |
  | Coordinates with a partner above 0.5 | 19 | 0.6 [0, 3] |
  | Pairs above 0.3 | 1,730 | 756 |
  | Participation (1/IPR) of the top principal direction | 304 | 395 |
  | Median \|ρ\| | 0.063 | 0.063 |

  - **The typical pair is unchanged, and the difference is all in the tail.** The rotated arm's own
    cache reads as the rotations do: no pair above 0.5, the largest 0.44.
  - **The group is block 4's.** It holds 13 of block 4's 14 coordinates with a partner above 0.5, and
    26 of its 29 such pairs. Coordinates 532 and 709 correlate at −0.93, against −0.95 in block 4, and
    369 and 611 at −0.89 in both. Seven of the eight outlying axes above are in it.
  - **A group like it is not enough.** SONAR has one (10 coordinates, the largest |ρ| 0.74, the top
    principal direction on 185 coordinates against 364 rotated) and closes 0%. The fixed mix has one
    of 12 and closes 17%. Recolouring removes both groups, and agreement doesn't consistently fall:
    SONAR goes from 0% to 22%, and the fixed mix from 17% to −3%.

**The arm.**
- **The rows.** `weightframe.py --rotate-except` writes the rescaled readout with the 19 coordinates
  that have a partner above |ρ| 0.5 left on their axes:
  `33,34,151,188,209,317,364,369,376,386,532,611,657,706,709,710,796,852,945`. The other 1,005 live
  coordinates are mixed by one random rotation (seed 0, drawn as `recolour.py` draws its rotation). The
  scale is the rescaled arm's, c = 376.83 (`--scale-like` on the rotated cache).
- **The set.** A second, disjoint sample of rows gives the same 19. The threshold decides it: 33
  coordinates have a partner above 0.45, and 12 above 0.55.
- **The written cache.** The 19 columns are the rescaled cache's exactly, and row norms agree within
  4e-5. 43 of the group's 44 pairs above 0.5 remain. Seven new pairs at 0.50–0.52 join three of the
  group's hubs (317, 369, 611) to rotated coordinates. The hubs share much of their variance with the
  top principal directions, so mixtures of the rotated coordinates still correlate with them.
- **Its loss.** Mixing changes each mixed coordinate's variance, so the arm's own weights span a factor
  of 2.37, against 2.46 for the rotated arm. Its loss is nearly the rotated arm's (copy baseline
  1.0621, against 1.0633) and not the readout's. The 19 carry 1.8% of the weight, their share of the
  coordinates.

**The control** (`lcm_readout_x_rand19`) is this arm with 19 other coordinates kept:
`30,61,277,315,320,330,359,385,426,438,450,550,591,731,797,868,888,933,1013`.
- **The draw.** The 19 were drawn at seed 19 from the 991 live coordinates with no partner above
  |ρ| 0.45, which leaves out the group at every threshold above. Each one's strongest partner has a
  median |ρ| of 0.29, close to the pool's 0.30, against the group's 0.64. Among themselves they
  correlate at a median |ρ| of 0.06, and at most 0.20.
- **Otherwise the same:** the same rotation of the other 1,005 coordinates (seed 0, the same size),
  and the same scale.
- **The written cache,** on 20,000 rows:
  - the 19 columns are the rescaled cache's exactly;
  - row norms agree within 1.2e-4;
  - a linear map from the rescaled rows reproduces the mixed block with a share of 8e-8 of its
    variance left over;
  - after mixing, no coordinate has a partner above |ρ| 0.5.
- **Its loss.** Its own weights span a factor of 2.80 (copy baseline 1.0631). The 19 again carry 1.8%
  of the weight.

**Runs.** `ARMS=lcm_readout_x_keep19 phase2.sh train`, then `score`, on the RTX 3090, as for the other
readout arms: seeds 42 and 43 to step 46,250, about 2 h and 2 h 20 min, and 4 h 26 min with
scoring. The control ran the same way after it, at about 2 h a seed, and was scored on 2026-11-23. The
stored spec computes in float32, and the GPU ran its matmuls at JAX's default precision, which is TF32
there. Scoring ran at `highest`.

**The readouts** from x_n alone:

| Arm | All 380 heads: matched NMI (null) | First 38 heads, within language (null) | Gap closed, all heads / within | Head-specific part | Its loss | Copy baseline | FVU_w, each seed |
|---|---|---|---|---|---|---|---|
| Unrotated | 0.0313 (0.0048) | 0.0907 (0.0495) | 84% / 38% | 0.0026 | the readout's | 1.056 | 0.2481 / 0.2475 |
| Rescaled | 0.0291 (0.0048) | 0.0891 (0.0500) | 72% / 35% | 0.0023 | the readout's | 1.056 | 0.2569 / 0.2559 |
| **Keep the group of 19, rotate the rest** | **0.0287** (0.0048) | **0.0886** (0.0501) | **69% / 34%** | **0.0024** | weights span 2.37 | 1.062 | **0.2584 / 0.2586** |
| **Keep 19 others, rotate the rest** | **0.0237** (0.0048) | **0.0837** (0.0506) | **40% / 25%** | **0.0020** | weights span 2.80 | 1.063 | **0.2676 / 0.2689** |
| Rotated | 0.0240 (0.0048) | 0.0842 (0.0506) | 41% / 26% | 0.0019 | weights span 2.46 | 1.063 | 0.2677 / 0.2670 |

Next-sentence FVU_w, each seed:
- keeping the group, 0.9660 / 0.9661, with ridge baseline 0.7186;
- keeping 19 others, 0.9702 / 0.9703, with ridge baseline 0.7193;
- for comparison, rescaled 0.9610 / 0.9606 and rotated 0.9698 / 0.9697.

**What it means.**
- **The mixing's cost sits mostly in a small group of block 4's coordinates.** They are 1.8% of the
  coordinates and of the loss weight. Keeping them undoes about 90% of what mixing costs, under a loss
  nearly as uneven as the rotated arm's.
- **Keeping a few axes unmixed is not what helps.** Nineteen other coordinates, kept the same way,
  undo none of it: 40% / 25%, against the full rotation's 41% / 26%.
- **Reconstruction moves with agreement.** FVU_w is 0.258 with the group kept, near the rescaled arm's
  0.257. With 19 others kept it is 0.268, as when rotated.
- **The three checks predicted this.** Only the correlations' tail found anything on the axes, and
  it was this group.
- **Why the trainer responds to them is untested.** Adam's per-coordinate step sizes make this recipe
  sensitive to the basis. A rotation-equivariant recipe would give the rescaled and rotated arms the
  same agreement, apart from the rotated arm's uneven weights, and keeping any coordinates would not
  matter. The group's coordinates are both strongly correlated and heavy-tailed, and these runs don't
  say which of the two the recipe uses.
- **The group matters as part of block 4's map, not on its own.** SONAR and the fixed mix have groups
  like it and agree poorly.

**Caveats.**
- **The group's own axes, or its being left unmixed?** Both arms leave the kept coordinates out of the
  mixing and on their axes. A rotation of the 19 among themselves, with the rest left as they are,
  would tell the two apart. It has not been run.
- **One draw of 19 others.** The control is one random draw, so it shows that 19 coordinates outside
  the group need not help, not that none would.
- **Seeds.** One scored pair, as for every control arm. On all heads:
  - keeping the group gives a matched NMI 0.0004 below the rescaled arm's and 0.0047 above the
    rotated arm's;
  - keeping 19 others gives 0.0003 below the rotated arm's.

  The LCM's own seed pairs spread by 0.0017 on the 38 slow heads.
- **The threshold.** The group is the coordinates with a partner above 0.5. How the result moves with
  the threshold (33 coordinates at 0.45, 12 at 0.55) is untested.
- **The checks' code.** The three checks and the control's draw first ran as one-off scripts.
  `axes.py` reproduces their numbers to rounding, and the control's 19 coordinates exactly.
- **Precision.** How much TF32 rounding adds to the recipe's dependence on the basis has not been
  measured.

```bash
A=data/activations/lcm-organism
# the three checks, the group, its disjoint-sample check and the control's draw
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/axes.py --disjoint --draw 19
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/weightframe.py --scale-like $A/lcm_readout_x_rot_b4.npy \
    --rotate-except 33,34,151,188,209,317,364,369,376,386,532,611,657,706,709,710,796,852,945 \
    --out $A/lcm_readout_x_keep19_b4.npy
ARMS=lcm_readout_x_keep19 experiments/lcm-organism/phase2.sh train
ARMS=lcm_readout_x_keep19 experiments/lcm-organism/phase2.sh score
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/weightframe.py --scale-like $A/lcm_readout_x_rot_b4.npy \
    --rotate-except 30,61,277,315,320,330,359,385,426,438,450,550,591,731,797,868,888,933,1013 \
    --out $A/lcm_readout_x_rand19_b4.npy
ARMS=lcm_readout_x_rand19 experiments/lcm-organism/phase2.sh train
ARMS=lcm_readout_x_rand19 experiments/lcm-organism/phase2.sh score
```

## 2026-11-22: the readout rescaled to its loss's frame, unrotated: the rotation's cost is mostly the mixing

**In short.**
- **Mixing the readout's coordinates costs most of its lead. Rescaling them costs little.** The
  rotation of the entry below changes the basis in two steps. Each coordinate is scaled by the root
  of its loss weight, and then the coordinates are mixed. Dictionaries on the rows after the first
  step alone close 72% of SONAR's gap to the LCM on all heads and 35% within a language. Unrotated,
  the readout closes 84% and 38%; rotated, 41% and 26%.
- **Of the 43 points the rotation costs on all heads, rescaling takes 12 and mixing 31.** Within a
  language the rotation costs 13 points: 4 from rescaling, which may be seed noise (Caveats), and 9
  from mixing.
- **The head-specific part** falls from 0.0026 to 0.0023 with rescaling, and to 0.0019 with the
  rotation.

**The control.**
- **The rows.** `weightframe.py` writes the readout from x_n alone as y' = c √w ⊙ y, unrotated, with
  c = 376.83 taken from the rotated cache (`--scale-like`). The rotated cache is these rows rotated
  (`test_the_weighted_frame_is_the_cache_recoloured_to_itself_unrotated`), so the two arms differ by
  the rotation alone.
- **The same loss as the readout's.** The arm's own weights are flat (they span a factor of
  1.0002), so its loss measures errors exactly as the readout's does. Its next-sentence baselines are
  the readout's to four places: copy 1.0557, ridge 0.7153. The rotated arm's own weights span a
  factor of 2.5, so its loss differs a little (copy 1.0633).
- **Fidelity.** On 20,000 rows, the rotated cache keeps these rows' norms within 1.3e-4 and their
  inner products within 1.1e-4. A linear map from these rows reproduces it with a share of 8e-8 of
  its variance left over. Written in float16, 24 values rounded to zero.
- **Its spectrum** on the tail, under its own weights, is the readout's: participation ratio 98,
  effective rank 212, 41 / 263 directions for 50% / 90%, 0.147 between languages, and
  within-language participation ratio 108.

**Runs.** `ARMS=lcm_readout_x_wf phase2.sh train`, then `score`, on the RTX 3090, as for the other
readout arms: seeds 42 and 43 to step 46,250, a little over 2 hours each.

**The readouts,** beside every other arm's seeds 42 and 43:

| Arm | All 380 heads: matched NMI (null) | First 38 heads, within language (null) | Gap closed, all heads / within | Head-specific part | Share between languages | Copy baseline | FVU_w, each seed |
|---|---|---|---|---|---|---|---|
| LCM, block 4 | 0.0340 (0.0048) | 0.1269 (0.0495) | | 0.0026 | 0.154 | 0.439 | 0.2820 / 0.2812 |
| Readout, x_n and context | 0.0316 (0.0048) | 0.1043 (0.0494) | 86% / 62% | 0.0026 | 0.153 | 0.454 | 0.2746 / 0.2727 |
| Readout, x_n alone | 0.0313 (0.0048) | 0.0907 (0.0495) | 84% / 38% | 0.0026 | 0.147 | 1.056 | 0.2481 / 0.2475 |
| **Readout, x_n alone, rescaled** | **0.0291** (0.0048) | **0.0891** (0.0500) | **72% / 35%** | **0.0023** | 0.147 | 1.056 | 0.2569 / 0.2559 |
| Readout, x_n alone, rotated | 0.0240 (0.0048) | 0.0842 (0.0506) | 41% / 26% | 0.0019 | 0.145 | 1.063 | 0.2677 / 0.2670 |
| Fixed mix | 0.0197 (0.0048) | 0.0812 (0.0508) | 17% / 20% | 0.0015 | 0.090 | 0.731 | 0.4364 / 0.4356 |
| Fixed mix recoloured | 0.0163 (0.0048) | 0.0810 (0.0510) | −3% / 19% | 0.0015 | 0.117 | 0.632 | 0.3484 / 0.3476 |
| SONAR recoloured | 0.0206 (0.0048) | 0.0751 (0.0508) | 22% / 10% | 0.0016 | 0.072 | 1.369 | 0.3241 / 0.3236 |
| SONAR, TPU | 0.0169 (0.0048) | 0.0692 (0.0507) | | 0.0013 | 0.043 | 1.533 | 0.4593 / 0.4585 |
| Untrained LCM | 0.0155 (0.0048) | 0.0749 (0.0509) | −8% / 9% | 0.0012 | 0.057 | 0.766 | 0.4370 / 0.4370 |

**What it means.**
- **The dictionaries use the readout's own coordinate axes.** The readout writes block 4's state, so
  its coordinates are block 4's. Mixing them at random costs about 70% of what the rotation costs
  (31 of 43 points on all heads, 9 of 13 within a language), and evening out their scales costs the
  rest. What the axes carry, these runs do not say. One candidate is the shape of each coordinate's
  own distribution, which a rotation blends toward a Gaussian and a rescaling keeps. (Later the same
  day: the marginals check in the entry above finds a typical coordinate on the axes shaped like a
  random direction, which mostly rules this out. The correlations' tail is what differs.)
- **Reconstruction follows the same order.** FVU_w is 0.248 unrotated, 0.257 rescaled and 0.268
  rotated. The rotated arm's figure is under weights that are not quite flat.

**Caveats.**
- **Seeds.** Two training seeds per arm, and one pair of them scored. The LCM's three seeds give
  three pairs, whose 38 slow heads match within 0.0017 of one another on all rows. Rescaling moves
  the within-language score of those heads by 0.0016, about that spread. The all-heads scores are
  means over 380 heads, and no arm has a second pair scored on them.

```bash
A=data/activations/lcm-organism
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/weightframe.py --scale-like $A/lcm_readout_x_rot_b4.npy
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/spectrum.py \
    --arm lcm_readout_x_wf_b4 $A/lcm_readout_x_wf_b4.npy $A/lcm_readout_x_wf_b4.mse_weights_matched.npy \
    --out data/out/lcm-organism/spectrum_readout_x_wf
ARMS=lcm_readout_x_wf experiments/lcm-organism/phase2.sh train
ARMS=lcm_readout_x_wf experiments/lcm-organism/phase2.sh score
```

## 2026-11-22: the readout from x_n alone under a random rotation: the coordinate basis carries half its lead on all heads

**In short.**
- **A random rotation halves the readout's lead on all heads.** Dictionaries on the readout from x_n
  alone, rotated at random with its spectrum kept, close 41% of SONAR's gap to the LCM on all heads,
  against 84% unrotated. Within a language they close 26%, against 38%.
- **The rest of the lead survives the rotation.** Rotated, the readout still beats SONAR recoloured to
  block 4's spectrum, the other randomly rotated linear map of SONAR's rows: 41% against 22% on all
  heads, and 26% against 10% within a language.
- **The head-specific part falls** from the LCM's level, 0.0026, to 0.0019. SONAR's is 0.0013.
- **The share of variance between languages does not carry the lead.** The rotation keeps the share
  (0.145 against 0.147) and loses half the lead. Over the nine arms, the correlation of agreement
  with that share falls to 0.83 on all heads and 0.78 within a language.

**The control.**
- **The map.** `recolour.py` with the readout as both source and target, under its matched weights.
  Recolouring a cache to its own spectrum is a rotation times a scale
  (`test_recolouring_to_its_own_spectrum_is_a_scaled_rotation`) of its rows in the frame its loss
  weighs: y_rot = c R Uᵀ (√w ⊙ y). So the loss measures errors nearly as it does on the readout
  (its own weights, next bullet, are not quite flat), and only the basis changes. ("Nearly" added
  2026-11-22, the same session.) Rotation seed 0, nothing zeroed, written into `lcm_readout_x_rot_b4.npy` with
  its own matched weights.
- **The weights move into the rows.** The trainer weights each coordinate by its inverse variance.
  The readout's weights span a factor of 19 (coefficient of variation 0.24). The rotated cache
  carries them in its rows, and its own weights span a factor of 2.5 (0.13).
- **Its spectrum** on the tail, under its own weights, is the readout's:

| Arm (the tail) | Participation ratio | Effective rank | Directions for 50% / 90% | Share between languages | Within-language participation ratio |
|---|---|---|---|---|---|
| Readout, x_n alone | 98 | 212 | 41 / 263 | 0.147 | 108 |
| Readout, x_n alone, rotated | 103 | 217 | 42 / 265 | 0.145 | 113 |

**Runs.** `ARMS=lcm_readout_x_rot phase2.sh train`, then `score`, on the RTX 3090, as for the other
readout arms: seeds 42 and 43 to step 46,250.

**The readouts,** beside every other arm's seeds 42 and 43:

| Arm | All 380 heads: matched NMI (null) | First 38 heads, within language (null) | Gap closed, all heads / within | Head-specific part | Share between languages | Copy baseline | FVU_w, each seed |
|---|---|---|---|---|---|---|---|
| LCM, block 4 | 0.0340 (0.0048) | 0.1269 (0.0495) | | 0.0026 | 0.154 | 0.439 | 0.2820 / 0.2812 |
| Readout, x_n and context | 0.0316 (0.0048) | 0.1043 (0.0494) | 86% / 62% | 0.0026 | 0.153 | 0.454 | 0.2746 / 0.2727 |
| Readout, x_n alone | 0.0313 (0.0048) | 0.0907 (0.0495) | 84% / 38% | 0.0026 | 0.147 | 1.056 | 0.2481 / 0.2475 |
| **Readout, x_n alone, rotated** | **0.0240** (0.0048) | **0.0842** (0.0506) | **41% / 26%** | **0.0019** | 0.145 | 1.063 | 0.2677 / 0.2670 |
| Fixed mix | 0.0197 (0.0048) | 0.0812 (0.0508) | 17% / 20% | 0.0015 | 0.090 | 0.731 | 0.4364 / 0.4356 |
| Fixed mix recoloured | 0.0163 (0.0048) | 0.0810 (0.0510) | −3% / 19% | 0.0015 | 0.117 | 0.632 | 0.3484 / 0.3476 |
| SONAR recoloured | 0.0206 (0.0048) | 0.0751 (0.0508) | 22% / 10% | 0.0016 | 0.072 | 1.369 | 0.3241 / 0.3236 |
| SONAR, TPU | 0.0169 (0.0048) | 0.0692 (0.0507) | | 0.0013 | 0.043 | 1.533 | 0.4593 / 0.4585 |
| Untrained LCM | 0.0155 (0.0048) | 0.0749 (0.0509) | −8% / 9% | 0.0012 | 0.057 | 0.766 | 0.4370 / 0.4370 |

**What it means.**
- **The basis is about half the readout's lead on all heads,** and about a third of it within a
  language.
- **What survives the rotation is more than the spectrum.** The rotated readout and the recoloured
  SONAR are both randomly rotated linear maps of SONAR's rows. They differ by 19 points on all heads
  and 16 within a language, in two ways:
  - **which directions get the variance;**
  - **the spectrum itself:** the recoloured SONAR has block 4's, and the readout its own
    lower-dimensional one (participation ratio 98 against 141).

  So this run does not isolate the directions.
- **The change of basis has two steps.** Each coordinate is scaled by √w_j, which evens out the
  input's scales (they span a factor of 4.4), and then the coordinates are mixed. The rows after the
  first step alone (`weightframe.py`) separate the two: they close 72% on all heads and 35% within a
  language, so the mixing costs most of the lead (2026-11-22, the entry above).

  (Corrected 2026-11-22, the same session: this bullet first said the rotation changed how the loss
  weighs the coordinates, and proposed training the unrotated readout with flat weights. The map
  carries the weights into the rows, so the loss is unchanged, and flat weights would have changed
  it. That arm was started and stopped after half an hour.)
- **Reconstruction is about as hard.** FVU_w is 0.268 against 0.248, each under its own weights. The
  copy baseline reads 1.063 against 1.056, and the next-sentence FVU_w 0.970 against 0.97 (ridge
  0.719).

**Caveats.**
- **Seeds.** Two training seeds, and one rotation seed.

```bash
A=data/activations/lcm-organism
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/recolour.py --source $A/lcm_readout_x_b4.npy \
    --source-weights $A/lcm_readout_x_b4.mse_weights_matched.npy --target $A/lcm_readout_x_b4.npy \
    --target-weights $A/lcm_readout_x_b4.mse_weights_matched.npy --out $A/lcm_readout_x_rot_b4.npy
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/spectrum.py \
    --arm lcm_readout_x_rot_b4 $A/lcm_readout_x_rot_b4.npy $A/lcm_readout_x_rot_b4.mse_weights_matched.npy \
    --out data/out/lcm-organism/spectrum_readout_x_rot
ARMS=lcm_readout_x_rot experiments/lcm-organism/phase2.sh train
ARMS=lcm_readout_x_rot experiments/lcm-organism/phase2.sh score
```

## 2026-11-21: the fixed mix recoloured to the LCM's spectrum agrees no more than the fixed mix

**In short.**
- **Dimension and context together do not give the LCM's lead.** SONAR's row plus its document's
  earlier mean, recoloured to block 4's spectrum, closes −3% of SONAR's gap to the LCM on all heads
  and 19% within a language. That is SONAR's level on all heads, and the plain fixed mix's (20%)
  within a language.
- **So the lead is in the LCM's particular map.** It is not in the spectrum that map gives, nor in
  the context mean, nor in both. The readout from x_n alone is also a linear map of SONAR's rows to a
  low-dimensional spectrum, and it closes 84% on all heads (entry below). The two recolourings close
  22% (SONAR) and −3% (the fixed mix).
- **The share of variance between languages does not carry it.** This arm has more of its variance
  between languages than the fixed mix (0.117 against 0.090) and agrees less. Over the eight arms, the
  correlation of agreement with that share falls to 0.86 on all heads and 0.84 within a language.

**The control.**
- **The map.** `recolour.py` now reads a source laid out as its target row for row. With `--source
  sonar_ctxmean_b4.npy` and that cache's matched weights, it gives the fixed mix block 4's
  weighted-frame spectrum rank for rank, under a random rotation. The direction `no837` zeroes stays
  zero.
- **Fidelity.** Its participation ratio under its own weights is 149, against the target's 143, the
  same as the recoloured SONAR's.
- **Its spectrum** on the tail: participation ratio 148, effective rank 322, 65 / 370 directions for
  50% / 90%, 0.117 between languages, and within-language participation ratio 173.

**Runs.** `ARMS=sonar_ctxmean_recol phase2.sh train`, then `score`, on the RTX 3090: seeds 42 and 43
to step 46,250, about 2 hours each. The machine restarted during scoring, after `seedmatch.py` had
written its scores, and the score stage was rerun for `eval_next.py`.

**The readouts,** beside every other arm's seeds 42 and 43:

| Arm | All 380 heads: matched NMI (null) | First 38 heads, within language (null) | Gap closed, all heads / within | Head-specific part | Share between languages | Copy baseline | FVU_w, each seed |
|---|---|---|---|---|---|---|---|
| LCM, block 4 | 0.0340 (0.0048) | 0.1269 (0.0495) | | 0.0026 | 0.154 | 0.439 | 0.2820 / 0.2812 |
| Readout, x_n and context | 0.0316 (0.0048) | 0.1043 (0.0494) | 86% / 62% | 0.0026 | 0.153 | 0.454 | 0.2746 / 0.2727 |
| Readout, x_n alone | 0.0313 (0.0048) | 0.0907 (0.0495) | 84% / 38% | 0.0026 | 0.147 | 1.056 | 0.2481 / 0.2475 |
| Fixed mix | 0.0197 (0.0048) | 0.0812 (0.0508) | 17% / 20% | 0.0015 | 0.090 | 0.731 | 0.4364 / 0.4356 |
| **Fixed mix recoloured** | **0.0163** (0.0048) | **0.0810** (0.0510) | **−3% / 19%** | 0.0015 | 0.117 | 0.632 | 0.3484 / 0.3476 |
| SONAR recoloured | 0.0206 (0.0048) | 0.0751 (0.0508) | 22% / 10% | 0.0016 | 0.072 | 1.369 | 0.3241 / 0.3236 |
| SONAR, TPU | 0.0169 (0.0048) | 0.0692 (0.0507) | | 0.0013 | 0.043 | 1.533 | 0.4593 / 0.4585 |
| Untrained LCM | 0.0155 (0.0048) | 0.0749 (0.0509) | −8% / 9% | 0.0012 | 0.057 | 0.766 | 0.4370 / 0.4370 |

**What it means.**
- **Two linear maps of SONAR's rows to a low-dimensional spectrum differ by 62 points on all heads:**
  the recoloured SONAR closes 22% and the readout from x_n alone 84%. They differ in two ways.
  - **Which directions get the variance.** The recolouring gives SONAR's own top directions the LCM's
    top eigenvalues. The readout gives them to whichever directions the LCM weighs up.
  - **The coordinate basis.** The recoloured arms are rotated at random, while the readouts come out
    in the LCM's own coordinates, and the trainer weights each coordinate by its inverse variance.
- **The readout from x_n alone under a random rotation separates the two.** It closes 41% on all
  heads and 26% within a language (2026-11-22, the entry above).

**Caveats.**
- **Seeds.** Two, and one rotation seed for the recolouring.

```bash
A=data/activations/lcm-organism
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/recolour.py --source $A/sonar_ctxmean_b4.npy \
    --source-weights $A/sonar_ctxmean_b4.mse_weights_matched.npy --out $A/sonar_ctxmean_recol_b4.npy
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/spectrum.py \
    --arm sonar_ctxmean_recol_b4 $A/sonar_ctxmean_recol_b4.npy $A/sonar_ctxmean_recol_b4.mse_weights_matched.npy \
    --out data/out/lcm-organism/spectrum_ctxmean_recol
ARMS=sonar_ctxmean_recol experiments/lcm-organism/phase2.sh train
ARMS=sonar_ctxmean_recol experiments/lcm-organism/phase2.sh score
```

## 2026-11-21: the readout from x_n alone: the LCM's map of a sentence by itself gives its lead on all heads, and the context adds within a language

**In short.**
- **On all heads, the sentence alone is enough.** Dictionaries on LCM block 4's linear readout from
  x_n alone close 84% of SONAR's gap to the LCM, against 86% with the context's mean as well (two
  entries below). They also reach the LCM's head-specific part, 0.0026.
- **Within a language, the context matters.** The readout from x_n alone closes 38%, and with the
  context's mean 62%. The fixed mix, the same context without the LCM's map, closes 20% (entry
  below).
- **So the LCM's lead has two parts.** How it reshapes each sentence's own directions carries the lead
  on all heads. How it combines the sentence with its context carries most of what it adds within a
  language. Beyond any linear readout, the LCM's states add the last 38% within a language.

**The control** (`readout.py --features row`).
- **The map.** y_n = Wᵀx_n + b: Gate B's `x` ridge, fitted as the full readout on the same 20,000
  prefix documents, and written into `lcm_readout_x_b4.npy`.
- **Fidelity.** Its held-out R² is 0.543 on the scored tail (0.542 on the whole tail), Gate B's 0.544.
- **Its spectrum** is the lowest-dimensional of any arm, with nearly the LCM's share between
  languages:

| Arm (the tail) | Participation ratio | Effective rank | Directions for 50% / 90% | Share between languages | Within-language participation ratio |
|---|---|---|---|---|---|
| SONAR | 413 | 701 | 211 / 711 | 0.043 | 461 |
| LCM, block 4 | 141 | 315 | 63 / 366 | 0.154 | 143 |
| Readout, x_n and context | 122 | 255 | 52 / 294 | 0.153 | 130 |
| Readout, x_n alone | 98 | 212 | 41 / 263 | 0.147 | 108 |

**Runs.** `ARMS=lcm_readout_x phase2.sh train`, then `score`, on the RTX 3090, as for the other
readout arm: seeds 42 and 43 to step 46,250, about 2 hours each, scored at `highest`.

**The readouts,** beside the other arms' seeds 42 and 43:

| Arm | All 380 heads: matched NMI (null) | First 38 heads, within language (null) | Gap closed, all heads / within | Head-specific part | Share between languages | FVU_w, each seed |
|---|---|---|---|---|---|---|
| LCM, block 4 | 0.0340 (0.0048) | 0.1269 (0.0495) | | 0.0026 | 0.154 | 0.2820 / 0.2812 |
| Readout, x_n and context | 0.0316 (0.0048) | 0.1043 (0.0494) | 86% / 62% | 0.0026 | 0.153 | 0.2746 / 0.2727 |
| **Readout, x_n alone** | **0.0313** (0.0048) | **0.0907** (0.0495) | **84% / 38%** | **0.0026** | 0.147 | 0.2481 / 0.2475 |
| Fixed mix | 0.0197 (0.0048) | 0.0812 (0.0508) | 17% / 20% | 0.0015 | 0.090 | 0.4364 / 0.4356 |
| SONAR recoloured | 0.0206 (0.0048) | 0.0751 (0.0508) | 22% / 10% | 0.0016 | 0.072 | 0.3241 / 0.3236 |
| SONAR, TPU | 0.0169 (0.0048) | 0.0692 (0.0507) | | 0.0013 | 0.043 | 0.4593 / 0.4585 |
| Untrained LCM | 0.0155 (0.0048) | 0.0749 (0.0509) | −8% / 9% | 0.0012 | 0.057 | 0.4370 / 0.4370 |

The gap closed is the arm's excess over the null, as a share of the way from SONAR's to the LCM's.

**What it shows.**
- **The gap.** The excess is 0.0265 on all heads, against the LCM's 0.0292 and SONAR's 0.0120. Within
  a language it is 0.041, against 0.077 and 0.019.
- **The two readouts differ only in the context.** They have nearly the same share between languages
  (0.147 and 0.153) and the same agreement on all heads (0.0265 and 0.0268). They part only within a
  language (0.041 against 0.055), which is the context's contribution through the LCM's map.
- **Error.** Its FVU_w, 0.248, is the lowest of any arm, as its dimension would give.
- **Without the context, adjacent rows are not alike.** Its copy baseline reads 1.056 (SONAR 1.533,
  the LCM 0.439). Its next-sentence FVU_w is 0.97 (ridge 0.715).

**Across the seven arms,** both scores rise with the share of the variance between languages: Pearson
0.98 on all heads and 0.88 within a language (Spearman 0.93 and 1.00). That is seven points. The
within-language score holds language fixed, so the share can at most mark something the LCM's map
does.

**What it means.**
- **The lead on all heads comes from the LCM's map of each sentence alone.** That map moves far more
  of the variance between languages (0.043 to 0.147). The recoloured control had the LCM's spectrum
  with SONAR's split between languages (0.072), and it closed 22%. So the language structure the map
  brings forward, more than its dimension, is the likelier carrier.
- **Within a language,** combining the sentence with its context through the LCM's map carries 24
  points of the gap. What the LCM computes beyond its linear readouts carries the last 38.

**Caveats.**
- **Dimension.** This is the lowest-dimensional arm (participation ratio 98), so some of its agreement
  on all heads may be dimension.
- **Seeds.** Two, on one LCM at one block.

```bash
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/readout.py --features row
A=data/activations/lcm-organism
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/spectrum.py \
    --arm lcm_readout_x_b4 $A/lcm_readout_x_b4.npy $A/lcm_readout_x_b4.mse_weights_matched.npy \
    --out data/out/lcm-organism/spectrum_readout_x
ARMS=lcm_readout_x experiments/lcm-organism/phase2.sh train
ARMS=lcm_readout_x experiments/lcm-organism/phase2.sh score
```

## 2026-11-21: the fixed-mix control: SONAR's row plus its document's earlier mean agrees across seeds little more than SONAR

**In short.**
- **Averaging in the context does not give the LCM's lead.** Dictionaries on
  y_n = x_n + mean(x_0 … x_{n−1}), in SONAR's coordinates with nothing fitted, close 17% of SONAR's
  gap to the LCM on all heads and 20% within a language. The linear readout takes the same two inputs
  through matrices fitted to the LCM's states, and it closed 86% and 62% (entry below).
- **So the lead is in the LCM's own linear map** of the sentence and its context, not in the context
  alone.
- **Smoothness is not it either.** The mix's adjacent rows are much more alike than SONAR's, about as
  alike as the untrained LCM's. Like the untrained LCM, it agrees little more than SONAR.

**The control (`ctxmean.py`).**
- **The map.** y_n = x_n + w · mean(x_0 … x_{n−1}) with w = 1, adding nothing at a document's first
  row, on `mc4_sent16_f16.npy`'s rows, laid out as `lcm_b4`.
- **The weight.** w = 1 counts the context so far as one more sentence. Nothing in it comes from the
  LCM. The readout can't supply w, since its matrices map into the LCM's coordinates.
- **Loss weights.** The SONAR arm's `no837`, scaled by 0.563 so the whitened error of predicting the
  mean equals SONAR's. Inverse-variance weights would differ from the SONAR arm's in more than the
  context. They would also give coordinate 837 (variance 8.5e-9, against a median of 4.1e-5) about
  5,000 times the median weight.
- **Rounding.** 0.59% of its float16 values are subnormal, against 0.79% in SONAR's own float16 copy.
- **Its spectrum** (the tail, under those weights) lies between SONAR's and the LCM's:

| Arm | Participation ratio | Effective rank | Directions for 50% / 90% | Share between languages | Within-language participation ratio |
|---|---|---|---|---|---|
| SONAR | 413 | 701 | 211 / 711 | 0.043 | 461 |
| Fixed mix | 235 | 555 | 149 / 660 | 0.090 | 294 |
| LCM, block 4 | 141 | 315 | 63 / 366 | 0.154 | 143 |
| Linear readout | 122 | 255 | 52 / 294 | 0.153 | 130 |

**Runs.** `ARMS=sonar_ctxmean phase2.sh train`, then `score`, on the RTX 3090, as for the readout arm:
seeds 42 and 43 to step 46,250, about 2 h 5 min each, scored at `highest`.

**The readouts,** beside the other arms' seeds 42 and 43:

| Arm | All 380 heads: matched NMI (null) | First 38 heads, within language (null) | Best over runner-up | Head-specific part | Copy baseline | FVU_w, each seed |
|---|---|---|---|---|---|---|
| LCM, block 4 | 0.0340 (0.0048) | 0.1269 (0.0495) | 0.0010 | 0.0026 | 0.439 | 0.2820 / 0.2812 |
| Linear readout of block 4 | 0.0316 (0.0048) | 0.1043 (0.0494) | 0.0005 | 0.0026 | 0.454 | 0.2746 / 0.2727 |
| **Fixed mix** | **0.0197** (0.0048) | **0.0812** (0.0508) | 0.0004 | **0.0015** | 0.731 | 0.4364 / 0.4356 |
| SONAR recoloured | 0.0206 (0.0048) | 0.0751 (0.0508) | 0.0006 | 0.0016 | 1.369 | 0.3241 / 0.3236 |
| SONAR, TPU | 0.0169 (0.0048) | 0.0692 (0.0507) | 0.0004 | 0.0013 | 1.533 | 0.4593 / 0.4585 |
| Untrained LCM, block 4 | 0.0155 (0.0048) | 0.0749 (0.0509) | 0.0003 | 0.0012 | 0.766 | 0.4370 / 0.4370 |

The copy baseline is `eval_next.py`'s FVU_w for predicting each row by the one before, under the
arm's weights: how alike adjacent rows are.

**What it shows.**
- **The gap.** The excess over the null is 0.0149 on all heads, against the LCM's 0.0292 and SONAR's
  0.0120 (17% of the gap). Within a language it is 0.030, against 0.077 and 0.019 (20%).
- **The head-specific part,** 0.0015, is SONAR's level (0.0013), not the LCM's and the readout's
  (0.0026).
- **For the record,** its next-sentence FVU_w is 0.774 (copy 0.731, ridge 0.560).

**What it means.**
- **Rows as alike as the untrained LCM's do not agree more than SONAR's.** Only the two arms whose rows
  are the LCM's own map of its inputs, the LCM and its readout, agree well above SONAR.
- **What the LCM contributes here is a linear map,** the readout's matrices on x_n and on the context
  mean. They could matter for how they weigh the context against the row, or for how they reshape
  x_n's own directions: the readout is far lower-dimensional than the mix. A readout from x_n alone
  (Gate B's `x`, R² 0.544 for seed 42 at block 4) would separate the two (the entry above: the
  reshaping carries the lead on all heads, the context part of it within a language).

**Caveats.**
- **One weight,** w = 1.
- **Dimension.** The mix's participation ratio, 235, is about twice the readout's. Recolouring SONAR
  to the LCM's spectrum closed 22% and 10% alone. A recoloured fixed mix would test the two together
  (the entry above: it closes −3% and 19%).
- **Seeds.** Two.

```bash
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/ctxmean.py --weight 1
A=data/activations/lcm-organism
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/spectrum.py \
    --arm sonar_ctxmean_b4 $A/sonar_ctxmean_b4.npy $A/sonar_ctxmean_b4.mse_weights_matched.npy \
    --out data/out/lcm-organism/spectrum_ctxmean
ARMS=sonar_ctxmean experiments/lcm-organism/phase2.sh train
ARMS=sonar_ctxmean experiments/lcm-organism/phase2.sh score
```

## 2026-11-21: the linear-readout control: block 4's linear readout from x_n and its context's mean agrees across seeds almost as the LCM's states do

**In short.**
- **Most of the LCM arm's lead survives a linear readout.** Dictionaries on Gate B's ridge readout of
  block 4's state, from x_n and the mean of its document's earlier rows, close 86% of SONAR's gap to
  the LCM in seed agreement on all heads, and 62% within a language.
- **It reproduces the head-specific part too:** 0.0026, the LCM's, against SONAR's 0.0013. Heads are
  still not matched one by one.
- **The readout's matrices were fitted to the LCM's states,** so this is still something the LCM
  learned. A fixed mix, SONAR's row plus its document's earlier mean with no LCM, separates averaging
  in the context from the LCM's own map (the entry above: averaging gives little of it).

**The control (`readout.py`).**
- **The map.** y_n = Wᵀ[x_n, mean(x_0 … x_{n−1}), 1{n = 0}] + b, with x the cache the LCM read
  (`mc4_sent16_f16.npy`). It is Gate B's `x_context` ridge (`train_lcm.Ridge`, alpha 1e-3), fitted on
  every row of 20,000 prefix documents (227,595 rows, seed 0) against `lcm_b4`'s states and written
  at every harvest row.
- **Fidelity.** Its held-out R² is 0.686 on the scored tail and on the whole tail, Gate B's 0.686 for
  seed 42 at block 4.
- **Layout.** It is laid out as `lcm_b4`, with its own inverse-variance weights matched to SONAR's
  loss scale.
- **Its spectrum** (`spectrum.py`, the tail, under those weights) is slightly lower-dimensional than the
  LCM's states, with the same split between languages, which the recoloured control lacked:

| Arm | Participation ratio | Effective rank | Directions for 50% / 90% | Share between languages | Within-language participation ratio |
|---|---|---|---|---|---|
| LCM, block 4 | 141 | 315 | 63 / 366 | 0.154 | 143 |
| Linear readout | 122 | 255 | 52 / 294 | 0.153 | 130 |

**Runs.** `ARMS=lcm_readout phase2.sh train`, then `score`, on the RTX 3090.
- **Training:** Phase 2's recipe (`train_pair.py --epochs 3 --pair-term agree --s-pair 0`), seeds 42
  and 43, to step 46,250, at the default precision. About 2 h 5 min a seed, at 6.9 steps a second.
- **Scoring:** at `highest`, as for the TPU arms.
- Outputs in `data/out/lcm-organism/phase2/`.

**The readouts,** beside the Phase 2 arms' seeds 42 and 43:

| Arm | All 380 heads: matched NMI (null) | First 38 heads, within language (null) | Best over runner-up | Head-specific part | FVU_w, each seed |
|---|---|---|---|---|---|
| LCM, block 4 | 0.0340 (0.0048) | 0.1269 (0.0495) | 0.0010 | 0.0026 | 0.2820 / 0.2812 |
| **Linear readout of block 4** | **0.0316** (0.0048) | **0.1043** (0.0494) | 0.0005 | **0.0026** | 0.2746 / 0.2727 |
| SONAR recoloured | 0.0206 (0.0048) | 0.0751 (0.0508) | 0.0006 | 0.0016 | 0.3241 / 0.3236 |
| SONAR, TPU | 0.0169 (0.0048) | 0.0692 (0.0507) | 0.0004 | 0.0013 | 0.4593 / 0.4585 |
| Untrained LCM | 0.0155 (0.0048) | 0.0749 (0.0509) | 0.0003 | 0.0012 | 0.4370 / 0.4370 |

The head-specific part is the first 38 heads' matched NMI minus a random one-to-one pairing's (the
Phase 2 entry's columns otherwise).

**What it shows.**
- **The gap.** The excess over the null is 0.0268 on all heads, against the LCM's 0.0292 and SONAR's
  0.0120 (86% of the gap). Within a language it is 0.055, against 0.077 and 0.019 (62%). The
  recoloured control closed 22% and 10%.
- **Within a seed,** two heads agree at 0.026–0.030, against 0.0305 between matched heads across
  seeds: the LCM's pattern (0.028–0.032 against 0.032–0.034).
- **Error.** Its FVU_w, 0.273–0.275, is below the LCM's 0.281–0.282, as its lower dimension would
  give.
- **Adjacent rows are nearly as alike as the LCM's.** `eval_next.py`'s copy baseline reads 0.454
  (LCM 0.439) and its next-sentence ridge 0.362 (0.309). The dictionaries' next-sentence FVU_w reads
  0.562–0.564 (0.560–0.561).

**What it means.**
- **The lead is mostly in the part of the state that is linear in x_n and its context's mean.** What
  the readout leaves out accounts for 14% of the gap on all heads and 38% within a language: the 31%
  of the state's variance it misses, which includes everything nonlinear in x_n and in the context.
- **The untrained LCM also mixes in context, yet reads like SONAR.** Either a context mean is not
  enough by itself, or the random weights' mixing hides it. The fixed mix tests this (the entry
  above).

**Caveats.**
- **Seeds.** Two.
- **Dimension.** The readout is lower-dimensional than the LCM's states, so some of its agreement may
  be dimension. The recoloured control says that effect is small.
- **Precision.** It trained on the GPU at the default precision, and the other arms on the TPU at
  `high`. The SONAR control's GPU and TPU runs agree within 0.0001 (the Phase 2 entry).

```bash
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/readout.py
A=data/activations/lcm-organism
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/spectrum.py \
    --arm lcm_readout_b4 $A/lcm_readout_b4.npy $A/lcm_readout_b4.mse_weights_matched.npy \
    --out data/out/lcm-organism/spectrum_readout
ARMS=lcm_readout experiments/lcm-organism/phase2.sh train
ARMS=lcm_readout experiments/lcm-organism/phase2.sh score
```

## 2026-11-20: closure on each Phase 2 arm: no arm's heads are more closed than random projections of its own states

**In short.**
- **No arm's heads are closer to closed levels than random projections of the same states.** In every
  arm, the trained heads miss within 2.1 points of what its random-projection heads miss, and the
  first 38 heads read like the other heads.
- **The states set what a head misses.** The share falls from 71% on SONAR to 61% on the untrained LCM
  and 43% on the LCM, for random projections and trained heads alike, as adjacent states grow more
  alike.
- **The spectrum explains almost none of the drop.** SONAR recoloured to the LCM's spectrum reads 70%,
  though the recolouring closed 76% of the gap in FVU_w (entry below).

**Runs.** `closure.py` (experiments/next-sentence, entry of 2026-11-14 there) on each arm's seeds at
step 46,250, on CPU, with its defaults: fit on the 321,161 adjacent pairs in rows 4.0M–4.4M, scored on
the tail's 26,335 pairs from 2,673 documents. Every arm reads the same pairs, since each harvest keeps
the cache's row numbers.
- **SONAR:** `long.sh`'s GPU runs (`doc2_sp0_ph38`, `_s43`, `_s44`) on `mc4_sent16.npy`, which the TPU
  retrain matches (entry below), with `sfa.py`'s directions as a group.
- **The other arms:** the TPU runs on their own harvests, with `--sfa ''`. SONAR's SFA directions are
  not directions of these arms' states.
- About 20 minutes for a two-seed arm. Outputs in `data/out/lcm-organism/closure/<arm>/`.

**The readouts.**

| Arm | Seeds | Heads keep their entry | I(x_t; z_t+1), heads (bits) | Missed: 38 random projections | Missed: first 38 heads | Missed: 38 other heads |
|---|---|---|---|---|---|---|
| SONAR | 42, 43, 44 | 0.095–0.097 | 0.25–0.26 | 71.9% | 71.1–71.4% | 70.8–71.3% |
| SONAR recoloured to the LCM's spectrum | 42, 43 | 0.108–0.109 | 0.33–0.35 | 69.7% | 70.5–70.7% | 70.4–70.5% |
| Untrained LCM, block 4 | 42, 43 | 0.251–0.253 | 1.25–1.28 | 62.8% | 60.7–61.1% | 60.9–61.0% |
| LCM, block 4 | 42, 43, 44 | 0.441–0.448 | 2.17–2.33 | 43.9% | 41.8–43.8% | 43.0–44.1% |

How to read the columns:
- **Missed** is the gap's share of what x_t says about the next entry: I(x_t; z_t+1 | z_t) over
  I(x_t; z_t+1), summed over a group's heads. A closed head misses 0%.
- **x_t is each arm's own row**: the LCM's state, which carries its document's context, or SONAR's
  embedding. Each share is against that arm's own micro level.
- **The first 38 heads** are the pair runs' slow heads. At `--s-pair 0` they are positions only.
- **The other heads** are 38 drawn at random from the rest of layer 0.
- **Random projections** take the argmax over 32 random directions of the arm's centered rows.
- **Errors.** The document bootstrap's error on a group's gap is 0.003–0.006 bits: about 1.4 points of
  share on SONAR and 0.25 on the LCM. The spread between seeds is larger.

**Heads read like random projections in every arm.**
- **Against the arm's random projections:** 0.5–1.1 points below on SONAR, 0.8–1.0 above recoloured,
  1.7–2.1 below on the untrained LCM, and from 2.1 below to 0.2 above on the LCM.
- **The test can tell trained heads from projections.** On SONAR, the Fano arm's slow heads miss 27%
  against the projections' 72% (experiments/next-sentence, 2026-11-14). No arm here comes near that.

**What a head misses follows how alike adjacent states are.**
- **The order.** Heads keep their entry 10% of the time on SONAR, 11% recoloured, 25% untrained and
  45% on the LCM. Missed falls 72, 70, 63, 44 for the projections in the same order. `eval_next.py`'s
  copy baseline orders the arms the same way (entry below).
- **The drop from SONAR to the LCM is 28.0 points on the random projections.**
  - The untrained LCM's states account for 9.2 points (33%), and training for the other 18.9 (67%).
  - Recolouring SONAR moves it 2.3 points (8%).
  - On the trained heads the drop is 27.9 points: the untrained LCM 10.3 (37%), training 17.6 (63%),
    recolouring 0.7 (2%).
- **Closure and seed agreement part ways across arms.** The untrained LCM's states are smoother than
  SONAR's, and its heads miss less, yet in Phase 2 they agreed across seeds no more than SONAR's.

**The most closed heads are a tail, not a group.** Only a few heads in any arm read below its most
closed random projection.
- **LCM:** 37 of the 228 heads scored, at down to 25%, against the projections' lowest, 42.1%. They
  carry less about the next entry: 1.5–2.3 bits, against 2.4–2.5 for every projection. Seed 43's
  first 38 heads hold 15 of them, and the other five groups 2–6 each.
- **Untrained LCM:** 21–29 of each 38 heads, but by little: down to 50.5%, against 61.6%.
- **SONAR and recoloured:** 1–3 of each 38.

**The SONAR rows reproduce the 2026-11-14 run.**
- **The same values to the fourth decimal:** the random projections' 0.2150 bits, SFA's 0.0042 (0.8%
  missed) and seed 42's first 38 heads' 0.1877.
- **Its other heads read 0.1874, against 0.1784 then.** They are a different random draw, since each
  checkpoint's draw follows the earlier ones' on the command line.

**SFA on the other arms** was left until an arm's heads read more closed than its own random
projections, which none do.

**Caveats.**
- **The 2026-11-14 entry's caveats apply:** one step of past and future, and a gap bounded from below
  by a probe linear in x_t.
- **Seeds.** The recoloured and untrained arms have two seeds each.

```bash
L=data/out/sonar/next-sentence/long
JAX_PLATFORMS=cpu uv run python experiments/next-sentence/closure.py \
    --ckpt $L/doc2_sp0_ph38 $L/doc2_sp0_ph38_s43 $L/doc2_sp0_ph38_s44 \
    --cache data/sonar_embeddings/mc4_sent16.npy --out data/out/lcm-organism/closure/sonar
P=data/out/lcm-organism/phase2_tpu
JAX_PLATFORMS=cpu uv run python experiments/next-sentence/closure.py \
    --ckpt $P/lcm_b4_doc2_sp0_ph38_s42 $P/lcm_b4_doc2_sp0_ph38_s43 $P/lcm_b4_doc2_sp0_ph38_s44 \
    --cache data/activations/lcm-organism/lcm_b4.npy --sfa '' --out data/out/lcm-organism/closure/lcm_b4
# likewise init_b4 and sonar_recol_b4, at seeds 42 and 43
```

## 2026-11-20: Phase 2 on a TPU: codes on the LCM's states agree across seeds at over twice SONAR's level, and the spectrum doesn't explain it

**In short.**
- **The plan's claim holds.** Codes on LCM block 4's states match across seeds well above codes on
  SONAR or on the untrained LCM: 2.4 times SONAR's excess over the null on all heads, and 4.2 times
  within a language. A third seed agrees with the first two at the same level.
- **It is not effective dimension, and not context averaging.** SONAR recoloured to the LCM's
  spectrum closes 76% of the gap in FVU_w, but only 22% of the gap in agreement, and 10% within a
  language. The untrained LCM mixes in context, and agrees no more than SONAR.
- **It is still not heads identified one by one.** Each head's best partner beats its runner-up by
  at most 0.001, and the matched NMI, 0.034, is far below the pair objective's 0.41.

**Runs.** Every arm is `train_pair.py --epochs 3 --pair-term agree --s-pair 0`, the SONAR control's
recipe (`doc2_sp0_ph38`), to step 46,250, keeping only the last checkpoint.
- **Where.** A v5e-4 under Flex-start through `experiments/cloud`, one run per chip, at matmul
  precision `high`. Each run took about 1 h 57 min, against the plan's estimate of about an hour.
- **Scoring,** at `highest`: `seedmatch.py --labels <harvest>.langs.npy --label-rows 262144
  --min-rows 1000`, with `--c` for three seeds, then `eval_next.py`. The queues are in `queues/` and
  the outputs in `data/out/lcm-organism/phase2_tpu/`.
- **The inputs.**
  - **LCM:** `harvest_lcm.py` on Phase 1's seed 42 (step 7,216) at block 4: the rows below 4,400,014
    and the tail, at float16, with matched MSE weights (`lcm_b4`).
  - **Untrained:** the same harvest of that run's step 0 (`init_b4`).
  - **SONAR:** `mc4_sent16_f16.npy` with the control's `no837` weights.
  - **Recoloured SONAR** (`recolour.py`, below), added after the spectrum check.
- **Interruptions.** LCM seed 44 and both recoloured runs died together when the VM's RAM disk filled,
  and resumed from their last checkpoints: step 7,000 for seed 44, and within the first 500 steps for
  the others.

**The readouts.**

| Arm | Seeds | All 380 heads: matched NMI (null) | First 38 heads, within language (null) | Best over runner-up | FVU_w, each seed |
|---|---|---|---|---|---|
| LCM, block 4 | 42, 43, 44 | **0.0340** (0.0048) | **0.1269** (0.0495) | 0.0010 | 0.2820 / 0.2812 / 0.2798 |
| SONAR recoloured to the LCM's spectrum | 42, 43 | 0.0206 (0.0048) | 0.0751 (0.0508) | 0.0006 | 0.3241 / 0.3236 |
| SONAR, TPU | 42, 43, 44 | 0.0169 (0.0048) | 0.0692 (0.0507) | 0.0004 | 0.4593 / 0.4585 / 0.4597 |
| SONAR, GPU (`long.sh`) | 42, 43, 44 | 0.0168 (0.0048) | 0.0692 (0.0507) | 0.0004 | 0.4593 / 0.4586 / 0.4596 |
| Untrained LCM, block 4 | 42, 43 | 0.0155 (0.0048) | 0.0749 (0.0509) | 0.0003 | 0.4370 / 0.4370 |

How to read the columns:
- **The matched NMI** is between seeds 42 and 43. The null shuffles B's rows.
- **Within language** scores the first 38 heads' matched pairs inside each language's rows, against a
  null that shuffles within each language (84 languages, 260,453 rows). At `--s-pair 0` nothing
  distinguishes those 38 heads from the rest.
- **Best over runner-up** is the median, over those heads, of the best partner's NMI minus the second
  best's.
- **FVU_w** is held-out, against the input, under each arm's own weights.

**The TPU retrain matches the GPU runs** within 0.0001 on every NMI and FVU_w, seed by seed, so the
arms compare across the two machines. Only the loop count below differs, by one head.

**Against the plan's predictions.**
- **The untrained arm** was predicted to read near SONAR. It reads 0.0155, slightly below, so mixing in
  context at init does not make heads agree.
- **The LCM arm** was predicted to read above both, and it does. Its excess over the null is 0.0292,
  against SONAR's 0.0120 and the init's 0.0107.
- **Language.** Within a language, the LCM's excess is 0.077 against SONAR's 0.019. So the lead is not
  language: it grows once language is held fixed.
- **Block-level.** The margin was predicted near 0.003. It reads 0.0010.
- **What would have counted against,** the three arms reading alike, did not happen.
- **Seed 44,** the plan's third seed if the result was positive, matches seeds 42 and 43 at the same
  level: 0.0341 and 0.0324 on the first 38 heads, against 0.0326 between 42 and 43.
  - Its matches close a loop (42→43→44 lands where 42→44 does) for 3 of the 38 heads, against 1 by
    chance.
  - SONAR's three seeds close 1 on the TPU and 2 on the GPU.
  - These counts are too small to read much from.

**Still not heads identified one by one.** On the first 38 heads:
- **Head-specific agreement.** The matched NMI minus a random one-to-one pairing's is 0.0026–0.0029
  for the LCM's three seed pairs. It is 0.0013–0.0014 for SONAR on the TPU (0.0012–0.0014 on the
  GPU), 0.0016 recoloured and 0.0012 untrained.
- **Most of each arm's NMI is a level any pairing reaches.** The LCM doubles the head-specific part
  too.
- **A head agrees with its match in another seed scarcely more than two heads of one seed agree:**
  0.032–0.034 against 0.028–0.032 for the LCM, and 0.016–0.017 against 0.014–0.015 for SONAR.

**Is it effective dimension?** `spectrum.py` measures each arm in the frame its loss weighs (rows
scaled by the square roots of their MSE weights), on 262,144 prefix rows and the tail's 262,144. The
tail:

| Arm | Participation ratio | Effective rank | Directions for 50% / 90% | Share between languages | Within-language participation ratio |
|---|---|---|---|---|---|
| LCM, block 4 | 141 | 315 | 63 / 366 | 0.154 | 143 |
| SONAR recoloured | 149 | 324 | 66 / 371 | 0.072 | 166 |
| Untrained LCM | 332 | 602 | 143 / 684 | 0.057 | 387 |
| SONAR | 413 | 701 | 211 / 711 | 0.043 | 461 |

- **The LCM's states are much lower-dimensional than SONAR's.** A code of fixed capacity covers more
  of such a source, and might agree more for that reason alone.
- **The gap in FVU_w is the spectrum.**
  - Take the rate at which reverse water-filling on each arm's spectrum (`pareto.gaussian_reference`)
    reaches the arm's FVU_w: 305 bits for the LCM, 315 for the untrained LCM and 363 for SONAR.
  - Measured against its own spectrum, SONAR's code is not the worse one.
- **The untrained arm already argued against dimension.** It is lower-dimensional than SONAR, and it
  agrees slightly less.

**The control (`recolour.py`).**
- **The map.** SONAR's rows go through one linear map that gives them the LCM's weighted-frame spectrum
  rank for rank:
  - y = c R diag(s) Uᵀ(√w ⊙ x), where U and λ are SONAR's weighted-frame eigenvectors and eigenvalues;
  - s = √(λ_LCM / λ_SONAR);
  - R is a random rotation, so the trainer's inverse-variance weights stay near uniform;
  - c brings each coordinate's variance near 1;
  - the one direction the `no837` weights null is zeroed.
- **Layout and training.** It is laid out as the `lcm_b4` harvest, with its sidecars and its own
  matched weights. It trains as the LCM arm, with only the cache and the weights changed.
- **Fidelity.** Under its own weights its participation ratio reads 149, against the target's 143.
  The float16 copy rounded 102 of its 4.8 billion values to zero.

What it shows:
- **FVU_w** reads 0.324: 76% of SONAR's gap to the LCM.
- **Agreement.** The excess on all heads, 0.0158, closes 22% of the gap. Within a language, 0.024
  closes 10%.
- **So the spectrum accounts for most of the LCM arm's lower error, and little of its agreement.**
- **At nearly the same spectrum,** the LCM's rows code to 0.281 and the recoloured rows to 0.324: 305
  Gaussian-equivalent bits against 255. The LCM's states have structure beyond their covariance that
  the code uses, which the recoloured rows lack.

**Context averaging does not explain it either.** `eval_next.py`'s copy baseline, which predicts each
row by the one before, measures how alike adjacent rows are. Its FVU_w is 0.44 for the LCM, 0.77 for
the untrained LCM, 1.37 recoloured and 1.53 for SONAR. The untrained LCM's adjacent rows are far more
alike than SONAR's, yet its codes agree no more.

**Caveats.**
- **The control matches the total spectrum, not the split between languages.** Its share between
  languages is 0.072 against the LCM's 0.154, and its within-language participation ratio is 166
  against 143.
  - A control that recolours the within- and between-language covariances separately would close
    that.
  - It matters most for the all-heads score, and less within a language, where the LCM leads most.
- **Seeds.** The control and the untrained arm have two seeds each.
- **Next-sentence FVU_w doesn't compare across arms,** since the LCM's rows carry their document's
  context. For the record, against each arm's copy and ridge baselines:
  - LCM: 0.559–0.561 (copy 0.439, ridge 0.309);
  - untrained: 0.746 (0.766, 0.542);
  - SONAR: 1.213–1.214 (1.533, 0.897);
  - recoloured: 1.190 (1.369, 0.832).

**Not yet run** (`closure.py` on each arm has run since: the entry above):
- the plan's across-models test, a dictionary on LCM seed 43's states, which needs `--cache-b`;
- Phase 3.

## 2026-11-20: Phase 1 on the RTX 3090: both gates pass, and the seeds' states agree up to nearly a rotation

**Runs.** `train_lcm.py --cache data/sonar_embeddings/mc4_sent16_f16.npy --epochs 1 --seed 42`, then
`--seed 43`: the default 104.9M-parameter SwiGLU model, 7,216 steps each at TF32, 51 minutes for both.
The outputs are in `data/out/lcm-organism/lcm_l8_d1024_s{42,43}`. The readings come from `phase1.sh`, into
`data/out/lcm-organism/phase1/`, with its tail-only harvests in `phase1/tail`.

**Gate A: it has learned dynamics.** Held-out next-sentence FVU_w on the 26,335 pairs:

| Step | Seed 42 | Seed 43 |
|---|---|---|
| 0 (untrained) | 2.587 | 2.580 |
| 2,500 | 0.842 | 0.845 |
| 5,000 | 0.829 | 0.829 |
| 7,216 | **0.8245** | **0.8250** |

- **The margin.** Against the context ridge's 0.870, the LCM is 0.045 better. The seeds differ by
  0.0005, so the margin is about 90 times the spread. Against the ridge from x_t (0.898), it is 0.073
  better.
- **Most of the gain came early.** The last 2,216 steps added 0.004, so a second epoch would probably add
  little.

**Gate B: the state is more than its input.** The held-out R² of the state on the scored tail's 32,771
rows, from ridges fitted on 227,791 rows of 20,000 training documents:

| Block | Seed | x_n | x_n + earlier mean | x_n read alone |
|---|---|---|---|---|
| 4 | 42 | 0.544 | 0.686 | 0.730 |
| 4 | 43 | 0.573 | 0.714 | 0.736 |
| 8 | 42 | 0.348 | 0.501 | 0.582 |
| 8 | 43 | 0.343 | 0.497 | 0.584 |

- **What the context adds.** Even counting everything the model computes from x_n alone, x_n explains
  73% of block 4's state and 58% of block 8's. The context supplies the remaining 26-27% at block 4 and
  42% at block 8.
- **The plan's reading.** It said to move to a later block only if x_n explained nearly all of the
  state, so both blocks qualify. Block 8 holds more context, by this measure.
- **The prefix mean carries much of it.** A linear readout from x_n and the mean of the earlier rows
  already reaches 0.69-0.71 at block 4, close to the 0.73 of the alone state.

**Agreement between the seeds' states** (`stateagree.py`, on the tail's last 65,536 rows, maps fitted on
the first half and scored on the second):

| Block | Models | CKA | Ridge R², 42→43 / 43→42 | Procrustes | Whitened Procrustes |
|---|---|---|---|---|---|
| 4 | trained | 0.912 | 0.868 / 0.865 | 0.183 | 0.801 |
| 4 | untrained inits | 0.378 | 0.153 / 0.154 | 1.380 | 1.654 |
| 8 | trained | 0.911 | 0.910 / 0.911 | 0.102 | 0.678 |
| 8 | untrained inits | 0.297 | 0.077 / 0.080 | 1.518 | 1.781 |

The Procrustes columns are residuals: 0 is identical, and about 2 means independent. CKA with rows
shuffled is at most 0.005.

- **Training makes the two seeds' representations far more alike** than their inits are.
- **Nearly a rotation.**
  - A rotation, after scaling each side to unit total variance, explains 82% of the other seed's
    variance at block 4 and 90% at block 8.
  - A general linear map explains 87% and 91%.
  - So in the variance-weighted metric, the states are identified nearly up to a rotation and a scale.
    The plan expected the rotation test to read clearly worse than the linear map, and it reads only a
    little worse.
- **But the agreement is in the high-variance directions.**
  - Whitened, every direction counts equally, and the residual is 0.80 at block 4 and 0.68 at block 8.
    That is well short of the inits' 1.65-1.78, and far from identity.
  - So the shared part is a high-variance subspace, while most of the 1,024 directions, each with
    little variance, are not shared.
  - Some of the whitened residual may be estimation noise in the smallest directions, fitted from
    32,768 rows. The canonical correlations would show how large the shared subspace is.
- **What the theory suggested by analogy** was that learned dynamics may fix a subspace but not a basis.
  Here the subspace is shared, and within it the two seeds differ by little more than a rotation.
  Categorical heads cannot rotate, so Phase 2 asks whether the dictionary supplies the basis.

**Scale and range.** The trained states are large.
- The largest magnitudes in the tail harvests are 150 (seed 42) and 101 (seed 43) at block 4, and 2,320
  and 1,943 at block 8. The untrained inits stay under 19.
- All of these fit float16, whose relative precision is the same at any magnitude. But a few very large
  coordinates at block 8 are a scale question for the dictionary recipe in Phase 2, alongside the norm
  question noted on 2026-11-19.

## 2026-11-19: throughput pilot on the RTX 3090: 23 minutes per epoch

**Run.** `train_lcm.py --cache data/sonar_embeddings/mc4_sent16_f16.npy --seed 42 --max-steps 175`, at the
default size (104.9M parameters, 512 documents per step) and JAX's default matmul precision on the GPU
(TF32). That is about 1.06M rows. This run's MLP was built from `nn.Dense`, and its output was deleted
when the MLP moved to `NLinear`. The rerun below is in `data/out/lcm-organism/pilot_gpu`.

**Speed.**
- **Training.** From step 50 on it trains at 31,300 rows per second: 6,058 real rows per step in 0.194 s.
  An epoch is 7,216 steps, which comes to 23 minutes.
- **FLOP rate.** Counting the padded positions, which are 26% of each batch, that is 27 TFLOP/s at
  6 × parameters per position. A 3090's dense TF32 peak is about 36 TFLOP/s.
- **Not the loader.**
  - A rerun with every batch already in the page cache trained at the same rate. There the loader alone
    delivers 637,000 rows per second.
  - Read cold from disk, the loader delivers 31,900 rows per second. That is nearly the same rate,
    which is a coincidence: random reads through mmap fault one page at a time.
  - So an epoch here that isn't in the page cache would also be held to about this rate by the disk. A
    TPU host that holds the cache in tmpfs reads it from memory.

**Learning.** The loss fell from 0.111 at step 25 to 0.044 by step 75, and flattened there: with 175
steps, the learning rate was still warming up. The held-out next-sentence FVU_w was 2.589 untrained and
0.966 at step 175. That is below the training mean (1.000), still short of the ridge from x_t (0.898).

**Reproducible.** The rerun matched every logged loss and both FVU_w values to every digit.

**Checkpoints.** 1.1 GB each, holding the parameters and both of Adam's moments. Step 0 is 372 MB.

**For the TPU estimate.**
- The plan's figure of "ten minutes per epoch at 30% of bfloat16 peak" leaves out two things. Matmul
  precision `high` takes three bfloat16 passes, which puts a v5e chip's ceiling near 66 TFLOP/s. And a
  quarter of every batch is padding.
- At this pilot's FLOP rate, an epoch on one chip would be about as long as on the 3090. That is an
  estimate until a TPU pilot runs.

**Rerun with the MLP from `NLinear`, SwiGLU and bilinear (same day).** The same 175 steps, after `GLU`
replaced the `nn.Dense` MLP. The init draws differ, because the parameter tree changed.

| Run | Rows/s (steps 50-175) | Loss at 175 | Held-out FVU_w, step 0 → 175 |
|---|---|---|---|
| `--gate silu` (SwiGLU), `pilot_gpu` | 30,900-31,300 | 0.0442 | 2.587 → 0.970 |
| `--gate none` (bilinear), `pilot_gpu_bilinear` | 31,100-31,500 | 0.0426 | 2.585 → 0.935 |

- **Speed.** Unchanged from the `nn.Dense` MLP (31,300 rows/s). Holding both up-projections in one
  weight costs nothing here.
- **The bilinear run is ahead at step 175.** But that is one seed, still inside the learning-rate
  warmup, so it doesn't compare the two MLPs.

## 2026-11-19: Phase 0 built; Gate A's baselines on mc4_sent16

**The baselines** (`train_lcm.baselines`), on `eval_next.py`'s 26,335 held-out adjacent pairs under the
`no837` weights. The ridges are fitted on the 191,440 adjacent pairs of 20,000 training documents:

| Predictor of x_{t+1} | Held-out FVU_w |
|---|---|
| the training mean | 1.0001 |
| copying x_t | 1.5327 |
| ridge from x_t | 0.8977 |
| ridge from x_t and the mean of the run's earlier rows | 0.8697 |

- The first three reproduce `eval_next.py`'s 1.0001, 1.53 and 0.896 (2026-10-13). The ridge differs
  in the third decimal because it is fitted on another sample.
- The context mean is worth 0.028 of FVU_w. Gate A asks the LCM to beat 0.870 by more than its seed
  spread.

**Checks on CPU.**
- **A planted dynamic.** On a synthetic cache whose next row is a rotation of the row before last, a
  two-block LCM reaches held-out FVU_w 0.23 in 400 steps. The ridge from x_t reads 0.995 there, and the
  context ridge 0.66 (`tests/test_lcm_organism.py`).
- **A harvest drops into the dictionary trainer.**
  - I trained a one-block, 1024-wide LCM for 20 steps on the float16 cache, and harvested 200k prefix
    rows plus the 262,144-row tail.
  - `train_pair.py` then ran a 77-step epoch on the harvest with its matched weights, and the loss fell
    from 1.21e-4 to 1.04e-4.
  - The harvest's scored tail is the cache's, 32,771 rows.
- **The states' scale.** Rows of that barely trained harvest have mean norm about 40, against about
  0.2 for SONAR rows. The dictionary init rescales to the input, and the matched weights put the loss on
  SONAR's scale. Whether the recipe's other settings are scale-free is still to check, in Phase 2.

## 2026-11-19: the float16 copy of mc4_sent16

`f16cache.py --bulk <dir> --weights data/sonar_embeddings/mc4_sent16.mse_weights.npy --weights
data/out/sonar/sentcache/mse_weights_no837.npy` wrote `data/sonar_embeddings/mc4_sent16_f16.npy`:
44,113,823 rows × 1024, 84.1 GiB. It took 8 minutes, reading at 0.36 GiB/s. The array is on another
disk behind the symlink (`float16.array` in its `meta.json`), and the sidecars are hard links to
mc4_sent16's.

- **Range.** The largest magnitude is 0.165, far inside float16's range, and no value is non-finite.
- **Small values.**
  - 0.8% of values (359M of 45.2G) fall in float16's subnormal range, below 6.1e-5, where the
    spacing is fixed at 6e-8.
  - 175,165 nonzero values, 4e-6 of the total, round to zero.
- **What rounding costs.** The figures are the same on 32,768 rows spread over the training prefix as
  on the 32,768-row tail:
  - each row moves by 2.07e-4 of its norm (median; the largest is 2.8e-4);
  - the whitened FVU of the rounding is 1.95e-7 under `mc4_sent16.mse_weights.npy`, and 5.4e-8
    under the control's `no837` weights.

  This matches the plan's measurement on the tail (2026-11-18).
- **Read back.**
  - Every checked row equals the source rounded to float16.
  - Through the symlink, `StackedPairSource` returns float32 rows equal to the rounded source.

# Compositional cells

*Research note from the GPT-2 small (layer 8) Ontologizer experiments, 2026-10-05; work in progress. Companion to [OVERVIEW.md](OVERVIEW.md).*

**Status (2026-10-07): with a strong judge and good descriptions, composition adds little at layer 0 (+0.01–0.02); the max-activating illusion is robust; layers 3–4 remain undescribable.** Jade and Claude agreed (2026-10-05) to
prioritize this track.

## Stronger judge: Claude Opus 5.5 (2026-10-07) — read this first
Plain API calls, no tools, the same instructions and the same detection items as the sandboxed Gemini judge; paired by label.

| set | Gemini | Claude |
|---|---|---|
| single cells, 9 private layer-0 heads: 8 top / 24 random / tagged 8-32-8 | 0.581 / 0.582 / 0.622 | 0.663 / 0.681 / 0.694 |
| private layer-0 intersections (6 pairs), tagged 8-32-8; nulls | 0.661; 0.50 | 0.704; 0.50 |
| direct layer-0 intersections (6 pairs); nulls | 0.641; 0.498 | 0.728; 0.479 |
| direct layer-0 single cells, 24 random members | 0.577 | 0.711 |
| direct single cells, top members, layers 0/1/2/3/4 | .87/.77/.68/—/.50 | .91/.88/.76/.53/.51 |
| direct single cells, random members, layers 0/1/2/3/4 | .58/.56/.51/—/.50 | .65/.60/.53/.52/.52 |
| private, same head at layers 0×1 / 0×2; different heads 0×2 | .602 / .543; .539 | .675 / .572; .568 |

- **Claude is the better judge on every set,** by 0.05 to 0.10, and ahead on every head of every single-cell arm.
- **The tagged 8/32/8 mix stays the best description input,** by a smaller margin with Claude; explicit contrast (a "differential"
  instruction, or tagged non-members) adds nothing with Claude. Effort: medium is the sweet spot (max adds ~0.01 at ~25x the output).
- **Composition adds little once single cells are described well:** intersections beat single cells by ~0.01–0.02 at layer 0. Most of
  the earlier gap came from describing single cells with a weak judge and poor samples.
- **The max-activating illusion is robust** and grows with depth; **layers 3–4 stay at chance** even from top members, and 48+16-token
  snippets don't help. Next: read deep heads as axes (their few principal directions), or describe labels by their effect on the
  next-token distribution.

## Clean re-run (2026-10-06) — read this first
On 2026-10-05 we found that the judging agents could read the answer keys, and, after the keys were moved, the earlier runs'
answers; some answer files copied their key. Every judge number on this page below this section came from those leaky runs.
Everything was re-judged with each call in a filesystem sandbox that contains only that call's task file,
with reshuffled items. Mean balanced accuracy over heads, random members (0.50 = chance):

| set | clean | leaky run |
|---|---|---|
| D40 direct, layer-0 intersections (6 pairs) | 0.64 | 0.77–0.96 |
| D40 direct, random-partition nulls | 0.50 | ~0.49 |
| D40 private, layer-0 intersections (6 pairs) | 0.66 | 0.72–1.00 |
| D40 private, nulls | 0.50 | ~0.50 |
| D40 private, single cells of the same heads | 0.59 | 0.59–0.65 |
| D40 direct, single cells (layer 0 / 1 / 2 / 4) | 0.58 / 0.56 / 0.51 / 0.50 | ~0.59 at layer 0 |
| D40 private, same head at layers 0 and 1 | 0.60 | new |
| D40 private, same head at layers 0 and 2 (nulls 0.50) | 0.54 | new |
| D40 private, different heads at layers 0 and 2 | 0.54 | new |

- **Composition adds a modest, significant amount at layer 0**: about +0.06 over single cells (4–5 label-level standard errors),
  on a discrimination within the same cell. Not the large effect the leaky runs suggested.
- **The leak inflated exactly the headline.** Single-cell scores came out the same; intersection scores fell (the private pair
  16×24 went from "1.000" to 0.57).
- **Deep refinements aren't nameable even given their parent.** A head's layer-2 label refines its own layer-0 label far more than
  another head's does (3–4× the label information, judge-free), yet the judge names both equally poorly (0.54).
- **Description protocol (Jade's idea, 2026-10-06/07).** Single cells, nine D40 private layer-0 heads, the same detection items for
  every arm: 8 top members 0.581; 24 random 0.582; tagged mix of 6 top + 12 typical + 6 boundary 0.613; tagged 12/24/12 0.617;
  tagged 8/32/8 0.622; the 6/12/6 mix with tags removed 0.594; the mix plus 12 tagged non-members 0.623; the mix with a no-tools
  instruction to the judge 0.611. Tagging carries most of the gain; bigger mixes and non-member contrast add about a point each;
  whether the judging agent uses tools makes no difference.
- **Composition under the better protocol.** On the six private intersections (new shared detection items) the tagged mix scores 0.649
  against 0.642 for 24 random intersection members; its nulls score 0.498. The mix helps single cells far more than intersections, so
  the composition advantage shrinks from about +0.06 to about +0.03 against the best single-cell arms. Composition still adds nameable
  meaning, but part of the earlier gap came from describing single cells poorly.

## Earlier work on this page (leaky judge; numbers superseded by the section above)

## The hypothesis
Single Ontologizer cells, especially in deep layers, may be hard to describe not because autointerp is weak but because
the meaningful units are *combinations* of cells: a label in one head, read together with a label in another head (same
layer), or with a label in an earlier layer (a conditional path). Jade's framing: cells are "only fully meaningful in a
compositional context"; interactions between rare pairs being incoherent is evidence *for* coherence (presheaf-style
reasoning). If true, deep-layer interpretability should describe pairs or paths, not more samples per cell; and design
(ii) in the transcoder-designs notes (conditional codes) is the architecture that makes paths first-class.

## The evidence so far (D40 direct, 2026-10-03)
Within a layer-0 cell *a* (head h), split members by their label in another head h′: the largest sub-cell *a∧b* (~11% of
*a*) against the rest of the same cell *a∧¬b*. Describe *a∧b* from 24 random members; the judge then tells *a∧b* members
from *a∧¬b* members. Layer 0: balanced accuracy 0.77 / 0.77 / 0.76 / 0.82 / 0.96 (five pairs). Layer 2: 0.50 (six pairs).
The layer-0 descriptions name topics and syntactic roles (banking, landscapes, cognition, second-person instructions,
attributive nouns), where single cells read as token classes.

## The confound (found 2026-10-05)
The single-cell baseline it was compared with (random-member detection 0.55–0.69) used descriptions written from **8
top-margin** members; the intersections used descriptions from **24 random** members. The gap may be "random-member
descriptions generalize better" rather than "composition carries meaning". The same-protocol control is the **R variant**
of the single-cell judge (describe each cell from 24 random members, detect on random members), prepared on 2026-10-03 but
killed by the out-of-memory crash before any answers. It must run before the intersection result is claimed.

## The missing null
For each intersection task, a twin where "membership" is a **random subset of cell a** of the same size as *a∧b*,
described from 24 of its members and detected against the rest of *a*. If the judge scores above 0.5 there, the protocol
leaks (generic descriptions, or a bias toward answering "member") and the real scores are inflated by that amount.

## Planned runs, in order
1. **R-judge** on D40 direct and D40 private, layers 0–1 (control for the confound). Already prepped for direct.
2. **Null intersections**: random-partition twins of the existing D40 direct layer-0 tasks (judge dir
   `judge/D40_direct_intersect_null`).
3. **Pair selection**: compute the pairwise label MI of the five judged pairs (is the 0.96 pair, H2×H4, also the most
   redundant?); then tasks on the **lowest-MI** pairs (`--pair-select low_mi`).
4. **Cross-layer pairs**: cell *a* at layer 0 ∧ cell *b* at layer 2, detected against *a*∧¬*b*, with the same null
   (judge dir `judge/D40_direct_xlayer`). Within-layer-2 pairs scored 0.50; this tests whether a deep label means something
   *conditional on* a coarse cell.

## What each outcome would mean
- R-judge ≈ intersections (0.77–0.96): random-member descriptions were the driver; composition not shown.
- R-judge ≪ intersections, nulls ≈ 0.5: composition carries nameable meaning at layer 0 (the headline stands).
- Cross-layer > 0.5 with nulls ≈ 0.5: deep labels are meaningful *as refinements of coarse cells*; build design (ii).
- Cross-layer ≈ 0.5: deep labels aren't nameable even conditionally (with this judge); look elsewhere (steering-based or
  transcoder-based characterisation).

## Results (2026-10-05 evening)
- **R-judge (the confound control), D40 direct, single cells described from 24 random members:** L0_H17 0.591,
  L0_H6 0.587, L1_H25 0.505, L1_H5 0.522 (balanced accuracy, random detection set) — no better than the top-member
  descriptions (0.55–0.59). Random-member description is not what drove the intersection scores, and the intersections
  (0.77–0.96) were measured on a *harder* discrimination (against members of the same cell). **The confound is ruled out.**
- **R-judge on the 40M private model** (same protocol): L0_H19 0.594, **L0_H7 0.766**, L1_H7 0.642, L1_H8 0.539 (vs 0.608,
  0.685, 0.585, 0.548 with top-member descriptions). For the private model, random-member descriptions *do* help some
  cells, and one single cell (L0_H7) reaches the low end of the direct model's intersection range. So: the confound is
  ruled out for the direct model, partly real for the private model; the private model's intersections are queued as
  their own test before any claim is made about it.
- **Random-partition nulls** (first three pairs): 0.481, 0.489, 0.498 — the protocol does not leak.
- Pending: remaining nulls, lowest-MI pairs, cross-layer L0 × L2 pairs (+ nulls), private-model R-judge.
- **Queued: does it hold with per-head dropout?** (Jade.) 10M private models with head dropout 0 / 0.1 / 0.3 (no winner
  dropout): layer-0 single cells under both description protocols, and four random layer-0 pairs as intersections with
  random-partition nulls (`run_comp_dropout.sh`). Prior expectation, stated before the result: head dropout forces each
  head to be useful alone, so single cells may become *more* nameable and the intersection advantage *smaller*; if the
  advantage survives dropout, composition is not just an artefact of heads leaning on each other.

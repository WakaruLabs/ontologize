# Compositional cells

*Research note from the GPT-2 small (layer 8) Ontologizer experiments, 2026-10-05; work in progress. Companion to [OVERVIEW.md](OVERVIEW.md).*

**Status (2026-10-06): modest, real effect at layer 0, measured with a clean judge.** Jade and Claude agreed (2026-10-05) to
prioritize this track.

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
- Describing from 24 random members does not beat 8 top members on single cells (0.57 vs 0.59). A description-protocol
  experiment (tagged top/typical/boundary mixes, 48 random, contrast with non-members, margin-weighted sampling) is running.

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

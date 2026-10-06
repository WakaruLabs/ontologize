# Compositional cells

*Research note from the GPT-2 small (layer 8) Ontologizer experiments, 2026-10-05; work in progress. Companion to [OVERVIEW.md](OVERVIEW.md).*

**Status: hypothesis with one encouraging result and one known confound; controls queued 2026-10-05.** Jade and Claude
agreed (2026-10-05) to prioritize this track.

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
- **Random-partition nulls** (first three pairs): 0.481, 0.489, 0.498 — the protocol does not leak.
- Pending: remaining nulls, lowest-MI pairs, cross-layer L0 × L2 pairs (+ nulls), private-model R-judge.
- **Queued: does it hold with per-head dropout?** (Jade.) 10M private models with head dropout 0 / 0.1 / 0.3 (no winner
  dropout): layer-0 single cells under both description protocols, and four random layer-0 pairs as intersections with
  random-partition nulls (`run_comp_dropout.sh`). Prior expectation, stated before the result: head dropout forces each
  head to be useful alone, so single cells may become *more* nameable and the intersection advantage *smaller*; if the
  advantage survives dropout, composition is not just an artefact of heads leaning on each other.

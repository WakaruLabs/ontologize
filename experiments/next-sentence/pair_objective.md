# A pair objective: slow heads from sentences of one document (sketch)

**Status:** design only, nothing is implemented.

This sketch follows the measurements in `notes.md` (2026-10-13, `transitions.py`) and the
identifiability reading in `scratch/literature/identifiability.md`. In the trained models, nothing
plays the role these theorems need:
- every head changes on 92–98% of adjacent pairs;
- no head is slow (κ ≤ 0.081);
- no head tracks the language.

The embeddings, though, share a lot within a document, *by direction*: cosine AUC 0.92, and 74% of the
adjacent similarity is still there 12 sentences apart. So the objective has to *create* heads that hold
still across a document, while every sentence still reconstructs itself.

## The objective

A batch is b pairs (x, x′) of sentences from one document, so 2b rows, each reconstructed from its own
code. Write p_{ℓh}(x) ∈ Δ^k for head h of layer ℓ's assignment: one-hot forward, softmax backward
(STE). S is a designated set of *slow* heads, s per layer.

L = Σ_rows L_stock(x) + s_pair · (1 / (b·|S|)) Σ_pairs Σ_{(ℓ,h) ∈ S} (1 − ⟨p_{ℓh}(x), p_{ℓh}(x′)⟩)

- **Stock loss.** `L_stock` is the stock loss: whitened MSE with deep supervision, plus the live
  regularizers. It includes `s_Hm`'s batch usage-entropy bonus, which is what stops a slow head from
  collapsing to one entry.
- **Pair term.** The second term is the share of slow heads that switch between x and x′, which is
  `transitions.py`'s "share of heads that switch" restricted to S.
- **Relation to SlowVAE.** Take a sticky transition prior per head: stay with probability 1 − ε,
  otherwise move uniformly. Its expected negative log is this term up to a constant, with s_pair
  standing for log((1 − ε)(k − 1)/ε). In head space, it is the discrete form of SlowVAE's sparse
  transition prior.
- **Heads outside S** carry no pair term and are free to code what changes between sentences.

**Why designate S rather than let a split emerge.**
- SlowVAE's guarantee is for continuous latents under an L1 transition penalty. There, a rotated code
  pays more than an axis-aligned one, so sparse transitions force the slow and fast factors onto
  separate latents.
- For one-hot heads, an expected-switch-count penalty has no such argument. Rare switches carry many
  bits each, so a penalty on all heads may just lower switching everywhere and keep the dense pattern.
- Designating S makes the slow/fast block split structural. The block split is also exactly what the
  literature says document pairs can identify: language, topic and register are nearly always shared
  within a document, so pairs separate the slow block from the fast one, not the factors inside it.
- A second arm puts the term on all heads, to see whether a split emerges. The κ distribution from
  `transitions.py` answers that: bimodal means a split, a shift of one mode means uniform slowing.

**Alternative pair term: per-head temporal mutual information.**
- The term: −Σ_{(ℓ,h) ∈ S} I(a_{ℓh}(x); a_{ℓh}(x′)), from the batch joint J = ½(P Q′ᵀ + Q P′ᵀ)/b.
- This is IIC (Ji, Henriques & Vedaldi 2019; from memory) applied to sentence pairs. It is also the
  discrete form of PCL's per-component pairing, whose logistic regression estimates exactly this log
  density ratio.
- It cannot collapse, since a constant head has MI 0, and it allows predictable transitions that
  aren't "stay" (Markov heads, below).
- The cost is estimation. With one-hot forward values, the plug-in MI from b pairs over k² cells is
  biased by roughly (k − 1)²/(2b) nats: 0.47 at k = 32 and b = 1024. So it needs soft assignments
  (recompute the logits in the probe and use softmax(logits/τ)), large batches, or both. Keep it as
  the second term to try, after the agreement term.

## Pairs

- **Default: two sentences of one document at segment distance ≥ 2.**
  - Doc-level structure dominates what pairs share: sentences 12 apart keep 74% of the adjacent
    cosine above random.
  - Distance ≥ 2 avoids the adjacent fragments of one capped sentence, which share sentence-level
    content.
- **Adjacent pairs are the comparison arm.** Use `d1_sent` (both full sentences), since 51% of
  adjacent pairs involve a line or a cap piece.
- **Same rows for both arms.** Draw partners from the training prefix (`--rows`, 4M by default), as
  `train_next.py` does, so the pair and control arms see the same rows.

## Model and heads

- **Model.** Start with `train_next.py`'s flat model: one layer of 380 heads at k = 32, with
  `--dict-init-scale 0.1 --s-hm 1e-4`. The target is the input, so the residual stack would also be
  well defined, but one layer keeps the head-level analysis free of the cascade.
- **|S|.** 38 heads (10%). Even at 5 bits a head, that is 190 bits per document, likely more than
  the document carries.
- **Redundancy is the expected failure.** Every slow head may encode the same easiest factor,
  language. The HSIC experiment's between-head dependence penalty (`hsic.pairwise_head_cka`,
  `s_hsic_heads`) restricted to S adds PCL's other assumption, independent components, and is
  already implemented.

## Implementation: the HSIC experiment's pattern, no change to `ontologize/`

- **`pairs.py`.**
  - `doc_partners(index, lo, hi, min_d, rng)`: one random partner per anchor, from the same document
    at segment distance ≥ `min_d`.
  - `PairSource(..., stacked=True)`: serves concat(x, x′) as one (2d,) record. Grain then batches
    (b, 2d), and `train()` treats it as an ordinary self-target batch.
- **`pairloss.py`.**
  - `pair_probe(module, X, ...)`: reshapes (b, 2d) to (2b, d), with the first sentences then their
    partners, and runs `hsic_hyperparams.probe_with_codes`. It returns the per-layer codes
    (l, 2b, h·k) in the ghost slot.
  - `PairHyperparams(DictWDHyperparams)`, so it composes with `--dict-init-scale`, with fields
    `s_pair`, `pair_term` (`agree` or `mi`), `pair_h` (heads per layer, to unflatten), `pair_heads`
    (|S| per layer, the first that many; 0 means all).
  - `init` is `super().init()` followed by `state.replace(apply_fn=pair_probe)`.
  - `loss` reshapes the target the same way, calls the stock loss, adds `s_pair` times the term, and
    records the raw term in column 2 (`MSE_ghost`, which is 0 with ghost off), as the HSIC loss does.
- **`train_pair.py`.**
  - `train_ste.parser()` plus `--pairs doc|adjacent`, `--min-d`, `--s-pair`, `--pair-term`,
    `--pair-heads`.
  - `train_ste.setup` picks its `Hyperparams` class from flags. It needs a small hook,
    `setup(cfg, src, hp=PairHyperparams, hp_extra={...})`, which asserts that `hp` subclasses the
    class the flags would have chosen.
- **Tests.**
  - Identical codes give a switch share of 0, and independent uniform codes give 1 − 1/k.
  - The reshape keeps row r paired with row r + b.
  - Stacked records batch as (b, 2d).
  - A short CPU training loop lowers the term with reconstruction intact.

## Evaluation

1. **Pilot sweep.** One seed, about 5k steps, s_pair on a log grid around the scale where the term's
   gradient matches the MSE's. The term starts near 0.97. Pick the smallest s_pair at which S becomes
   slow (held-out κ on `d2+` above 0.5) with held-out FVU_w within a few percent of the control.
2. **Arms, two seeds each, matched steps:**
   - control (s_pair = 0, same pair batches);
   - slow subset S;
   - optionally, all heads at a lower s_pair.
3. **On the held-out tail.**
   - `transitions.py`: κ by distance for S and for the rest, NMI with language, and the share of
     heads that switch.
   - FVU_w of each sentence against the control's.
4. **Across seeds** (`experiments/ste-arm/partition.py`):
   - matched NMI within S, within the fast heads, and for the control's heads (the flat arm reached
     0.018 on documents);
   - block-level agreement: how well seed B's slow codes are predicted from all of seed A's slow
     heads, against from A's fast heads.

   The literature predicts block-level agreement for S. Head-level agreement within S needs
   factors with different sharing patterns (next section).

## Extensions, in order

- **Multi-relation pairs.** Add pairs from *different* documents in the same language, with their own
  weight.
  - They share the language alone. Locatello et al.'s condition (two pairs' shared sets sometimes
    intersect in factor i alone) then holds for language.
  - That separates language from the rest of the document-level block, which document pairs alone
    cannot do.
  - Topic and register stay mixed until some relation shares one without the other.
- **Triples.** For discrete codes, Kruskal identifiability needs three conditionally independent
  views, and two never suffice. Serve three sentences of a document (positions spread, e.g. 1, 6 and
  11), reshape (b, 3d) to (3b, d), and apply the term over the three pairs.
- **Markov heads.** Replace the sticky prior with a learned k × k transition matrix per slow head,
  T_h, trained on consecutive sentences with the term −log T_h[a(x_t), a(x_{t+1})]. This is the
  discrete HMM that Allman et al.'s Theorem 6 identifies from windows of three, but it covers one
  chain; per-head identifiability in a factorial model is unproven.

## Risks

- **Collapse.** A slow head that settles on one entry satisfies the pair term. `s_Hm` was meant to
  guard against it; watch the per-head usage entropy (`transitions.py` reports it).
  - The pilot (`notes.md`, 2026-10-13) found that `s_Hm` at 1e-4 does not guard against it: every
    s_pair from 1e-4 up collapses the slow heads.
  - The term needs the slow heads' usage entropy inside it, as the mutual-information form has.
- **Batches drawn from b documents.** Batch statistics (`s_Hm`, mean entropy) see half as many
  documents as an i.i.d. batch of the same row count. The control arm shares the batches, so the
  comparison stays fair, but absolute numbers aren't comparable to the i.i.d. runs.
- **Theory as analogy.** Every guarantee above is for continuous latents, a single chain, or exact
  distribution matching. A positive result would be evidence, not a proof.

# partition-autointerp

Tests the paper's claim (findings §"Per-latent lenses mis-measure classifier
codes") directly: if a head's content lives in the partition it induces
rather than in per-entry magnitudes, then describing and blind-testing the
HEAD as one categorical variable should recover interpretability that
per-latent detection (autointerp.py) misses.

The unit of description becomes a head's set of entry decodes (mode
`entries`: parameter decodes through the model's own decoder + M2M100, as
in autointerp's `params` mode; mode `acts`: per-entry top-activating texts,
as in autointerp's `acts` mode), and blind detection is scored at the head
level: the judge assigns held-out snippets to entry values (or "none" for
in-partition distractors drawn from the head's non-described entries).
Scoring machinery (judge plumbing, top-k harvest merge, null pairing,
snippet handling, texts recovery) is reused from `autointerp.py`, not
reimplemented.

## Run (from the repository root)

```bash
# 1. harvest: per-entry top rows + entry decodes for a stratified head
#    sample (GPU JAX; restores the checkpoint)
uv run python experiments/partition-autointerp/headinterp.py harvest \
    --ckpt data/out/sonar/multilingual/resid_nc \
    --cache data/sonar_embeddings/mc4_4M.npy

# 2. texts: reuse autointerp.py's stage 2 verbatim; it only reads top_i /
#    neg_i from the npz, which heads.npz provides. Pointing --out at the
#    existing campaign dir appends only the missing rows to its texts.jsonl.
uv run python autointerp.py texts \
    --features experiments/partition-autointerp/out/heads.npz \
    --out data/out/sonar/autointerp \
    --cache data/sonar_embeddings/mc4_4M.npy

# 3. describe: entries mode (torch M2M100 decode + 1 LLM call/head for the
#    dimension name) and acts mode (1 LLM call/head). Needs ANTHROPIC_API_KEY
#    unless --dry-run.
uv run python experiments/partition-autointerp/headinterp.py describe --mode both

# 4. score: head-level assignment detection, with the donor-description
#    chance floor (--null), mirroring autointerp's null control
uv run python experiments/partition-autointerp/headinterp.py score --null

# 5. compare: join with an existing per-latent campaign's scores.csv
uv run python experiments/partition-autointerp/headinterp.py compare \
    --latent-scores data/out/sonar/autointerp/onto/scores.csv --latent-mode cacts
```

## Inputs

- `--ckpt`: Ontologizer checkpoint dir. Default
  `data/out/sonar/multilingual/resid_nc` (matches the existing autointerp
  campaign); `data/out/sonar/multilingual/resid_nc_hm` is the other live
  run. Checkpoints restore on GPU JAX only (same constraint as
  pareto/textfid).
- `--cache`: embedding cache from `encode_corpus.py`
  (default `data/sonar_embeddings/mc4_4M.npy`; the `.langs.npy` sidecar is
  used by the texts stage's drift check).
- `--texts`: dir containing `texts.jsonl` (default
  `data/out/sonar/autointerp`, the shared campaign dir).
- compare stage: an existing `autointerp.py score` output
  (`data/out/sonar/autointerp/onto/scores.csv`).

## Outputs (into `experiments/partition-autointerp/out/`)

- `heads.npz` + `meta.json`: harvest artifacts (heads, kept entries, usage,
  per-entry top rows, distractor rows, entry decode embeddings).
- `descriptions.jsonl`: `{head, mode, dimension, values}` records.
- `scores.csv`: `head, mode, kind(self|null), n_items, acc, macro_f1,
  bin_f1, dis_rej`. `bin_f1` (any-value vs none) is the column directly
  comparable to autointerp's per-latent F1; `acc` is the finer partition
  test. The claim predicts: head-level `acc`/`bin_f1` minus null clears
  zero by a margin, while the same heads' mean per-latent F1 minus null
  does not.
- `comparison.csv`: per head, head-level self/null scores next to the mean
  per-latent F1 of that head's autointerp-sampled entries (NaN where the
  stratified latent sample covered none of the head's entries), plus a
  printed Spearman between the two lenses.

## Cost

- harvest: one activation pass over `--rows` (default 262144) cache rows
  plus one decode of `n_heads * entries_per_head` (160) constant codes.
  ~5-15 min on the training GPU; not CPU-feasible (checkpoint restore).
- texts: network-bound mC4 re-stream up to the largest needed row
  (bounded by `--rows`); tens of minutes, CPU.
- describe: M2M100 decode of 160 embeddings (minutes, CPU ok with
  `--device cpu`) + ~20 LLM calls per mode.
- score: 20 heads x 2 modes x (self + null) = ~80 judge calls, each ~30
  snippets + 8 value labels (~2-3x autointerp's per-call size). Well under
  a dollar on haiku.

## Assumptions / design choices (flagged, not silently guessed)

- "Head activation" per entry is the tag probability `p` from the soft
  forward at `--temperature` (default 0.03), identical to autointerp's
  onto harvest (`onto_acts_fn`); feature id layout `f = (l_i*h + h_i)*k + k_i`.
- Only the `entries_per_head` most-used entries per head are described
  (default 8 of k=32): a 32-way judge prompt is unreliable and expensive.
  Distractors are drawn from rows whose argmax entry is one of the head's
  NON-described entries, so "0 = none" is a real partition cell, and
  rejecting distractors requires value-level (not topic-level) knowledge.
  This under-tests full-head exhaustiveness; raise `--entries-per-head` to
  trade prompt reliability for coverage.
- Truth for a positive snippet is "the entry whose top-activation list it
  came from", not a fresh argmax at score time; ranks used for scoring
  (`n_desc..n_desc+n_test`) are disjoint from ranks used for describing,
  matching autointerp's held-out-positives discipline.
- Head sample is stratified over layers, uniform within a layer. It is NOT
  stratified by usage entropy; sharp and diffuse heads land in the sample
  by chance (usage is recorded in heads.npz for post-hoc splits).
- The compare stage's latent-side coverage is whatever autointerp's
  stratified 256-latent sample happened to include per head (~1-2 entries
  of 32 on average); `comparison.csv` reports `n_latents_scored` so thin
  rows can be excluded.

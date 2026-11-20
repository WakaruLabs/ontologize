# An LCM as model organism (plan)

**Status:** plan (2026-11-18).
- Phase 0 is built and tested, and the float16 copy of the cache is written (2026-11-19).
- Phase 1 ran on the RTX 3090 (2026-11-20). Both gates pass, and the two seeds' states agree up to
  nearly a rotation in their high-variance directions.
- Phase 2 is next. The results are in `notes.md`, and `README.md` covers the code.

## The question

The suggestion (2026-11-18) is that a Large Concept Model (LCM), or a JEPA-like model, may be easier to
interpret than the embeddings it reads, because a model that learns dynamics over embeddings already
carries an inductive bias toward the variables that matter. Here "easier" means, first, that a
dictionary's heads reproduce across training seeds, the constraint the writeup calls binding. So:

> Do Ontologizer codes trained on a Base-LCM's hidden states reproduce across seeds better than codes
> trained on the SONAR embeddings that LCM reads, with the same recipe on the same sentences?

## What the record already says

- **Codes on the embeddings don't reproduce.** The recipe below (one layer of 380 heads at k = 32), on
  `mc4_sent16`, matches heads across seeds at NMI 0.016–0.017 against a 0.005 null
  (`experiments/next-sentence/notes.md`, 2026-10-15 and 2026-11-10). The flat arm on `mc4_4M`
  documents reads 0.018.
- **Dynamics in the objective help, as a block.**
  - The pair objective rewards 38 slow heads for keeping their entry across two sentences of one
    document. Those heads match at 0.41 across three seeds, and at 0.31 within a language, for 1.8–1.9%
    of FVU_w.
  - The match is block-level only: a slow head's best partner beats its runner-up by 0.003–0.004
    (2026-11-10).
  - The slow heads carry their own future far better than other heads: the embedding adds 27% to what
    a slow head's entry says about its next entry, against 71% for every other head (2026-11-14).
- **A model trained on dynamics is not enough by itself.** GPT-2 learns dynamics over tokens. Its seed
  pair agrees substantially only in layer 0, at under a fifth of a within-run control, and layers 2–4
  lose even layer membership (findings, "Replication on GPT-2"). So the claim under test is narrower:
  dynamics learned at the level of sentences, over a continuous embedding.
- **The theory motivates this but doesn't cover it** (Balestriero & LeCun 2026,
  `scratch/literature/platonic-jepa.md`).
  - A whitened JEPA with an identity predictor is identified up to a rotation (Proposition 1), and
    next-token prediction up to an invertible linear map (Proposition 4).
  - Pairs cut from one document share its nuisance, and the model spends directions on it
    (Proposition 3).
  - Neither a Base-LCM (fixed target, learned predictor) nor a JEPA used as a world model is covered.
  - What the results suggest by analogy: learned dynamics may fix a subspace but not a basis.
    Categorical heads cannot rotate, so the dictionary would have to supply the basis.
- **The nuisance here is language.** In `sfa.py` the top four slow directions are language, and
  language is 54% of what a document keeps over the top 38 (2026-11-03).
- **How predictable the next sentence is.** On the held-out pairs, `eval_next.py`'s FVU_w is 1.0001 for
  the training mean, 1.53 for copying x_t, and 0.896 for a ridge from x_t on 200,000 pairs
  (2026-10-13).

## Why build it here

- **No trained LCM is available.**
  - LCM team (2024, arXiv:2412.08821) trained their ablation models at 1.6B parameters. The Base-LCM
    there has 32 layers at d_model 2048 and 16 attention heads, with RoPE, RMSNorm pre-normalization,
    SwiGLU and dropout 0.1. It was trained for 250k steps of 229k concepts.
  - `facebookresearch/large_concept_model` (MIT, last pushed 2025-01-29) gives training recipes on a
    fairseq2 0.3.0 release candidate. Its README links no trained weights (checked 2026-11-18).
  - So we would train our own either way.
- **`mc4_sent16` is already an LCM training set.**
  - It was built the LCM's way: SaT Capped at 200 characters, and SONAR's masked-mean embedding without
    an L2 step, which is what the LCM reads.
  - It holds 44,113,823 rows from 3,866,644 documents, in document order, with up to 16 consecutive
    segments per document.
- **Most of the tooling exists.**
  - `train_pair.py`, `seedmatch.py` and `closure.py` take `--cache`, and they find the cache's
    sidecars by its stem (`<stem>.index.npy`).
  - So a harvest written in that layout, 1024 wide, needs no new trainer or base config.

## Phase 0: build

All of it is built (2026-11-19). `README.md` records where the code departs from this table.

| File | What it does |
|---|---|
| `lcm.py` | The Base-LCM in Flax: a robust scaler (per-dimension median and IQR, fitted on a sample of training rows; the paper's Eq. 4), PreNet, a causal decoder-only transformer and PostNet. Shapes are fixed: each document is padded to 16 rows, with a mask. `states(…, block)` returns a block's residual output |
| `train_lcm.py` | Trains on mc4_sent16 documents outside the scoring tail. The loss is the MSE between the PostNet output at position n and x_{n+1}, in SONAR space (Eqs. 5–6). Saves orbax checkpoints and logs held-out FVU_w as `eval_next.py` computes it |
| `harvest_lcm.py` | Runs a checkpoint, or an untrained init, over the harvest rows. Writes `data/activations/lcm_<run>_b<block>.npy` with these sidecars: `.index.npy`, `.docstart.npy` and `.langs.npy`, sliced from mc4_sent16's; `.mse_weights.npy` (inverse variance); a copy of the weights rescaled the way `gpt2_l8.matched_weights` does it, against the control's weights on mc4_sent16, so the recipe's loss weights transfer; and `meta.json`, which records the source rows |
| `stateagree.py` | Compares two LCMs' states on the same rows: linear CKA, the orthogonal Procrustes residual (each side scaled to unit total variance: identity up to rotation), the same after whitening each side, and ridge R² in each direction (both of these last two test an invertible linear map, since whitening absorbs one). Two untrained inits give the floor |
| `f16cache.py` | Writes the cache's float16 copy for the upload, links its sidecars under the copy's stem, and records in `meta.json` what rounding cost. `--bulk` puts the array on another disk behind a symlink |
| `tests/test_lcm_organism.py` | Checks causality (a state at n doesn't change with rows after n), no leakage from padding or other documents, the scaler's round trip, that the LCM learns a planted dynamic that x_t alone can't predict, stop and resume, and that the harvest's sidecars stay aligned row for row with the cache |

**Choices, and why:**
- **JAX, not the LCM repository.** JAX keeps the project's stack and fixed shapes, avoids the
  fairseq2 release candidate, and lets harvesting run in the same process.
- **Base-LCM first.** Its states are deterministic functions of the prefix.
  - A diffusion LCM generates better. The paper finds that Base-LCM averages plausible
    continuations, which gives it lower contrastive accuracy and mutual information.
  - But a diffusion denoiser's states depend on the noise step, so what to harvest would be a design
    question of its own. A Two-Tower LCM's contextualizer is the alternative if this one falls short.
- **Width 1024.** It matches SONAR, so harvests drop into the tools above.
- **Size.**
  - 8 blocks at d_model 1024, with 16 heads and SwiGLU inner width 2816: about 105M parameters, against
    the paper's 1.6B.
  - Gate A decides whether it grows.
- **No end-of-text sentence.** The paper appends one for generation, and we don't generate.
- **Context.** A row's context is its own run of at most 16 segments, not the whole document. Runs
  skip unusable segments.
- **Rows.**
  - The LCM trains on every document before the last 262,144 rows, snapped to whole documents.
  - Harvests take two ranges:
    - the rows below 4.4M (snapped), where the dictionaries train (below 4M) and `closure.py` fits
      (4.0M–4.4M);
    - the last 262,144 rows, where everything is scored: the 32,768-row tail, and the full range for
      within-language NMI.
  - The prefix keeps its row numbers, so `--rows 4000000` and `--fit-rows 4000000:4400000` mean what
    they did.
- **Layer.** Block 4 of 8 first, then block 8 (just before PostNet).
- **Storage precision: float16,** for the cache upload and the harvests. A harvest is then 9.5 GB,
  against 19 GB at float32.
  - **Measured on the 32,768-row tail (2026-11-18):**
    - Rounding to float16 moves a row by 2.1e-4 of its norm (median).
    - It costs a whitened FVU of 2e-7 under `mc4_sent16.mse_weights.npy`.
    - All 128 decodes tested came back identical. The NLL of the float32 decode rose by 7e-6 nats on
      average.
  - **bfloat16 is ten times coarser.** In one entry it reaches 17% of the standard deviation of
    dimension 837, the near-constant dimension that the `no837` weights drop.
  - **The readers (checked 2026-11-19).** `StackedPairSource`, which `train_pair.py` trains from, and
    `seedmatch.py` convert rows to float32 as they read them, so both take the float16 copy as it is.
    `NpyDataSource`, the stock trainer's source, returns rows in the dtype they're stored in, so a run
    through it would get float16 batches.
  - **The earlier precision failure was a different one.** SONAR itself was run in bfloat16 (see the
    comment in `sonar.py`: "reducing precision breaks the embeddings"). SONAR runs only where the
    cache was built, at float32.
- **Compute precision.**
  - **The risk.** On a TPU, JAX's default matmul precision for float32 is a single bfloat16 pass,
    coarser than the GPU's TF32 default. Hard codes are sensitive to it: on 12 rows, 226 of the hard
    stack's 4,560 codes changed between the GPU's default precision and float32
    (`experiments/ste-arm/notes.md`, 2026-10-09).
  - **The remedy.** Train with `jax_default_matmul_precision` at `high` and score at `highest`.
    Retrain the SONAR control on the TPU (Phase 2), so that every arm shares hardware and precision.

## Phase 1: the LCM and its gates

1. **Train seeds 42 and 43** for one epoch. Before that, run a 1M-row pilot to measure throughput.
2. **Gate A: has it learned dynamics?**
   - Compare its held-out next-sentence FVU_w with the ridge from x_t (0.896), and with a new ridge from
     x_t plus the mean of the run's earlier rows, a cheap linear context baseline.
   - If it doesn't beat the context baseline by more than its seed spread, grow the model or train
     longer before going on.
   - The baselines, measured on the 26,335 held-out pairs (2026-11-19, `notes.md`): the ridge from x_t
     reads 0.898 and the context ridge 0.870. So 0.870 is the bar.
3. **Gate B: is the state more than its input?**
   - Fit a ridge to the block's state from x_n alone, and from x_n together with the prefix mean.
   - If x_n alone explains nearly all of it, a dictionary on the state is a dictionary on SONAR with
     extra steps, so move to a later block.
   - `stateinput.py` (2026-11-19) adds a third reading. It is the ridge from the model's own state for
     x_n read alone, as a one-row document. The two ridges from x_n are linear, so they would count
     whatever the block computes nonlinearly from x_n as context. The `alone` reading counts all of
     that as x_n's.
4. **Is the representation identified before any dictionary?** Run `stateagree.py` between the two
   seeds, against two untrained inits. Read loosely from the theory, ridge R² and whitened Procrustes
   (the linear-map tests) should be high, with plain Procrustes (the rotation test) worse.

## Phase 2: dictionaries on the states (the main test)

Every arm is `train_pair.py` at `--s-pair 0`. That is the control exactly as `long.sh` trained
`doc2_sp0_ph38`: three epochs, to step 46,250. Here `--cache` and `--mse-weights` point at a harvest.

| Arm | Input | Seeds |
|---|---|---|
| LCM | LCM seed 42, block 4 | 42, 43 (and 44 if the result is positive) |
| untrained | the untrained init, block 4 | 42, 43 |
| SONAR | `mc4_sent16`, retrained on the TPU. The GPU runs `doc2_sp0_ph38`, `_s43` and `_s44` check that the retrain matches | 42, 43, 44 |

**Readouts.**
- From `seedmatch.py --cache <harvest> --labels <harvest>.langs.npy`, with `--c` for three seeds:
  - matched NMI over all 380 heads, against its null;
  - whole-model assignment;
  - within-language NMI;
  - best partner over runner-up;
  - the cycle share.
- Held-out FVU_w.
- `closure.py` on each arm's heads.

**Predictions, stated before running.**
- **The untrained arm.** It should read close to the SONAR arm (0.017). If it reads higher, context
  mixing alone makes heads agree, and that becomes the bar the LCM arm must clear.
- **The LCM arm.** It reads above both: this is the claim under test.
- **If it does, the agreement is block-level,** as for the slow heads, with the margin near 0.003. A
  margin well above that would be the stronger result: heads identified one by one.
- **Language.** Much of the agreement will be language, so the within-language score is the one to
  read.
- **What would count against.** The LCM, untrained and SONAR arms read alike. Together with GPT-2, that
  would say learned dynamics in the model don't carry over to the dictionary at this scale.

**Across models.** A further test is a dictionary on LCM seed 42's states against one on seed 43's,
on the same rows. It asks whether the variables belong to the model class rather than to one model.
`seedmatch.py` takes one `--cache`, so this needs a `--cache-b`. The first reading doesn't depend on
it.

## Phase 3: if Phase 2 is informative

- **The 2×2.** The Fano pair objective on LCM states (`--s-pair 1e-3`). Does the model's bias replace
  the objective's, add to it, or do nothing?
- **A JEPA arm.**
  - The model: a per-sentence encoder f (an MLP on SONAR) and a causal predictor over f(x_≤n).
  - The loss: ‖pred_n − f(x_{n+1})‖², with an anti-collapse term (whitening, as in Proposition 1, or
    the SIGReg term in Balestriero & LeCun's runs).
  - What gets interpreted: f's output or the predictor's states.
  - Why it matters: the representation is learned rather than fixed to SONAR. That is the suggestion
    in its strongest form, and the case the theorems come closest to.
- **Good views.** Pair each sentence with a translation of itself (decode x_t into another language,
  then re-encode; `platonic-jepa.md`, item 1), for a model that should carry no language.
- **Other blocks.**
- **Steering through the context** (the LCM entry in `scratch/literature.md`). Swap a head's entry in
  one context row and see whether the LCM's predicted continuation moves, decoded through `textfid.py`'s
  `SonarDecoder`.

## Cost and constraints

Both the LCM and the dictionaries train on Cloud TPU, in a project of their own, through the harness
in `experiments/cloud` (its README). The cache and every output sit in a bucket in the node's region,
so nothing large comes back.

**The machine.** A v5e-4 in us-west4-a, under Flex-start. As of 2026-11-18 that is the only zone
offering Flex-start for v5e.
- **Memory.** Its four chips share 192 GB of host RAM, enough to hold the float16 cache in a tmpfs
  (`ONTO_RAM_GB`). A one-chip host has 48 GB.
- **Parallel runs.** Four dictionary runs go at once, one per chip.
- **The time limit.** A Flex-start node is deleted when its requested time is up, so the runner's
  bucket sync is what keeps the work.

**Prices.** These are Google's list prices on 2026-11-18, for v5e in us-west4:
- $1.20 per chip-hour on demand, or $0.60 under DWS Flex-start;
- Standard storage at $0.020 per GiB-month;
- reads from a VM in the same region are free;
- downloads to the internet cost $0.12 per GiB.

| Item | Estimate | On demand | Flex-start |
|---|---|---|---|
| LCM: the pilot, two seeds × 3 epochs, and the harvests | 10–24 chip-hours | $12–29 | $6–14 |
| Dictionaries: 8 runs (LCM ×3, untrained ×2, SONAR ×3) | 8–12 chip-hours | $10–15 | $5–8 |
| Evaluations at `highest` precision | 1–3 chip-hours | $1–4 | $1–2 |
| Storage: the cache (84 GiB as float16), the harvests (27 GiB) and checkpoints | about 115 GiB | $2.30 per month | |
| Egress: results and decode slices | under 1 GiB | under $0.12 | |

That comes to about $25–50 on demand, or $15–30 under Flex-start, plus storage.

- **The basis for the compute figures.**
  - **The LCM** needs about 2.8e16 FLOPs per epoch (6 × 105M parameters × 44M rows). That is roughly
    ten minutes on one v5e chip at 30% of its bfloat16 peak, which is a guess. Setup, the pilot,
    harvests and the time the TPU sits up but idle cost more than the epochs themselves.
  - **Measured on the 3090 (2026-11-19, `notes.md`):** 23 minutes per epoch, at TF32 and 27 TFLOP/s
    counting padding. The guess above leaves out the three bfloat16 passes of precision `high`, and
    the 26% of each batch that is padding.
  - **Each dictionary run** is costed at the 3090's figure, about an hour: seed 43's two arms took
    2¼ hours for three epochs, evaluations included (2026-10-15). `high` precision makes matmuls
    slower than the default.
- **Growing the LCM.** At the paper's 1.6B parameters it needs about 12 more chip-hours, and four or
    more chips for memory.
- **Before starting.**
  - The new project needs billing, the TPU API, and preemptible v5e quota in us-west4. The laptop's
    gcloud needs updating for Flex-start (`gcloud components update`).
  - The float16 copy of the cache exists (2026-11-19, `notes.md`). The harvests' float16 path
    doesn't yet.
  - The upload is 84 GiB, about 2 hours at 100 Mbit/s.
  - The code is written to be TPU-compatible. Budget a short shakedown run anyway.
- **Text evaluations** (`textfid.py`, `steerfid.py`) run SONAR's decoder in torch. They stay on the
  local GPU, on small slices.

## Defaults to confirm

- Base-LCM before the JEPA arm.
- About 105M parameters to start.
- A v5e-4 in us-west4-a under Flex-start, with the bucket in us-west4.
- float16 storage for the cache and the harvests.

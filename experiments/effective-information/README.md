# effective-information: how much does each head's value matter downstream?

The writeup's formal reading (`writeup/sections/discussion.tex`, "heads
as causal variables") treats each head as a categorical causal variable,
`withArgs` set-interventions as the do-operator, and the residual stack
as the causal graph, in which every earlier head is a parent of every
later one. It measures identifiability (seed matching), realization
(steering) and composition (`compose.py`), but nothing weighs the edges
of that graph. Effective information does (Hoel, Albantakis & Tononi
2013): intervene on a source head uniformly over its k entries and
measure the mutual information between the forced entry and an effect
head's choice,

    EI = H(mean_j W_j) - mean_j H(W_j) = determinism - degeneracy,

where W_j is the effect head's entry distribution under do(source = j),
pooled over held-out contexts. Determinism is low when entry j's effect
depends on the context; degeneracy is high when different entries have
the same effect. EI is at most log2 k bits.

It is the interventional counterpart of realized bits (appendix,
"Realized bits, and what they bound"): realized bits measure how evenly a
head uses its entries, EI how much the entry it is forced to changes what
happens downstream.

## Why it is exact here

Heads are categorical, so the uniform intervention is enumerable: k
forward passes per head, no binning of continuous activations (the
obstacle for EI on ordinary networks). The effect variables, downstream
heads' entries, are categorical too, so each W_j is a k-bin histogram.
Every W_j is measured on the same contexts, so a pair the intervention
never changes has identical histograms for every j and EI exactly 0;
there is no finite-sample floor under a null effect, and positive EI
means some context's effect choice depends on j. Finite contexts bias
only the part of the effect that does change, and a split-half
correlation is reported (it can include bias that is systematic across
halves, since both halves share the effect heads).

## Files

- `effinfo.py` -- the measurement. `assignments` is `withArgs`'s
  per-layer loop with every layer's classification kept (tested equal to
  `withArgs` in `tests/test_effinfo.py`). Also records each head pair's
  observational mutual information on the natural forward pass (which
  includes the shared input as a common cause, which EI excludes), each
  head's realized bits, and its natural sharpness.
- `notes.md` -- results.

## Usage

    uv run python experiments/effective-information/effinfo.py \
        --ckpt data/out/sonar/multilingual/bl_ctl_full

Defaults: 4096 contexts from the cache's held-out tail (the `sae.py`
split), two halves for reliability; observational statistics over the
whole 32768-row tail; temperature read from the run's `log.jsonl`
(`temperature_end`, else the constant `temperature`). Outputs go to
`<ckpt>/effinfo`: `ei.npy`, `determinism.npy`, `degeneracy.npy`,
`ei_a.npy`, `ei_b.npy`, `ei_live.npy` (each `(l, h, l, h)`, [source
layer, source head, effect layer, effect head], NaN where the effect is
not downstream), `mi_obs.npy`, `heads.csv`, `summary.json`.

## Caveats

- **The intervention distribution matters, and one-hot entries are only
  on-distribution for a hard code.** A softmax head with near-uniform
  assignments never takes a one-hot value; forcing one swaps its averaged
  atom for a single atom, which can swamp the residual the next layer
  classifies. EI is then a response to off-distribution perturbations.
  This is the dependence on the intervention distribution that critiques
  of EI centre on (Eberhardt & Lee 2022). Read each source head's EI
  with its natural sharpness (mean max assignment: 1 for a hard code,
  1/k at uniform), and prefer straight-through (hard-code) checkpoints,
  whose natural states are the one-hot entries being forced.
- **Even on a hard code, uniform over all k entries forces dead entries.**
  A head that naturally takes 3 of its 32 entries spends 29 of its 32
  interventions on entries it never uses. `ei_live.npy` repeats the
  measurement with the intervention uniform over the head's live entries
  (argmax usage >= `--live-min` on the natural tail), from the same
  counts; on a hard code that is the on-distribution number. Its ceiling
  is log2 of the number of live entries.
- **Pairwise, not joint.** EI is computed per (source, effect) pair. The
  EI of a source on all downstream heads jointly is at most log2 k, so
  sums of pairwise EI over effect heads count shared influence many
  times; heads.csv's `ei_sum` is a summary, not an information quantity.
- **The last layer has no downstream heads**, so its EI is undefined with
  this effect variable.
- EI measures how distinguishable and context-robust a head's effects
  are, not whether the head is reproducible across seeds; the writeup
  measures that separately.

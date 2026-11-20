# nonlinear-steering: how dictionary configurations cope when the network uses a direction non-linearly

Hewitt (2026), "Do Not Rely on the Linear Representation Hypothesis", builds two counterexamples to the
tenets that a feature's intensity is the scale of its direction and that features compose by addition.
On Qwen3 he trains:
- **one direction whose scale picks the task**: a single vector v with four learned scales, where each
  scale alone steers a different task;
- **a sum that steers something else**: two vectors that each steer one task, whose sum steers a third.

The code is at https://github.com/john-hewitt/odd-steering-expts. It carries no licence, so this
experiment reimplements the method from its description.

Those constructions are ground truth for a question about this project's dictionaries: when the network
uses a direction non-linearly, does a configuration's reconstruction keep the behaviour (functional
robustness), and does its code describe what the network does (interpretive robustness)? Each
configuration builds in assumptions that bear on this:
- linear SAEs read intensity as a coefficient and compose additively;
- categorical heads quantize intensity into entries;
- gain-shape makes the code blind to the overall scale of a layer's input.

## Site

GPT-2 small at `blocks.8.resid_post`, the output of `transformer.h[8]`. These are the rows
`encode_acts.py` harvested into `data/activations/gpt2_l8.npy`, which every `gpt2_l8` Ontologizer and SAE
trained on.
- **BOS:** position 0 is the prepended BOS, which the harvest drops, so edits apply at positions ≥ 1 only.
  No metric scores position 0's prediction, which an edit there cannot reach.
- **Parity:** `common.parity` checks that the hook reads the harvested stream, by comparing its output on
  held-out windows with the stored rows.
- **Held-out data:** windows are the full-length documents in the cache's held-out tail, as stored in
  `gpt2_l8.docs.jsonl` (token ids, BOS first).

## Files

| File | What it does |
|---|---|
| `common.py` | GPT-2 in float32, `site_edit` (a hook replacing the site's output at positions ≥ 1), `capture`, the held-out windows, the parity check, `orth_step` (a step of a given length orthogonal to the activation) and `next_token_stats` (KL, cross-entropies, top-1 agreement) |
| `rescale.py` | The scale a gain-shape code can't see. It rescales the site's output by c, and compares GPT-2's next-token change with orthogonal steps of the same length: one random, one toward another held-out activation. Edits apply at every position ≥ 1 or at alternate positions. It then encodes the same edits with Ontologizer checkpoints (share of heads whose winner changes) and SAEs (Jaccard distance between active sets, relative code change), and reports the natural spread of the residual's norm. A rescale by 1 + 1e-6 marks the float32 floor: an exactly invariant code still flips near-tied decisions under rounding, and power-of-two scales flip none |
| `tasks.py` | Text for four behaviours a 124M base model can show, as BOS + T-token windows from disjoint train and test documents: `french` (mC4, from `mc4_sent16.sentences.jsonl`), `code` (Python source from the installed packages), `recipes` and `sports` (harvest documents by keyword), and `neutral` text for crosstalk |
| `pilot.py` | Hewitt's objective at the site: the NLL of a task's text under its steer, plus 0.2 times the KL that steer causes on each other task's text. It trains the shared direction with four softplus scales (initialized at 1, 3, 5, 7), the composition pair (v1 → `french`, v2 → `recipes`, v1 + v2 → `code`), and four independent single-task vectors as the linear control, all zero-initialized, with Adam at 0.02 for 1,200 steps. It scores held-out efficacy (the drop in NLL on a task's test windows) and crosstalk (the KL on the other tasks' windows), and applies the pass criteria fixed in `verdict` |

The tests are in `tests/test_nonlinear_steering.py`.

## Run

```bash
uv run python experiments/nonlinear-steering/rescale.py          # ~10 min on a GPU
uv run python experiments/nonlinear-steering/pilot.py --seed 0   # 1,200 steps
```

Outputs go to `data/out/gpt2_l8/nonlinear_steering/{rescale,pilot/seed<k>}`: CSVs, `norms.json`,
`metrics.json`, `vectors.npz` and `loss.csv`.

## Design notes

- **Teacher-forced scoring.** Hewitt scores behaviour with an LLM judge, fastText language ID and a
  docstring regex on chat models. A 124M base model has no chat behaviour to judge, so efficacy and
  crosstalk here are teacher-forced: the same NLL and KL quantities as the objective, on held-out windows.
- **Steers start after BOS.** The BOS prompt is the only unsteered position, so the steer has to create the
  behaviour. The first text token's prediction comes from BOS and is never scored.
- **Equal-length controls.** In `rescale.py`, the orthogonal steps have the length of the rescale they
  accompany (`|c - 1| |h|`), so the comparison is between changing the residual's size and changing its
  direction by the same amount. Equal length does not mean equally natural: the data-shaped step moves toward
  real activations and the random one does not.
- **What comes next.** If the pilot shows GPT-2 can host the constructions, the robustness measurements
  follow. Each configuration's reconstruction gets spliced in place of the steered activation (as
  `experiments/gpt2/common.py`'s `Splicer` does) and efficacy is re-scored. Code distances are compared with
  behaviour distances across the steering conditions, against the linear control.

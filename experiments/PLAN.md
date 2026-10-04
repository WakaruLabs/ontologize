# The experiment program

Twelve experiments, built as additive harnesses under `experiments/`.
Nothing here modifies the `ontologize/` package, the root scripts, the
tests, or the paper; every script imports the repo's real APIs and
parameterizes checkpoint and cache paths (defaults documented per
README, keyed to `data/out/sonar/multilingual/resid_nc_hm` and
`resid_nc`). The tree ships no data or checkpoints, so each README
states honestly what a run needs.

## Run order

The order below respects the dependency chain; each stage's outputs
feed the next.

| # | experiment | why this order | status |
|---|---|---|---|
| 1 | `verify-mse-weights/` | Load-bearing for everything that trains or evaluates; CPU-only, minutes. Ground-truth the reconstructed recipe first. | ready (CPU) |
| 2 | `layer4-freeze/` (`freeze_diag.py`) | Diagnostic pass over the existing reference run; quantifies the frozen-head caveat every table carries. Doubles as the head-health eval for later arms. | needs GPU, minutes |
| 3 | `fresh-eval/` | Builds the held-out cache and re-runs the Pareto; closes the disclosed train/eval-tail asymmetry. The fresh cache is also the honest eval set for everything after. | needs GPU + hours of stream skip |
| 4 | `steering-overlay/` | The decisive comparison: classification-forcing vs difference-of-means and probe baselines at matched budgets. | needs GPU, ~30 min |
| 5 | `task-naturalness/` | Scores the steered decodes from stage 4 (or any steers.jsonl): chrF content preservation, NLL fluency proxy, blinded human rating sheet. | needs GPU, minutes |
| 6 | `seed-stability/` | Second-seed replica plus partition-level matching; produces the Ontologizer number to sit beside the SAEs' 0.31/0.24/0.15/0.03. | needs GPU, ~a day of training |
| 7 | `partition-autointerp/` | Tests the paper's own claim that per-latent lenses mis-measure classifier codes; reuses the autointerp campaign machinery at head level. | needs GPU + API key |
| 8 | `devinterp-tracking/` | Checkpoint sweep of head structure; do classifications sharpen at stage boundaries? | needs GPU, ~30 min |
| 9 | `layer4-freeze/` (`retrain_variants.py`) | Three single-knob mitigation retrains vs baseline; run after the diagnostic says which knob to prioritize. | needs GPU, ~a day per variant |
| 10 | `hsic-bottleneck/` | HSIC/CKA loss prototype (self-test passed on CPU) + three-arm ablation. Note: the head penalty rewards independence but does not punish freezing; keep `s_Hm`. | needs GPU per arm |
| 11 | `scaling-sweep/` | Seven-point (h, k, l) star around the live config; tests polysemanticity-vs-parameters. Configs and queue.sh are pre-generated. | needs GPU-days |
| 12 | `rl-classifications/` | Exploratory: reward = tag probability, REINFORCE through the decode cycle; the embed tier is the runnable sanity ceiling. | needs GPU; partly design |
| 13 | `transcoder/` | Design-first (`DESIGN.md`): what must change for x-to-y mapping, why `forward="resid"` leaks the target, plus a labels-mode harness and ridge baseline. | needs author input |
| 14 | `dictenc-toy/` | A single `DictEnc` on a planted factorial code with exact recovery metrics; the only place the code's hardness and head/factor identity have ground truth. Pilot: softmax stays soft under the anneal, `ste` recovers the code. | ready (CPU, 30 s/arm) |
| 15 | `ste-arm/` | Hard-code (straight-through) arm at raised head count, with the calibration the live schedule does not transport: relative logit noise, a surrogate temperature matched to the logit scale, a rescaled `s_Hm`. Motivated by the live model's argmax reconstructing at whitened FVU 8.7e6 and by the 800-bit rate floor of 0.226. | needs GPU, ~a day |

## Reading the results back into the paper

Stages 3-6 retire the two disclosed caveats and fill the missing
numbers in Preliminary results (the steering-overlay and seed-stability
readouts are the ones the proposal calls decisive). Stages 7-8 are new
sections if they hold up. Stages 9-11 revise the reference model and
the Concerns section. Numbers flow from each experiment's CSV outputs;
the working notes remain the canonical record.

## Judgment calls to confirm with the author

Each README flags its own assumptions; the ones worth a human glance
before burning GPU:
- steering-overlay uses language as the shared steering concept (the
  only labeled concept the pipeline ships).
- scaling-sweep trains 4 epochs per point, not the live 24.
- layer4-freeze mitigation magnitudes are the harness author's; the
  directions come from the training code's own comments.
- hsic-bottleneck rides the (inert, ghost=False) ghost slot to carry
  per-layer classifications without touching the package.
- rl-classifications' REINFORCE hyperparameters are first guesses.
- ste-arm's surrogate temperature is hand-matched to a logit scale
  measured once; the principled version normalizes the logits and
  needs a package change.

# transcoder: Ontologizer as a map between representation spaces

Adapting the Ontologizer from reconstruction (`y = x`) to transcoding
(`x` from one embedding space, target `y` from another). This
experiment is **mostly design**: the stack turns out to be pair-ready
except for a handful of precisely-locatable seams, and the interesting
decision (what replaces residual forwarding when the target is not the
input) is architectural, not mechanical.

## Files

- `DESIGN.md` -- the audit and specification. Read this first. It pins
  down, file by file: what already supports transcoding (`d_in`/`d_out`
  are already separate through `Ontologizer`, `Hyperparams`, and the
  jitted `update` step), the single autoencoder hardcode
  (`ontostate.train`'s `Y = X`), the loader pairing
  (`PairedNpyDataSource` + `srctype="embedding_pair"`), the target-dim
  and `mse_weights` implications, and why `forward="resid"` must become
  a new inference-safe `forward="input"` mode rather than being patched
  with the target (target leakage).
- `transcode_train.py` -- the thin harness that is honestly buildable
  today without touching the package: a row-aligned paired `.npy`
  source, a training loop that mirrors `ontostate.train` but feeds
  `(x, y)` pairs into the **stock** jitted `update`/`schedules`/
  checkpointing, `forward="labels"` only, plus a closed-form ridge
  baseline `x -> y` and cache-tail FVU for both.

## Run

From the repo root:

```bash
uv run python experiments/transcoder/transcode_train.py \
    --cache-x data/sonar_embeddings/mc4_4M.npy \
    --cache-y /path/to/target_space.npy \
    --mse-weights /path/to/target_mse_weights.npy

# evaluate an existing run (+ ridge floor) without training
uv run python experiments/transcoder/transcode_train.py \
    --cache-x ... --cache-y ... --eval-only
```

## Inputs

| input | default | notes |
|---|---|---|
| `--cache-x` | (required) | input-space cache; e.g. `data/sonar_embeddings/mc4_4M.npy` |
| `--cache-y` | (required) | target-space cache, **row-aligned** with `--cache-x` (same corpus, same order; two `encode_corpus.py` passes). The harness can only verify lengths -- the alignment contract is yours. |
| `--mse-weights` | `''` (unweighted) | must be computed over the **target** cache (`make_mse_weights.py cache_y.npy out_y.npy`); the shipped SONAR weights are wrong for any other target space |
| `--out` | `experiments/transcoder/runs/transcoder` | standard run dir |

Model/optimization defaults mirror sonar.py (e_dec=2048, k=h=32, l=5,
b=256, lr=5e-5, temperature 1.0 -> 0.03 over 50k steps, winner-dropout
ramp, sd_K anneal). `--deepsup` is available (per-prefix losses work
under labels forwarding); `ghost` is fixed off.

**There is no paired target cache in the tree today** -- producing one
(e.g. a second encoder over the mC4 corpus via `encode_corpus.py`) is a
prerequisite for actually running this.

## Outputs

- a standard run dir under `--out`: orbax checkpoints, `loss.csv`
  (stock 9-column layout), `log.jsonl`; resumable.
- stdout at the end: whitened (or raw) FVU on the cache tail for the
  ridge baseline and the trained transcoder, with an explicit warning
  when the transcoder fails to beat the linear floor.

## Known limitations (by design; see DESIGN.md)

- `forward="labels"` only. It carries the documented starvation risk
  under hard temperatures (sonar.py: layers 2-4 can receive constant
  input); watch upper-layer head health with
  `experiments/layer4-freeze/freeze_diag.py`, which works on any run
  dir. The proper fix (`forward="input"`) requires the package changes
  specified in DESIGN.md 2d.
- No ghost path (inert under the bilinear config anyway).
- SONAR-decoding evals (`textfid.py`, `steerfid.py`, `decode.py`) only
  apply when the *target* space is SONAR.

## Cost (honest estimate)

- Harness training: same order as a sonar.py run for the same cache
  size and epochs -- for 4M pairs, b=256, 24 epochs (~375k steps),
  **many hours to ~a day of GPU**; scale `--epochs` down for pilots
  (1 epoch = ~15.6k steps, well under an hour on GPU). Not
  CPU-feasible at full scale.
- Ridge baseline + FVU eval: one 65k-row normal-equation solve
  (d_x=1024: seconds, ~1.5 GB RAM in float64) plus a 32k-row forward
  pass -- **minutes on CPU, trivial on GPU** (`--eval-only`).
- Prerequisite target cache: one `encode_corpus.py`-style pass with the
  second encoder over the corpus -- for 4M sentences expect **hours of
  GPU**, dominated by the frozen encoder, same as the original cache
  build.

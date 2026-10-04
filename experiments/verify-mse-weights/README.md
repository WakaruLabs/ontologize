# verify-mse-weights: reconstruction check for make_mse_weights.py

`make_mse_weights.py` (repo root) reconstructs the per-dim
inverse-variance whitening weights that `sonar.py` and the eval suite
load from `data/out/sonar/mse_weights.npy`. Its docstring warns to
**verify against the original before regenerating on top of an existing
run**, because a cache trained with one weight vector must be evaluated
with the same one -- and `sonar.py` records that the shipped vector was
computed from the *step-97600 census corpus embeddings*, which need not
be byte-identical to the current `mc4_4M.npy` cache. This experiment is
that verification, made runnable.

`verify_mse_weights.py` recomputes a fresh weight vector by **shelling
out to the repo's own `make_mse_weights.py`** (so the recipe cannot
drift from the shipped one), then compares fresh vs original and prints
a clear verdict.

## Run

From the repo root:

```bash
# full verify: recompute from the cache, compare against the shipped vector
uv run python experiments/verify-mse-weights/verify_mse_weights.py

# non-default paths
uv run python experiments/verify-mse-weights/verify_mse_weights.py \
    --cache data/sonar_embeddings/mc4_4M.npy \
    --original data/out/sonar/mse_weights.npy

# compare two existing vectors, no recompute
uv run python experiments/verify-mse-weights/verify_mse_weights.py \
    --fresh /path/to/candidate.npy
```

## Inputs

| input | default | notes |
|---|---|---|
| `--cache` | `data/sonar_embeddings/mc4_4M.npy` | cache to recompute from (relative paths resolve against the repo root) |
| `--original` | `data/out/sonar/mse_weights.npy` | the vector runs actually used |
| `--fresh` | none | existing candidate `.npy`; skips the recompute |
| `--pass-cos` / `--pass-maxrel` | `0.9999` / `0.01` | verdict thresholds |

## Outputs (into `--out-dir`, default this directory)

- `mse_weights_fresh.npy` -- the recomputed vector (only when not using
  `--fresh`). Written here, **not** over the original.
- `verify_scatter.csv` -- per-dim `dim, w_orig, w_fresh, rel_dev`
  (1024 rows for the SONAR runs), for scatter-plotting fresh vs orig.
- `verify_summary.json` -- all statistics plus the verdict.
- stdout: cosine, max/p50/p99 relative deviation, mean ratio, the worst
  dims, and one of:
  - **PASS** -- recipe over this cache reproduces the original;
    regenerating is safe.
  - **MARGINAL** -- globally aligned, a few dims out of tolerance;
    inspect the worst dims first.
  - **FAIL** (exit code 1) -- materially different. Expected cause:
    the original came from a different corpus snapshot; the original
    stays authoritative for all existing runs, do not overwrite it.

## Cost (honest estimate)

CPU-only, no JAX, no GPU. The recompute is one streaming pass over the
cache with `make_mse_weights.py`'s 100k-row chunking: for the 4M x 1024
float32 cache (~16 GB) expect **roughly 5-15 minutes** dominated by
disk reads, with **< 2 GB RAM** (one float64 chunk at a time). The
comparison itself is instant. With `--fresh` the whole script runs in
seconds.

# fresh-eval

Closes pareto.py's stated caveat: the mc4_4M cache tail is held out only
for the SAEs — the Ontologizer runs trained on the full cache, so the
published Pareto favors the Ontologizer end. Two stages:

1. `encode_fresh.py` replays encode_corpus.py's deterministic mC4 stream,
   skips the whole training corpus (the training cache's row count,
   3,974,400 for mc4_4M), and encodes the NEXT `--n` samples with the
   identical machinery (same tokenizer, the training cache's
   source-language tags, encoder dtype, pooling) into a fresh `.npy` +
   `.langs.npy` cache no model has seen. mc4_4M ends where its smallest
   language ran out, so this stream interleaves with
   `stopping_strategy="all_exhausted_without_replacement"`: the same order
   (mc4_4M is its exact prefix), continuing over the languages that still
   have documents.
2. `pareto_fresh.py` re-runs the reconstruction Pareto (importing
   `pareto.sae_points` / `pareto.onto_points` unmodified) scoring every
   point on the fresh rows, with the E[p] origin still measured on
   training-cache rows.

## Run

From the repo root:

```bash
# stage 1: build the fresh cache (resumable; safe to interrupt)
uv run python experiments/fresh-eval/encode_fresh.py

# stage 2: both architectures on the fresh rows
uv run python experiments/fresh-eval/pareto_fresh.py
uv run python experiments/fresh-eval/pareto_fresh.py \
    --ckpt data/out/sonar/multilingual/resid_nc --code both --plot
```

## Inputs

- Training cache `data/sonar_embeddings/mc4_4M.npy` (+ `.langs.npy`
  sidecar, used as the stream-drift guard while skipping and as the E[p]
  origin source in stage 2).
- Ontologizer checkpoint dir: known runs
  `data/out/sonar/multilingual/resid_nc_hm` (default) and
  `data/out/sonar/multilingual/resid_nc`.
- SAE runs auto-discovered from `data/out/sonar/sae/*/params.npz`
  (override with `--sae`).
- `--skip` defaults to the training cache's row count, read from its
  `.langs.npy`; pass it only for a cache without that sidecar.

## Outputs (in `experiments/fresh-eval/out/`)

- `mc4_fresh.npy`, `mc4_fresh.langs.npy`, `mc4_fresh.meta.json` — the
  fresh cache (default 131,072 rows ≈ 540 MB float32). Usable anywhere a
  `--cache` is accepted (textfid, langprobe, splitting, the
  steering-overlay experiment...).
- `pareto_fresh.csv` (label, coeffs, index_bits, fvu_w) and optionally
  `pareto_fresh.png`.

Read the result against `data/out/sonar/pareto/pareto.csv`: SAE points
should be ~unchanged (their tail was already held out); any FVU increase
on the onto points is the size of the train/eval-tail asymmetry.

## Cost

- **Stage 1 skip:** streaming past ~4M mC4 samples is network/CPU-bound —
  no encoding, but expect **2–6 h** depending on bandwidth (same cost
  encode_corpus.py pays on resume). Interruptions are cheap to resume for
  the encode phase, but the skip replays from zero each run.
- **Stage 1 encode:** 131,072 samples at b=256 ≈ 512 encoder batches —
  **~20–60 min on a single GPU** (SONAR encoder ~2.4 GB VRAM); not
  practical on CPU.
- **Stage 2:** same cost as pareto.py — **~5–15 min GPU** for all SAE
  runs + the onto sweep (checkpoint restore requires GPU JAX).

## Assumptions / caveats

- "Genuinely held out" rests on stream determinism: encode_corpus.py's
  own resume logic already depends on it, and the drift guard (language
  sidecar comparison over all skipped rows) aborts if the HF dataset
  revision no longer reproduces the cached order.
- A language that runs out is skipped from then on, not repeated: plain
  `all_exhausted` would restart it from its first document, which is
  training data. The fresh slice therefore lacks the languages that ran
  out inside the training corpus and loses more as others run out, so its
  mix is not the training head's (the `.langs.npy` sidecar records it).
  The whitened-FVU metric is per-dimension, not per-language, so this
  shifts difficulty identically for both architectures. The cache is
  truncated only if every language runs out before `--n` rows.
- The fresh rows carry the training cache's source-language tags, so for
  mc4_4M they share its English-tag issue (`experiments/lang-tags`) and
  stay on the distribution the models trained on.
- `pareto.onto_points`/`sae_points` are imported, not reimplemented — any
  future change to pareto.py's scoring flows through automatically.

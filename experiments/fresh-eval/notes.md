# fresh-eval: notes

## The fresh cache (2026-10-09)

`encode_fresh.py --b 64 --out data/sonar_embeddings`, run as a systemd
unit, wrote `data/sonar_embeddings/mc4_fresh.npy` (+ `.langs.npy`,
`.meta.json`): 131,072 rows, stream rows 3,974,400..4,105,472, encoded
with mc4_4M's own source tags (`MC4_4M_TAGS`, recorded as `src_tags`)
under `stopping_strategy="all_exhausted_without_replacement"`. The
language drift guard passed over all 3,974,400 skipped rows. Encoding
took 67 minutes at 33 rows/s on the shared GPU.

The slice holds all 86 configs, about 1,542 rows each, except one with a
single row: the language whose last document is where mc4_4M ends. So
its mix is the training corpus's uniform one, short one language.

It was written to `data/sonar_embeddings/`, not this experiment's
`out/`, so `pareto_fresh.py` needs `--fresh-cache
data/sonar_embeddings/mc4_fresh.npy`.

## The Pareto on fresh rows (2026-10-09)

`pareto_fresh.py` on all 131,072 fresh rows, origin from mc4_4M's head,
mirroring the writeup's two Pareto runs:

    P="uv run python experiments/fresh-eval/pareto_fresh.py \
        --fresh-cache data/sonar_embeddings/mc4_fresh.npy --plot"
    $P --ckpt data/out/sonar/multilingual/ste_h76_init01 \
        --temperature 0.00015 --sae <the 14 sae_conv runs of pareto_unified> \
        --out experiments/fresh-eval/out/unified
    $P --ckpt data/out/sonar/multilingual/resid_nc --ms 1 2 4 8 16 --sae \
        --out experiments/fresh-eval/out/resid_nc

Every point reproduces its counterpart on mc4_4M's tail
(`data/out/sonar/pareto_unified`, `pareto_resid_nc`): the same
coefficients and index bits (the L1 SAEs' measured L0 moves by about one
latent, 60 to 61 at 1e-4), and FVU within 0.3% everywhere but the k5120
SAE, +0.9% of 1.1e-4. `resid_nc`, the model that trained on that tail,
moves least of all: its soft code is 0.0054 on both, its deviation codes
-0.3% to 0.0%, its argmax 17.34 against 17.36. The SAEs move +0.1% to
+0.2% and the straight-through stack not at all (0.1536 against 0.1535).

The selection-sweep run `sweep_softmax_shm` (same 5 x 32 x 32 shape,
T = 0.03, out at `experiments/fresh-eval/out/sweep_softmax_shm`, onto
points only) behaves the same against its own run on the tail
(`data/out/sonar/pareto/pareto_sweep_softmax_shm`): every point within
0.13%, its soft code 0.000675 against 0.000676. Out of sample it stays
6.5x below the single-layer soft variant `g160softmax` (0.00436 on fresh
rows), so the results section's "whether the stack beats it out of
sample is open" is answered: it does.

So the asymmetry this experiment was built to measure is not there:
`pareto.py`'s caveat that the softmax Ontologizer is scored on rows it
trained on changes none of its numbers by more than 0.3%, and the
writeup's in-sample qualifications on the softmax models can be read as
disclosures rather than effects.

## The quantization sweep on fresh rows (2026-10-09)

`quantrate.py --cache data/sonar_embeddings/mc4_fresh.npy --fit-cache
data/sonar_embeddings/mc4_4M.npy` scores the last 32,768 fresh rows with
origins and gain statistics from the training cache
(`data/out/sonar/quantrate_fresh`). Against the same sweep on mc4_4M's
tail every readout moves by at most 0.0004
(`experiments/ste-arm/notes.md`, "Operational rate").

That includes the asymmetry this experiment was built to measure:
`resid_nc`, which trained on mc4_4M's tail, scores 0.00539 unquantized
on fresh rows against 0.00538 on the tail, 0.2% of its error. On this
evidence the softmax Ontologizer's in-sample scores are not inflated by
having seen the rows.

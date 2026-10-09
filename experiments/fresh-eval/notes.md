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
data/sonar_embeddings/mc4_fresh.npy`. `pareto_fresh.py` has not been
run.

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

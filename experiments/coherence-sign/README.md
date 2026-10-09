# coherence-sign: which setting makes soft heads coherent?

`headcoh.py` scores the heads of `resid_nc`, the run behind the writeup's
"soft heads are contrast sets", as repelling: mean ζ_cos −17.6 at step
372.6k. It scores those of `resid_nc_hm`, the softmax model behind the
steering and development results, as coherent: +120.6 at step 370k. The
two runs differ in three settings:

| setting | `resid_nc` | `resid_nc_hm` |
|---|---|---|
| dictionary and encoder width `e` | 4096 | 2048 |
| batch mean-entropy bonus `s_Hm` | 0 | 1e-6 |
| between-head output cosine `s_hcossim` | 0 | 1e-5 (1e-6 after its first resume, at 372.6k) |

This directory retrains `resid_nc` with one of them changed at a time, on
a copy of the code that trained it. Results are in `notes.md`.

## Files

- `train_old.py` trains `resid_nc`'s configuration on the copy.
  `--s-hcossim`, `--s-hm` and `--e` set the three settings and default to
  `resid_nc`'s values. `--run` names the output directory under
  `data/out/sonar/multilingual/`, and `--dry-run` prints the model spec
  and exits. It reads `data/sonar_embeddings/mc4_4M.npy` and
  `data/out/sonar/mse_weights.npy`, the files `resid_nc` trained on, and
  writes the run's checkpoints and logs plus a `provenance.json`. This
  repository's scripts load the checkpoints as they load `resid_nc`'s.
- `ontologize/` is the earlier codebase's training path at commit
  `4d32e33`, the "PWAK integration" merge of 2026-09-11. It holds the 17
  files that training imports (15 modules and two empty subpackage
  `__init__.py`s), extracted with `git archive`, plus an `__init__.py`
  that makes it a regular package. Python puts a script's directory first
  on `sys.path`, and a regular package shadows this repository's
  namespace package, so `train_old.py` imports the copy; it stops if it
  imported anything else. The code is the commit's. Comments and
  docstrings were corrected where they contradicted it, and they are all
  that may change.
- `check_copy.py` checks that the copy's code still equals the commit's.
  It checks the file set and compares each file's syntax tree, docstrings
  removed, with a fingerprint recorded from the commit. It exits 1 on a
  mismatch.
- `notes.md` holds the results.

## Why a copy of the old code

Both runs were trained on the project's earlier codebase, before this
repository existed, and this repository's code differs from it in more
than the three settings. Several known differences can be switched off
here: the L1 gradient, which that code never applied (`sparse_F` never
reached `DictEnc`; see `sonar.py`'s `s_L1F` comment), `resid_gain`, the
constant coordinate at layer 0 and the held-out cache tail. One cannot:
`cossim_h`, the statistic `s_hcossim` weights, is retired here, and
setting its weight raises. And these may not be all the differences. The
copy avoids both problems. It is later than the code that trained
`resid_nc` in July and runs in this repository's newer environment; the
control checks that neither matters.

## Run

From the repository root:

```bash
OUT=data/out/sonar/multilingual
uv run python experiments/coherence-sign/check_copy.py
uv run python experiments/coherence-sign/train_old.py --run resid_nc_ctl \
    >> $OUT/resid_nc_ctl.log 2>&1
uv run python experiments/coherence-sign/train_old.py --run resid_nc_hcos1e-5 \
    --s-hcossim 1e-5 >> $OUT/resid_nc_hcos1e-5.log 2>&1
uv run python experiments/coherence-sign/train_old.py --run resid_nc_shm1e-6 \
    --s-hm 1e-6 >> $OUT/resid_nc_shm1e-6.log 2>&1
```

`resid_nc_ctl` is the control, with no setting changed. Each run is 24
epochs over the 3,974,400-row cache, 372,600 steps at batch 256, which
took `resid_nc` about 6.5 hours. Rerunning a command resumes the run from
its latest checkpoint. Memory is allocated on demand, so on a shared GPU
another process can push a run out of memory; rerun it.

## Score

Besides `resid_nc` and `resid_nc_hm`, the earlier codebase trained
`resid_nc_hc`, which is `resid_nc_hm` without the bonus. All six runs
checkpoint every 10,000 steps and keep those checkpoints, so they can be
compared at any step they have all reached:

```bash
STEP=370000
for r in resid_nc resid_nc_hc resid_nc_hm \
         resid_nc_ctl resid_nc_hcos1e-5 resid_nc_shm1e-6; do
  uv run python headcoh.py --model onto --ckpt $OUT/$r --step $STEP \
      --out $OUT/$r/headcoh_$STEP
done
```

Compare by layer from each `heads.csv` as well as in the mean.

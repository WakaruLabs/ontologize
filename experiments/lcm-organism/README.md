# lcm-organism: a Base-LCM as a model organism

Do Ontologizer codes trained on a Large Concept Model's hidden states reproduce across seeds better than
codes trained on the SONAR embeddings that LCM reads? `plan.md` gives the question, the gates and the
arms. Results go in `notes.md`.

## Files

| File | What it does |
|---|---|
| `lcm.py` | The Base-LCM (LCM team 2024, Sec. 2.3.1) in Flax: robust scaler, PreNet, a causal pre-norm transformer (RoPE, RMSNorm, a gated MLP, dropout), PostNet. `states(..., block)` is the residual stream after a block. Also the document batches: `doc_spans`, `gather`, `batches` |
| `train_lcm.py` | Trains it on the cache's documents outside the last 262,144 rows. Logs held-out next-sentence FVU_w and Gate A's baselines |
| `harvest_lcm.py` | Writes a checkpoint's states, or its untrained init's (`--step 0`), as a cache the dictionary tools read |
| `stateagree.py` | Two harvests of the same rows compared before any dictionary: CKA, Procrustes (rotation), whitened Procrustes and ridge R² (linear map) |
| `stateinput.py` | Gate B: the share of a block's state on the scored tail that a ridge explains from x_n, from x_n with the run's earlier mean, and from the model's own state for x_n read alone (everything it computes from x_n without context) |
| `f16cache.py` | The float16 copy of the cache, for the TPU upload |
| `phase1.sh` | Phase 1's readings once both seeds have trained: Gate B on blocks 4 and 8, tail-only harvests of each seed and its untrained init, and `stateagree.py` between the seeds and between the inits |
| `spectrum.py` | Each arm's covariance spectrum in the frame its loss weighs, on the same prefix and tail rows of every arm: participation ratio, effective rank, directions per share of the variance, `pareto.gaussian_reference`'s FVU at given rates, and the share of the variance between languages, of the total and of the pooled within-language covariance |
| `recolour.py` | The effective-dimension control: SONAR's rows through one linear map that gives them a harvest's spectrum, laid out as that harvest |

`tests/test_lcm_organism.py` covers the eight Python files.

## Run

```bash
# a pilot: throughput and shapes
uv run python experiments/lcm-organism/train_lcm.py --max-steps 2000 --out data/out/lcm-organism/pilot
# the LCM, about 105M parameters, one seed per run
uv run python experiments/lcm-organism/train_lcm.py --seed 42 --epochs 1
# block 4 of the trained model, and of its untrained init
uv run python experiments/lcm-organism/harvest_lcm.py --run data/out/lcm-organism/lcm_l8_d1024_s42 --block 4
uv run python experiments/lcm-organism/harvest_lcm.py --run data/out/lcm-organism/lcm_l8_d1024_s42 --block 4 --step 0
# Gate B: how much of each block's state its own row explains
uv run python experiments/lcm-organism/stateinput.py --run data/out/lcm-organism/lcm_l8_d1024_s42 --blocks 4 8
# two seeds' states; two untrained inits give the floor. --prefix-rows 0 harvests the tail alone
uv run python experiments/lcm-organism/harvest_lcm.py --run <run> --block 4 --prefix-rows 0 --out <tail>.npy
uv run python experiments/lcm-organism/stateagree.py --a data/activations/lcm_l8_d1024_s42_b4.npy \
    --b data/activations/lcm_l8_d1024_s43_b4.npy
# the control recipe on a harvest (plan.md, Phase 2)
uv run python experiments/next-sentence/train_pair.py --cache data/activations/lcm_l8_d1024_s42_b4.npy \
    --mse-weights data/activations/lcm_l8_d1024_s42_b4.mse_weights_matched.npy \
    --epochs 3 --seed 42 --keep-every 2500 --pair-term agree --s-pair 0 --out <dir>
# each Phase 2 arm's spectrum (default arms: lcm_b4, init_b4, sonar; --arm NAME CACHE WEIGHTS)
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/spectrum.py
# SONAR recoloured to block 4's spectrum, then trained and scored as an arm
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/recolour.py
```

On a TPU (`experiments/cloud`), stage the float16 cache with the sidecars the scripts read
(`mc4_sent16_f16.npy`, `.index.npy`, `.docstart.npy`, `.kind.npy`, `.langs.npy`) and
`data/out/sonar/sentcache/mse_weights_no837.npy`. Then queue lines such as:

```
lcm_s42	data/out/lcm-organism/lcm_l8_d1024_s42	python experiments/lcm-organism/train_lcm.py --cache data/sonar_embeddings/mc4_sent16_f16.npy --seed 42 --epochs 1
```

A run resumes from its latest checkpoint, and a SIGTERM saves the current step first, so a preempted or
expired node loses only what had not reached the bucket.

## Design notes

- **Targets.** A prediction at row n is scored only where row n + 1 is the next segment of the text
  (`--targets adjacent`). Runs skip unusable segments, so 12% of neighbouring rows within a document
  are not consecutive sentences. These are the pairs `eval_next.py` scores. The context still holds
  every row of the run.
- **No start or end embedding.** A run's first row is read but not predicted. The paper appends an
  end-of-text sentence for generation, and nothing here generates.
- **The scaler** is fitted on 131,072 training rows drawn with seed 0, whatever `--seed` is. So every
  run on one cache normalizes alike, and seeds differ only in the init and the data order.
- **Step 0 is kept.** It is the untrained init that the untrained arm harvests, at the same scaler.
- **The MLP** (`GLU`) is built from the package's layers. Its up-projection is `NLinear` with n = 2:
  one weight of shape (2, d_ff, d), gated on the first slice. The down projection is a `Linear`.
  - `--gate silu`, the default, is the paper's SwiGLU.
  - `--gate none` makes the up-projection a `Bilinear` holding the same weight. Its interaction tensor
    and eigendecomposition (Pearce et al. 2025) then read the trained LCM. Such a run is named
    `..._bilinear_s<seed>`.
  - Both layers default to bfloat16 for parameters and compute, so `lcm.F32` sets float32.
- **Optimizer.** AdamW, with β2 0.95 and weight decay 0.1 on weight matrices only. Linear warmup, then
  cosine decay to a tenth of the peak. Dropout 0.1, the paper's rate, on each residual branch; the
  paper doesn't say where it applies its dropout.
- **One device per run.** On a v5e-4, four runs go at once, one per chip.
- **Precision.** Training runs at the runner's matmul precision (`high`). Held-out scores and harvests
  run at `highest`.
- **Harvest numbering.**
  - Harvest row i is cache row i in the prefix, so `--rows` and `--fit-rows` mean what they mean on
    the cache.
  - The tail follows, so the last 32,768 rows are the cache's last 32,768.
  - `harvest_lcm.py` refuses a harvest whose scored tail differs from the cache's.
- **Procrustes.** `plan.md` asked for Procrustes after whitening as the rotation test. Whitening absorbs
  any invertible linear map, though, so whitened Procrustes tests the same thing as the ridge.
  `stateagree.py` reports both: `procrustes` scales each side to unit total variance and tests rotation;
  `procrustes_whitened` tests the linear map.
- **The weighted frame.** Each row is scaled by the square root of the MSE weights its dictionary
  trains under, so the eigenvalues are what FVU_w divides by. A harvest's weights are inverse
  variance, so its spectrum there reflects correlation, not each coordinate's scale.
- **Recolouring** (`recolour.py`).
  - **The map.** In SONAR's weighted frame, each principal direction is rescaled so its variance is
    the harvest's at the same rank. A random rotation then spreads the variance over every coordinate,
    because the inverse-variance weights the trainer applies would otherwise flatten the spectrum
    again. A last constant brings each coordinate's variance to about 1, inside float16's precision.
  - **What it keeps.** The map is linear and has no offset. So each row keeps what its sentence's
    embedding carries, except the direction SONAR's weights zero, and depends on no other row.
  - **What it doesn't match.** It matches the total spectrum only. The split between variance
    across languages and within them, and the mean's share of a row's norm, stay SONAR's.

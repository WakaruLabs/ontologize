# lcm-organism: a Base-LCM as a model organism

Do Ontologizer codes trained on a Large Concept Model's hidden states reproduce across seeds better than
codes trained on the SONAR embeddings that LCM reads? `plan.md` gives the question, the gates and the
arms. Results go in `notes.md`.

## Files

| File | What it does |
|---|---|
| `lcm.py` | The Base-LCM (LCM team 2024, Sec. 2.3.1) in Flax: robust scaler, PreNet, a causal pre-norm transformer (RoPE, RMSNorm, SwiGLU, dropout), PostNet. `states(..., block)` is the residual stream after a block. Also the document batches: `doc_spans`, `gather`, `batches` |
| `train_lcm.py` | Trains it on the cache's documents outside the last 262,144 rows. Logs held-out next-sentence FVU_w and Gate A's baselines |
| `harvest_lcm.py` | Writes a checkpoint's states, or its untrained init's (`--step 0`), as a cache the dictionary tools read |
| `stateagree.py` | Two harvests of the same rows compared before any dictionary: CKA, Procrustes (rotation), whitened Procrustes and ridge R² (linear map) |
| `f16cache.py` | The float16 copy of the cache, for the TPU upload |

`tests/test_lcm_organism.py` covers all five.

## Run

```bash
# a pilot: throughput and shapes
uv run python experiments/lcm-organism/train_lcm.py --max-steps 2000 --out data/out/lcm-organism/pilot
# the LCM, about 105M parameters, one seed per run
uv run python experiments/lcm-organism/train_lcm.py --seed 42 --epochs 1
# block 4 of the trained model, and of its untrained init
uv run python experiments/lcm-organism/harvest_lcm.py --run data/out/lcm-organism/lcm_l8_d1024_s42 --block 4
uv run python experiments/lcm-organism/harvest_lcm.py --run data/out/lcm-organism/lcm_l8_d1024_s42 --block 4 --step 0
# two seeds' states; two untrained inits give the floor
uv run python experiments/lcm-organism/stateagree.py --a data/activations/lcm_l8_d1024_s42_b4.npy \
    --b data/activations/lcm_l8_d1024_s43_b4.npy
# the control recipe on a harvest (plan.md, Phase 2)
uv run python experiments/next-sentence/train_pair.py --cache data/activations/lcm_l8_d1024_s42_b4.npy \
    --mse-weights data/activations/lcm_l8_d1024_s42_b4.mse_weights_matched.npy \
    --epochs 3 --seed 42 --keep-every 2500 --pair-term agree --s-pair 0 --out <dir>
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

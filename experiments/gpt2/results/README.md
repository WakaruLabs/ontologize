# Headline results: two discrete codes for GPT-2 small, layer 8

Branch `headline` (from `hard-codes` / `gpt2-experiments` at `f1f2158`, 2026-10-01). Both runs:
GPT-2 small residual stream entering block 8, FineWeb activations, **one pass over 40M fresh
tokens**, 5 layers × 32 heads × 32 labels = **800 bits per token, no continuous coefficients**,
gain-shape residual coding, argmax forward / softmax backward (STE) with the temperature annealed
1 → 0.03 over the first 60k of 156k steps. Held-out evaluation on 520k tokens from documents
disjoint from training (`fvu_hard`; `ce.hard_recovered` = fraction of GPT-2's loss recovered when
the reconstruction is spliced into the model).

| run | where the entries live | FVU | GPT-2 loss recovered | per-layer prefix FVU |
|---|---|---|---|---|
| `D40_direct_gain_ste` | activation space (`R^768`, signed; no decoder) | 0.195 | 97.6% | .472 .347 .278 .231 .195 |
| `D40_private_gain_ste` | head-private 48-dim blocks of a 1536-dim latent, concatenated, one shared linear decoder | 0.186 | 97.5% | .451 .333 .265 .218 .186 |

For scale: a 10-epoch top-k SAE (k = 32, 5120 latents, 100M tokens) reaches FVU 0.189 / 96.2%
with 32 continuous coefficients plus their indices per token. With 64 labels per head (960 bits)
the private design reaches 0.164 / 97.9% (`D40_private_gain_ste_k64`, on the `hard-codes` branch).

Each directory holds `config.json`, `diag_156160.json` and `loss.csv.gz` (one row per step:
`loss, MSE, MSE_ghost, L1_K, L1_F, entropy, cossim_b, cossim_h, KL_m, aux`; train FVU ≈ MSE × 768).
Checkpoints live under `data/out/gpt2/<run>/` on the training machine (not in git).

## Reproduction

```
uv run python experiments/gpt2/make_cache.py                                   # 10M-token cache data/gpt2_l8
uv run python experiments/gpt2/extend_cache.py --train-tokens 40000000         # 40M-token cache data/gpt2_l8_40m

uv run python experiments/gpt2/train_onto.py --name D40_direct_gain_ste --cache data/gpt2_l8_40m \
    --epochs 1 --lr 2e-4 --anneal-steps 60000 --resid-first --direct --signed-dict --select ste \
    --resid-gain --gain-clip --save-each 5000 --max-to-keep 3

uv run python experiments/gpt2/train_onto.py --name D40_private_gain_ste --cache data/gpt2_l8_40m \
    --epochs 1 --lr 2e-4 --anneal-steps 60000 --resid-first --resid-gain --gain-clip --select ste \
    --private-heads --save-each 5000 --max-to-keep 3

uv run python experiments/gpt2/diagnose.py data/out/gpt2/<run> --cache data/gpt2_l8_40m --ce
```

Design note for the direct variant: [`../DIRECT_GAIN_STE.md`](../DIRECT_GAIN_STE.md).

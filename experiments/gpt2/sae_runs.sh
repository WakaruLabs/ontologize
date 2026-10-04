#!/bin/bash
# SAE baselines on the GPT-2 cache (train rows only; eval = held-out tail),
# then FVU / L0 / splice CE for each.
cd "$(dirname "$0")/../.."
C="--cache data/gpt2_l8/acts.npy --eval-rows 520192 --lr 4e-4 --b 4096 --epochs 10 --overwrite"
run() { name=$1; shift
  .venv/bin/python sae.py $C --mse-weights "" --out data/out/gpt2/sae/$name "$@" > data/out/gpt2/logs/sae_$name.log 2>&1 &&
  .venv/bin/python experiments/gpt2/eval_sae.py data/out/gpt2/sae/$name >> data/out/gpt2/logs/sae_$name.log 2>&1; }
( run m5120_k32 --m 5120 --topk 32; run m5120_g160softmax --m 5120 --topk 0 --groups 160 --group-fn softmax;
  run m16384_k32 --m 16384 --topk 32 ) &
( run m5120_k128 --m 5120 --topk 128; run m5120_g160top1 --m 5120 --topk 0 --groups 160 --group-fn top1 ) &
wait

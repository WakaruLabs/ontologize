#!/bin/bash
# 5-layer LR sweep, 1 epoch each, two runs at a time; then diagnostics for
# the two baseline runs whose splice eval hit OOM in the first attempt.
cd "$(dirname "$0")/../.."
run() { .venv/bin/python experiments/gpt2/train_onto.py --epochs 1 --overwrite "$@" \
          > data/out/gpt2/logs/$2.log 2>&1; }
( run --name B_lr5e-5 --lr 5e-5 --resid-first; run --name B_lr1e-3 --lr 1e-3 --resid-first ) &
( run --name B_lr2e-4 --lr 2e-4 --resid-first; run --name B_lr5e-4 --lr 5e-4 --resid-first ) &
wait
for n in A_lr5e-5 A_lr5e-4; do
  .venv/bin/python experiments/gpt2/diagnose.py data/out/gpt2/$n --ce > data/out/gpt2/logs/$n.diag.log 2>&1
done

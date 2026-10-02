#!/bin/bash
# Diagnostics for wide-encoder models beside a running training pair: small
# eval batches and a small GPU pool. Usage: diag_small.sh NAME...
cd "$(dirname "$0")/../.."
for n in "$@"; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 .venv/bin/python experiments/gpt2/diagnose.py data/out/gpt2/$n --ce --rows 32768 --bs 256 --splice-batch 8 \
    > data/out/gpt2/logs/$n.diag.log 2>&1
done

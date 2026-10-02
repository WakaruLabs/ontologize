#!/bin/bash
# Diagnostics on the CPU (for models whose diagnostics don't fit beside training runs).
cd "$(dirname "$0")/../.."
for n in "$@"; do
  JAX_PLATFORMS=cpu .venv/bin/python experiments/gpt2/diagnose.py data/out/gpt2/$n --ce --rows 32768 > data/out/gpt2/logs/$n.diag.log 2>&1
done

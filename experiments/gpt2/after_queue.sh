#!/bin/bash
# After the paired queues: runs that need the GPU alone.
cd "$(dirname "$0")/../.."
L=data/out/gpt2/logs
until grep -q "DRIVER DONE" $L/after_yolo2.out 2>/dev/null; do sleep 60; done
while pgrep -f '^\.venv/bin/python experiments/gpt2/(train_onto|diagnose)\.py' >/dev/null; do sleep 30; done
echo "shared 20480 start $(date)"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 .venv/bin/python experiments/gpt2/train_onto.py --name V_direct_topk64_m20480_shared --overwrite $(cat $L/args_20480.txt) > $L/V_direct_topk64_m20480_shared.log 2>&1
experiments/gpt2/diag_small.sh V_direct_topk64_m20480_shared
echo "AFTER QUEUE DONE $(date)"

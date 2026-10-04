#!/bin/bash
# Finish the interrupted YOLO run alone on the GPU (resampling off from 125k: the
# resampler leaked ~4 GB host memory per resample after the resume), then the queue.
cd "$(dirname "$0")/../.."
L=data/out/gpt2/logs
while pgrep -f '^\.venv/bin/python experiments/gpt2/train_onto\.py' >/dev/null; do sleep 30; done
COMMON="--cache data/gpt2_l8_40m --epochs 1 --lr 2e-4 --anneal-steps 60000 --resid-first --direct --signed-dict --select ste --resid-gain --gain-clip --head-sparse topk --m-h 4096 --k-z 64 --hs-lr-mult 2 --resample-period 0 --save-each 5000 --checkpoint-each 100000 --max-to-keep 1"
echo "YOLO resume start $(date)"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 .venv/bin/python experiments/gpt2/train_onto.py --name Y_topk64_m4096 --resume --max-steps 156250 --b 256 $COMMON > $L/Y_topk64_m4096.resume.log 2>&1
echo "YOLO resume done $(date)"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 .venv/bin/python experiments/gpt2/diagnose.py data/out/gpt2/Y_topk64_m4096 --cache data/gpt2_l8_40m --ce --rows 32768 --bs 256 > $L/Y_topk64_m4096.diag2.log 2>&1
echo "YOLO diag done $(date)"
# 2048x16 per head alone (does not fit beside another run), then the paired queues
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 .venv/bin/python experiments/gpt2/train_onto.py --name V_direct_topk16_m2048 --overwrite $(sed 's/^V_direct_topk16_m2048 //' $L/q_2048.txt) > $L/V_direct_topk16_m2048.log 2>&1
experiments/gpt2/diag_small.sh V_direct_topk16_m2048
exec experiments/gpt2/drive.sh q_wide2 q_late2 q_topm q_long

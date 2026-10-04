#!/bin/bash
# The 4096x64-per-head YOLO run: waits for the running training, the cache
# build and their diagnostics, smoke-tests memory at b=256 (falls back to
# b=128), then trains alone on the GPU on the 40M-token cache.
cd "$(dirname "$0")/../.."
L=data/out/gpt2/logs
while pgrep -f '^\.venv/bin/python experiments/gpt2/train_onto\.py' >/dev/null; do sleep 30; done
while pgrep -f '^\.venv/bin/python experiments/gpt2/extend_cache\.py' >/dev/null; do sleep 30; done
while pgrep -f '^\.venv/bin/python experiments/gpt2/diagnose\.py' >/dev/null; do sleep 30; done
[ -f data/gpt2_l8_40m/meta.json ] || { echo "no 40M cache"; exit 1; }
COMMON="--cache data/gpt2_l8_40m --epochs 1 --lr 2e-4 --anneal-steps 60000 --resid-first --direct --signed-dict --select ste --resid-gain --gain-clip --head-sparse topk --m-h 4096 --k-z 64 --hs-lr-mult 2 --resample-period 5000 --resample-count-every 50 --resample-scale 1.1 --save-each 5000 --checkpoint-each 100000 --max-to-keep 1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
B=256
if false && ! .venv/bin/python experiments/gpt2/train_onto.py --name Y_smoke --overwrite --max-steps 40 --save-each 20 --b 256 $COMMON > $L/Y_smoke.log 2>&1; then
  echo "b=256 failed (see Y_smoke.log); trying b=128"; B=128
  .venv/bin/python experiments/gpt2/train_onto.py --name Y_smoke --overwrite --max-steps 40 --save-each 20 --b 128 $COMMON > $L/Y_smoke_b128.log 2>&1 || { echo "b=128 failed too"; exit 1; }
fi
rm -rf data/out/gpt2/Y_smoke
STEPS=$(( 156250 * 256 / B ))   # one pass over the 40M cache; 7.8 it/s at b=256 measured: ~5.6 h
echo "YOLO start $(date) b=$B steps=$STEPS"
.venv/bin/python experiments/gpt2/train_onto.py --name Y_topk64_m4096 --overwrite --max-steps $STEPS --b $B $COMMON > $L/Y_topk64_m4096.log 2>&1
echo "YOLO training done $(date)"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 .venv/bin/python experiments/gpt2/diagnose.py data/out/gpt2/Y_topk64_m4096 --cache data/gpt2_l8_40m --ce --rows 32768 --bs 256 > $L/Y_topk64_m4096.diag.log 2>&1
echo "YOLO diag done $(date)"
# resume the paused queue: remaining frozen-fiber runs, then the rest
exec experiments/gpt2/drive.sh q_rest q_wide q_late2 q_topm q_long

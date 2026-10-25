#!/usr/bin/env bash
# SAEs that can spend as many bits per row as the Ontologizers' hard codes,
# for quantrate.py on mc4_sent16. Each trains on the whole cache under the
# masked MSE weights (`cachestats.py mask`), with block-shuffled batches as
# in sae_whole_run.sh, and with sae_conv's recipe for its kind
# (sae_ladder.sh, sae_evals.sh, latex/config/sae.tex):
# - k32: m = 5120, lr 4e-4, aux-k 512, 144,300 steps of 4,096 (13.4 epochs
#   here). This is the ladder's lr, which sae_whole_run.sh's k32 (sae.py's
#   default 1e-4) lacks;
# - the L1 sweep: m = 5120, lr 4e-4, neuron resampling every 5,000 steps in
#   place of aux-k, the same 144,300 steps;
# - k160: m = 11264, lr 1e-4, aux-k 512, 336,000 steps (31.3 epochs).
# The L1 weights are sae_conv's five times L1_SCALE. The L1 term grows with
# the codes' raw norm and the whitened MSE with the rows' whitened variance;
# on this cache under these weights the first is 0.255 and the second 0.0118
# of mc4_4M's, so 0.0118 / 0.255 keeps the two terms' balance (notes.md).
#
#   sae_rate_runs.sh [adam_eps]
#
# adam_eps defaults to optax's 1e-8; any other value goes to sae.py
# --adam-eps and adds _eps<adam_eps> to the run and log names.
# Training waits until the GPU is idle, as full_runs.sh does. Every sae.py
# run resumes from its own snapshot, and a finished one only re-evaluates
# and exits, so this runs as a systemd unit with Restart=on-failure and a
# restart picks up the run it was in.
set -eu
cd "$(dirname "$0")/../.."
eps=${1:-1e-8}
tag=""
[ "$eps" = 1e-8 ] || tag="_eps$eps"
L1_SCALE=0.0464
cache=data/sonar_embeddings/mc4_sent16.npy
weights=data/out/sonar/sentcache/mse_weights_no837.npy
common="--cache $cache --mse-weights $weights --adam-eps $eps
        --block-rows 65536 --buffer-blocks 16"
out=data/out/sonar/sae_sent16
logs=experiments/logs

# idle: utilization at most 10% at five samples two seconds apart
idle() {
    for _ in 1 2 3 4 5; do
        u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        [ "$u" -le 10 ] || return 1
        sleep 2
    done
}
calm=0
while [ "$calm" -lt 3 ]; do
    if idle; then calm=$((calm + 1)); else calm=0; fi
    if [ "$calm" -lt 3 ]; then sleep 300; fi
done

name=m5120_k32_lr4e-4_no837_whole$tag
/usr/bin/uv run python sae.py --m 5120 --topk 32 --lr 4e-4 $common \
    --epochs 14 --max-steps 144300 --out $out/$name \
    >> $logs/sae_sent16_$name.log 2>&1
for base in 3e-6 1e-5 3e-5 1e-4 3e-4; do
    l1=$(awk -v b="$base" -v s="$L1_SCALE" 'BEGIN { printf "%.3g", b * s }')
    name=m5120_l1${l1}_no837_whole$tag
    /usr/bin/uv run python sae.py --m 5120 --topk 0 --l1 "$l1" --aux-k 0 \
        --resample-every 5000 --lr 4e-4 $common \
        --epochs 14 --max-steps 144300 --out $out/$name \
        >> $logs/sae_sent16_$name.log 2>&1
done
name=m11264_k160_no837_whole$tag
/usr/bin/uv run python sae.py --m 11264 --topk 160 --lr 1e-4 $common \
    --epochs 32 --max-steps 336000 --out $out/$name \
    >> $logs/sae_sent16_$name.log 2>&1

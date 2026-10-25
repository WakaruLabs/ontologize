#!/usr/bin/env bash
# Full-length Ontologizer training on mc4_sent16 under the masked MSE
# weights (`cachestats.py mask`: coordinate 837 at 0, every other weight the
# cache's): the direct k = 128 stack, then the hard stack, with the shipped
# recipes apart from the mean-entropy weight, each over the whole cache for
# 2 epochs (344,383 steps of 256, about the shipped runs' 369,528 on
# mc4_4M).
#
#   full_runs.sh [s_hm [adam_eps]]
#
# s_hm defaults to the recipe's 1e-4, which on this cache leaves the MSE a
# few percent of the loss (notes.md, 2026-10-22); any other value names the
# runs and logs ..._shm<s_hm>. adam_eps defaults to optax's 1e-8, a floor in
# the loss's units that this cache's small loss sits near; any other value
# trains through experiments/adam-eps/train_eps.py, which takes
# train_ste.py's flags and --adam-eps, and adds _eps<adam_eps> to the names.
# Training waits until the GPU is idle
# (utilization at most 10% at three checks five minutes apart), so it starts
# when no other job is running rather than beside one. The two runs go one
# after the other because both read the cache at random. Every trainer
# resumes from its own checkpoints, so this runs as a systemd unit with
# Restart=on-failure, and a restart waits for the GPU again.
set -eu
cd "$(dirname "$0")/../.."
s_hm=${1:-1e-4}
eps=${2:-1e-8}
tag=""
[ "$s_hm" = 1e-4 ] || tag="_shm$s_hm"
trainer=experiments/ste-arm/train_ste.py
if [ "$eps" != 1e-8 ]; then
    tag="${tag}_eps$eps"
    trainer="experiments/adam-eps/train_eps.py --adam-eps $eps"
fi
cache=data/sonar_embeddings/mc4_sent16.npy
weights=data/out/sonar/sentcache/mse_weights_no837.npy
ste="--temperature 0.00015 --lr 1e-5 --noise-k batchnorm --sd-k 0.3 --s-hm $s_hm
     --dict-init-scale 0.1 --cache $cache --mse-weights $weights --epochs 2"
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

/usr/bin/uv run python $trainer $ste \
    --h 54 --k 128 --signed --direct \
    --out data/out/sonar/sent16/ste_k128_h54_direct_no837_e2$tag \
    >> $logs/sent16_ste_k128_h54_direct_no837_e2$tag.log 2>&1
/usr/bin/uv run python $trainer $ste \
    --out data/out/sonar/sent16/ste_h76_init01_no837_e2$tag \
    >> $logs/sent16_ste_h76_init01_no837_e2$tag.log 2>&1

#!/usr/bin/env bash
# Seed agreement against width in the trained stack: one-layer stacks of h
# heads of 32 (the recipe ladder's all-auxiliary-weights-off arm, with
# training noise, on mc4_sent16) at seeds 42 and 43, for each h in WIDTHS,
# each pair scored by partition.py (matched head NMI against the shuffled
# null) and ratecurve.py (FVU at the layer's rate).
#
# Without a classifier or training, LSQ's matched NMI halves with each
# doubling of the codebooks, 0.57 at one to 0.0089 at 76 (notes.md,
# 2026-11-21), and the stack's layer 0 reads 0.025-0.027 at 76. If one
# trained head reproduces near k-means' 0.61 and the curve follows LSQ's,
# the stack's disagreement is what any additive code of its width does; if
# trained heads fall below LSQ at small h too, the classifier or the
# training loses agreement on top of that.
#
# Like gauss_seeds.sh it waits for an idle GPU and trains one run at a
# time; the trainer resumes from its own checkpoint and passes over a
# finished run, so it runs as a systemd unit with Restart=on-failure.
#
#   WIDTHS="1 4 16 76" experiments/seed-stability/width_seeds.sh
set -eu
cd "$(dirname "$0")/../.."
out=data/out/sonar/width_seeds
logs=experiments/logs
cache=data/sonar_embeddings/mc4_sent16.npy
weights=data/out/sonar/sentcache/mse_weights_med837.npy
read -ra widths <<< "${WIDTHS:-1 4 16 76}"
recipe="--temperature 0.00015 --lr 1e-5 --noise-k batchnorm --sd-k 0.3
        --s-hm 2e-6 --dict-init-scale 0.1 --cache $cache --mse-weights $weights
        --steps 50000 --adam-eps 1e-15 --max-to-keep 1 --checkpoint-each 50000
        --s-bcossim 0 --s-support 0 --s-l1f 0 --s-hm 0 --l 1"

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

mkdir -p $out/scores
for h in "${widths[@]}"; do
    for seed in 42 43; do
        name=l1_h${h}_s$seed
        /usr/bin/uv run python experiments/adam-eps/train_eps.py $recipe \
            --h $h --seed $seed --out $out/$name >> $logs/width_seeds_$name.log 2>&1
    done
done

score() {
    local name=$1; shift
    /usr/bin/uv run python "$@" > $out/scores/$name.log 2>&1 \
        || echo "$name failed" >> $out/scores/$name.log
}
for h in "${widths[@]}"; do
    a=$out/l1_h${h}_s42
    b=$out/l1_h${h}_s43
    score partition_h$h experiments/ste-arm/partition.py --a $a --b $b --cache $cache
    score ratecurve_h$h experiments/rate-penalty/ratecurve.py --ckpt $a $b \
        --out $out/scores/ratecurve_h$h
done
echo "done $(date '+%F %H:%M')" >> $out/scores/done.log

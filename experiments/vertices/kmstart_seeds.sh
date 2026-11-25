#!/usr/bin/env bash
# The one-head seed test from a spherical k-means start (kmstart.py).
#
# The stock one-head pair (seed-stability/width_seeds.sh, h = 1) agrees at
# NMI 0.418, spherical k-means at 32 cells at 0.732. This trains the same
# recipe from k-means starts:
#   km42_s42, km43_s43  each seed from its own k-means fit
#   km42_s43            seed 43 from seed 42's fit: the shared start
# and scores every run against its start and every pair of runs. If the own
# pair keeps most of its starts' agreement, the agreement belongs to the
# start and the optimizer; if both pairs fall to about 0.42, training
# forgets its start, and the archetypal SAE's centroid start would not
# explain its stability either.
#
# It fits the starts on CPU unless they exist, waits for an idle GPU as
# width_seeds.sh does, and trains one run at a time. The trainer resumes
# from its own checkpoint and passes over a finished run, so it runs as a
# systemd unit with Restart=on-failure (README.md).
set -eu
cd "$(dirname "$0")/../.."
out=data/out/sonar/vertices/kmstart
logs=experiments/logs
cache=data/sonar_embeddings/mc4_sent16.npy
weights=data/out/sonar/sentcache/mse_weights_med837.npy
stock=data/out/sonar/width_seeds
# width_seeds.sh's recipe, flag for flag
recipe="--temperature 0.00015 --lr 1e-5 --noise-k batchnorm --sd-k 0.3
        --s-hm 2e-6 --dict-init-scale 0.1 --cache $cache --mse-weights $weights
        --steps 50000 --adam-eps 1e-15 --max-to-keep 1 --checkpoint-each 50000
        --s-bcossim 0 --s-support 0 --s-l1f 0 --s-hm 0 --l 1 --h 1"

mkdir -p $out/scores $logs
if [ ! -f $out/fit/summary.json ]; then
    JAX_PLATFORMS=cpu /usr/bin/uv run python experiments/vertices/kmstart.py fit \
        --cache $cache --mse-weights $weights \
        --margin-from $stock/l1_h1_s42 \
        --stock $stock/l1_h1_s42 $stock/l1_h1_s43 \
        --out $out/fit >> $logs/kmstart_fit.log 2>&1
fi

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

for run in "42 42" "43 43" "42 43"; do
    read -r km seed <<< "$run"
    name=km${km}_s$seed
    /usr/bin/uv run python experiments/vertices/train_kmstart.py $recipe \
        --seed $seed --km-start $out/fit/start_s$km.npz \
        --out $out/$name >> $logs/kmstart_$name.log 2>&1
done

JAX_PLATFORMS=cpu /usr/bin/uv run python experiments/vertices/kmstart.py score \
    --cache $cache \
    --runs km42_s42:$out/km42_s42:$out/fit/start_s42.npz \
           km43_s43:$out/km43_s43:$out/fit/start_s43.npz \
           km42_s43:$out/km42_s43:$out/fit/start_s42.npz \
    --out $out/scores/score.json > $out/scores/score.log 2>&1 \
    || echo "score failed" >> $out/scores/score.log
echo "done $(date '+%F %H:%M')" >> $out/scores/done.log

#!/usr/bin/env bash
# The paired runs behind notes.md's "Coordinate 837's weight in training":
# the k32 SAE, the hard stack and the direct k = 128 stack on mc4_sent16,
# each under the cache's MSE weights (_w837) and under `cachestats.py mask`'s
# copy with coordinate 837 at 0 (_no837), with the shipped recipes. The
# Ontologizer runs' 50,000 steps of 256 see the cache's first 12.8M rows
# once (`--steps` exposes only the head); the SAE trains on the same rows
# (`--train-rows`) for sae_conv's 144,000 steps, since sampling the whole
# 180 GB cache at random is bound by the disk. The head is read into the
# page cache once first. The SAE pair runs beside the Ontologizer runs,
# which go one at a time. Every trainer resumes from its own checkpoints, so
# this runs as a systemd unit with Restart=on-failure: a rerun passes over
# the runs already complete.
set -eu
cd "$(dirname "$0")/../.."
cache=data/sonar_embeddings/mc4_sent16.npy
rows=12800000
declare -A weights=(
    [w837]=data/sonar_embeddings/mc4_sent16.mse_weights.npy
    [no837]=data/out/sonar/sentcache/mse_weights_no837.npy)
ste="--temperature 0.00015 --lr 1e-5 --noise-k batchnorm --sd-k 0.3 --s-hm 1e-4
     --dict-init-scale 0.1 --cache $cache --steps 50000"
logs=experiments/logs

# the head's 12.8M rows of 4 KiB and the .npy header, in 64 MiB reads
dd if=$cache of=/dev/null bs=64M count=$(( (rows * 4096 + 128) / (64 << 20) + 1 )) status=none

(
    for arm in w837 no837; do
        /usr/bin/uv run python sae.py --m 5120 --topk 32 --cache $cache \
            --mse-weights "${weights[$arm]}" --train-rows $rows \
            --epochs 50 --max-steps 144000 \
            --out data/out/sonar/sae_sent16/m5120_k32_$arm \
            >> $logs/sae_sent16_m5120_k32_$arm.log 2>&1
    done
) &
sae=$!

for arm in w837 no837; do
    /usr/bin/uv run python experiments/ste-arm/train_ste.py $ste \
        --mse-weights "${weights[$arm]}" \
        --out data/out/sonar/sent16/ste_h76_init01_$arm \
        >> $logs/sent16_ste_h76_init01_$arm.log 2>&1
done
for arm in w837 no837; do
    /usr/bin/uv run python experiments/ste-arm/train_ste.py $ste \
        --h 54 --k 128 --signed --direct --mse-weights "${weights[$arm]}" \
        --out data/out/sonar/sent16/ste_k128_h54_direct_$arm \
        >> $logs/sent16_ste_k128_h54_direct_$arm.log 2>&1
done
wait $sae

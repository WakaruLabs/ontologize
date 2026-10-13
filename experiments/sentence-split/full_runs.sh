#!/usr/bin/env bash
# Full-length Ontologizer training on mc4_sent16 under the masked MSE
# weights (`cachestats.py mask`: coordinate 837 at 0, every other weight the
# cache's): the direct k = 128 stack, then the hard stack, with the shipped
# recipes, each over the whole cache for 2 epochs (344,383 steps of 256,
# about the shipped runs' 369,528 on mc4_4M). They run one after the other
# because both read the cache at random. Every trainer resumes from its own
# checkpoints, so this runs as a systemd unit with Restart=on-failure.
set -eu
cd "$(dirname "$0")/../.."
cache=data/sonar_embeddings/mc4_sent16.npy
weights=data/out/sonar/sentcache/mse_weights_no837.npy
ste="--temperature 0.00015 --lr 1e-5 --noise-k batchnorm --sd-k 0.3 --s-hm 1e-4
     --dict-init-scale 0.1 --cache $cache --mse-weights $weights --epochs 2"
logs=experiments/logs

/usr/bin/uv run python experiments/ste-arm/train_ste.py $ste \
    --h 54 --k 128 --signed --direct \
    --out data/out/sonar/sent16/ste_k128_h54_direct_no837_e2 \
    >> $logs/sent16_ste_k128_h54_direct_no837_e2.log 2>&1
/usr/bin/uv run python experiments/ste-arm/train_ste.py $ste \
    --out data/out/sonar/sent16/ste_h76_init01_no837_e2 \
    >> $logs/sent16_ste_h76_init01_no837_e2.log 2>&1

#!/usr/bin/env bash
# The Ontologizer half of the recipe ladder (README.md): the hard stack and
# the direct k = 128 stack on mc4_sent16, each with one training noise or
# one auxiliary weight removed, against a control and a seed-43 replica of
# it. Every arm has the eps 1e-15 full runs' flags (full_runs.sh 2e-6 1e-15)
# at mask_runs.sh's schedule: 50,000 steps of 256, which see the cache's
# first 12.8M rows once (--steps exposes only the head).
#
#   ladder.sh WEIGHTS
#
# WEIGHTS names the MSE weights file
# data/out/sonar/sentcache/mse_weights_<WEIGHTS>.npy (no837 or med837) and
# the runs. Each arm keeps only its final checkpoint (--max-to-keep 1
# --checkpoint-each 50000): collapse is read from loss.csv, which logs every
# 250 steps. The arms go in pairs, both architectures per arm, in the order
# below, so a partial queue leaves complete comparisons. Training waits
# until the GPU is idle (utilization at most 10% at three checks five
# minutes apart), as full_runs.sh does. Every trainer resumes from its own
# checkpoint, so this runs as a systemd unit with Restart=on-failure, and a
# finished arm is passed over.
set -eu
cd "$(dirname "$0")/../.."
wt=${1:?usage: ladder.sh WEIGHTS}
cache=data/sonar_embeddings/mc4_sent16.npy
weights=data/out/sonar/sentcache/mse_weights_$wt.npy
[ -f "$weights" ] || { echo "no $weights" >&2; exit 1; }
rows=12800000
recipe="--temperature 0.00015 --lr 1e-5 --noise-k batchnorm --sd-k 0.3
        --s-hm 2e-6 --dict-init-scale 0.1 --cache $cache --mse-weights $weights
        --steps 50000 --adam-eps 1e-15 --max-to-keep 1 --checkpoint-each 50000"
out=data/out/sonar/recipe_ladder
logs=experiments/logs

declare -A arch=(
    [ste_h76_init01]=""
    [ste_k128_h54_direct]="--h 54 --k 128 --signed --direct")
arms=(ctrl ctrl_s43 nosdk nosdf nopdrop nonoise nobcos nosupp nol1f nohm)
# a later flag overrides the recipe's
declare -A flags=(
    [ctrl]="" [ctrl_s43]="--seed 43"
    [nosdk]="--sd-k 0" [nosdf]="--sd-f 0" [nopdrop]="--p-drop 0"
    [nonoise]="--sd-k 0 --sd-f 0 --p-drop 0"
    [nobcos]="--s-bcossim 0" [nosupp]="--s-support 0" [nol1f]="--s-l1f 0"
    [nohm]="--s-hm 0")

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

# the head's 12.8M rows of 4 KiB and the .npy header, in 64 MiB reads
dd if=$cache of=/dev/null bs=64M count=$(( (rows * 4096 + 128) / (64 << 20) + 1 )) status=none

mkdir -p $out
for arm in "${arms[@]}"; do
    for a in ste_h76_init01 ste_k128_h54_direct; do
        name=${a}_${wt}_$arm
        /usr/bin/uv run python experiments/adam-eps/train_eps.py $recipe \
            ${arch[$a]} ${flags[$arm]} --out $out/$name \
            >> $logs/recipe_ladder_$name.log 2>&1
    done
done

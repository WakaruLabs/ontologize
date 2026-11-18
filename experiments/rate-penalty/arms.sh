#!/usr/bin/env bash
# The hard stack trained with a rate penalty (README.md): the recipe
# ladder's best hard-stack arm, with every auxiliary weight off
# (ladder.sh's nobcos_nosupp_nol1f_nohm), plus a negative s_Hm, which
# charges each head's batch-mean usage entropy. ARM names the penalty
# -s_Hm; the default queue runs the middle weight first, so a partial queue
# leaves the most informative arm. The control and its seed-43 replica are
# the ladder's runs, at the same schedule: 50,000 steps of 256 over the
# head of mc4_sent16 under the med837 weights.
#
#   arms.sh [ARM ...]
#
# Training waits until the GPU is idle (utilization at most 10% at three
# checks five minutes apart), as ladder.sh does, and runs one arm at a
# time. Every trainer resumes from its own checkpoint, so this runs as a
# systemd unit with Restart=on-failure, and a finished arm is passed over.
# Once every arm is done, ratecurve.py scores them beside the control and
# its replica on the CPU.
set -eu
cd "$(dirname "$0")/../.."
cache=data/sonar_embeddings/mc4_sent16.npy
weights=data/out/sonar/sentcache/mse_weights_med837.npy
rows=12800000
recipe="--temperature 0.00015 --lr 1e-5 --noise-k batchnorm --sd-k 0.3
        --dict-init-scale 0.1 --cache $cache --mse-weights $weights
        --steps 50000 --adam-eps 1e-15 --max-to-keep 1 --checkpoint-each 50000
        --s-bcossim 0 --s-support 0 --s-l1f 0"
out=data/out/sonar/rate_penalty
ladder=data/out/sonar/recipe_ladder
logs=experiments/logs
arms=(2e-8 8e-8 5e-9)
[ $# -eq 0 ] || arms=("$@")

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
runs=()
for arm in "${arms[@]}"; do
    name=ste_h76_init01_med837_rate$arm
    /usr/bin/uv run python experiments/adam-eps/train_eps.py $recipe \
        --s-hm=-$arm --out $out/$name >> $logs/rate_penalty_$name.log 2>&1
    runs+=($out/$name)
done

base=$ladder/ste_h76_init01_med837_nobcos_nosupp_nol1f_nohm
JAX_PLATFORMS=cpu /usr/bin/uv run python experiments/rate-penalty/ratecurve.py \
    --ckpt $base ${base}_s43 "${runs[@]}" --out $out/ratecurve \
    >> $logs/rate_penalty_ratecurve.log 2>&1

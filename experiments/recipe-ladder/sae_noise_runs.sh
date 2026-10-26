#!/usr/bin/env bash
# The SAE half of the recipe ladder (README.md): sae_rate_runs.sh's k32 at
# eps 1e-15 with the Ontologizer's training noises added, one at a time
# and all three together, at the Ontologizer's values (sae.py --sd-pre 0.3,
# --p-drop 0.1, --sd-out 0.1), and a seed-43 replica. The stock run,
# data/out/sonar/sae_sent16/m5120_k32_lr4e-4_no837_whole_eps1e-15, is the
# control: the noises draw from their own stream, so an arm shares its
# initialization and batch order.
#
# It starts once unit sent16-med has finished (active, activating or
# deactivating all count as running, so a restart's gap does not let it
# through), then waits for an idle GPU as full_runs.sh does. sae.py resumes
# from its own state.npz, so this runs as a systemd unit with
# Restart=on-failure, and a finished arm is passed over.
set -eu
cd "$(dirname "$0")/../.."
cache=data/sonar_embeddings/mc4_sent16.npy
weights=data/out/sonar/sentcache/mse_weights_no837.npy
common="--m 5120 --topk 32 --lr 4e-4 --adam-eps 1e-15 --cache $cache
        --mse-weights $weights --block-rows 65536 --buffer-blocks 16
        --epochs 14 --max-steps 144300"
out=data/out/sonar/sae_sent16
logs=experiments/logs

arms=(sdpre0.3 pdrop0.1 sdout0.1 allnoise s43)
declare -A flags=(
    [sdpre0.3]="--sd-pre 0.3" [pdrop0.1]="--p-drop 0.1" [sdout0.1]="--sd-out 0.1"
    [allnoise]="--sd-pre 0.3 --p-drop 0.1 --sd-out 0.1" [s43]="--seed 43")

running() {
    case "$(systemctl --user is-active sent16-med || true)" in
        active|activating|deactivating|reloading) return 0 ;;
        *) return 1 ;;
    esac
}
while running; do sleep 600; done

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

for arm in "${arms[@]}"; do
    name=m5120_k32_lr4e-4_no837_whole_eps1e-15_$arm
    /usr/bin/uv run python sae.py $common ${flags[$arm]} --out $out/$name \
        >> $logs/sae_sent16_$name.log 2>&1
done

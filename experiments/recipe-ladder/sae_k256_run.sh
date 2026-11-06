#!/usr/bin/env bash
# sae_rate_runs.sh's k160 at eps 1e-15 with --topk 256 and nothing else
# changed: m = 11264, lr 1e-4, aux-k 512, 336,000 steps of 4,096 on the
# whole cache under the masked weights, with block-shuffled batches. A
# top-k code's length varies less from row to row than an L1 code's, so
# this tests a code that can spend more of a fixed per-row budget
# (README.md).
#
# It starts once unit recipe-ladder has finished (active, activating or
# deactivating all count as running, so a restart's gap does not let it
# through), so the two never share the GPU, then waits for an idle GPU as
# full_runs.sh does. sae.py resumes from its own state.npz, so this runs as
# a systemd unit with Restart=on-failure.
set -eu
cd "$(dirname "$0")/../.."
cache=data/sonar_embeddings/mc4_sent16.npy
weights=data/out/sonar/sentcache/mse_weights_no837.npy
out=data/out/sonar/sae_sent16
logs=experiments/logs
name=m11264_k256_no837_whole_eps1e-15

running() {
    case "$(systemctl --user is-active recipe-ladder || true)" in
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

/usr/bin/uv run python sae.py --m 11264 --topk 256 --lr 1e-4 --adam-eps 1e-15 \
    --cache $cache --mse-weights $weights --block-rows 65536 --buffer-blocks 16 \
    --epochs 32 --max-steps 336000 --out $out/$name \
    >> $logs/sae_sent16_$name.log 2>&1

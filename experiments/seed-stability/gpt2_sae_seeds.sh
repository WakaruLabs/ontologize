#!/usr/bin/env bash
# A seed pair of the GPT-2 k32 SAE at sae.py's defaults (20 epochs of 4,096,
# lr 1e-4, aux-k 512), seeds 42 and 43, on the current gpt2_l8 harvest and
# its MSE weights. It is the language-model counterpart of the SONAR pair
# sae_conv/m5120_k32{,_s43}: the same SAE recipe on a site where the field
# reports seed agreement, so splitting.py and subspace.py on the two pairs
# say whether the SONAR pair's low agreement is SONAR's or the fit's.
#
# It starts once unit recipe-ladder-nonoise has finished (active,
# activating or deactivating all count as running, so a restart's gap does
# not let it through), so the two never share the GPU, then waits for an
# idle GPU as ladder.sh does. sae.py resumes from its own state.npz, so
# this runs as a systemd unit with Restart=on-failure.
set -eu
cd "$(dirname "$0")/../.."
cache=data/activations/gpt2_l8.npy
weights=data/activations/gpt2_l8.mse_weights.npy
out=data/out/gpt2_l8/sae_seeds
logs=experiments/logs

running() {
    case "$(systemctl --user is-active recipe-ladder-nonoise || true)" in
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

mkdir -p $out
for seed in 42 43; do
    name=m12160_k32
    [ "$seed" = 42 ] || name=${name}_s$seed
    /usr/bin/uv run python sae.py --m 12160 --topk 32 --seed $seed \
        --cache $cache --mse-weights $weights --out $out/$name \
        >> $logs/gpt2_sae_seeds_$name.log 2>&1
done

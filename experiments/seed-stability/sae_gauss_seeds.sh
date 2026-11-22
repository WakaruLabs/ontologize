#!/usr/bin/env bash
# The k32 SAE seed pair on a Gaussian copy of mc4_4M: sae_conv/m5120_k32's
# settings (sae.py's defaults, 150 epochs, 144,300 steps of 4,096) at seeds
# 42 and 43, through sae_gauss.py, on as many rows of the copy as the real
# pair trained on. Then the pair is scored as the real pair was:
# subspace.py --mode sae on the copy's held-out tail, and splitting.py on
# the first 524,288 rows of its training stream (head.npy), as splitting.py
# read the real pair's training rows.
#
# The real pair's unmatched latents span a subspace the other seed
# reproduces at 96-97% of the optimum, about twice the data's top-r
# (notes.md, 2026-11-20). If the Gaussian pair reads the same, that
# agreement is what an SAE does with the covariance; if it reads near the
# data's top-r, the SAE's agreement is the data's beyond its second moments.
#
# It waits for the unit named by AFTER (default width-seeds) to be down at
# three checks five minutes apart, then for an idle GPU as width_seeds.sh
# does. sae.py resumes from its state snapshot and passes over a finished
# run, so it runs as a systemd unit with Restart=on-failure.
#
#   experiments/seed-stability/sae_gauss_seeds.sh
set -eu
cd "$(dirname "$0")/../.."
g=data/out/sonar/gausscopy_4M
out=data/out/sonar/sae_gauss
logs=experiments/logs
after=${AFTER:-width-seeds}

down=0
while [ "$down" -lt 3 ]; do
    case "$(systemctl --user show -p ActiveState --value "$after")" in
        active|activating|deactivating|reloading) down=0 ;;
        *) down=$((down + 1)) ;;
    esac
    if [ "$down" -lt 3 ]; then sleep 300; fi
done

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

for seed in 42 43; do
    name=m5120_k32
    if [ "$seed" != 42 ]; then name=${name}_s$seed; fi
    /usr/bin/uv run python experiments/seed-stability/sae_gauss.py --gauss $g \
        --m 5120 --topk 32 --epochs 150 --seed $seed --out $out/$name \
        >> $logs/sae_gauss_$name.log 2>&1
done

if [ ! -f $g/head.npy ]; then
    /usr/bin/uv run python experiments/seed-stability/gausscopy.py head --out $g --rows 524288
fi
score() {
    local name=$1; shift
    JAX_PLATFORMS=cpu /usr/bin/uv run python "$@" > $out/scores/$name.log 2>&1 \
        || echo "$name failed" >> $out/scores/$name.log
}
mkdir -p $out/scores
a=$out/m5120_k32/params.npz
b=$out/m5120_k32_s43/params.npz
score subspace experiments/seed-stability/subspace.py --mode sae --a $a --b $b \
    --cache $g/eval.npy --out $out/scores/subspace
score splitting splitting.py --a $a --b $b --cache $g/head.npy \
    --out $out/scores/splitting
echo "done $(date '+%F %H:%M')" >> $out/scores/done.log

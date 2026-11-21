#!/usr/bin/env bash
# The recipe ladder's all-auxiliary-weights-off hard stack
# (nobcos_nosupp_nol1f_nohm, with training noise) at seeds 42 and 43, trained
# on a Gaussian copy of mc4_sent16 (gausscopy.py) in place of its rows, then
# scored as the cache pair was: partition NMI, head-contribution cosine,
# per-layer subspaces and the prefix rate curve, on the held-out Gaussian
# file. Agreement the cache pair has beyond this pair's is structure beyond
# the data's second moments.
#
# It prepares the moments and the held-out file if they are missing, waits
# for an idle GPU as ladder.sh does, and trains one seed at a time; the
# trainer resumes from its own checkpoint, so this runs as a systemd unit
# with Restart=on-failure, and a finished seed is passed over. A scorer that
# fails is logged and does not restart the unit.
set -eu
cd "$(dirname "$0")/../.."
out=data/out/sonar/gausscopy
logs=experiments/logs
eval=$out/eval.npy
weights=data/out/sonar/sentcache/mse_weights_med837.npy
[ -f "$out/moments.npz" ] && [ -f "$eval" ] || \
    /usr/bin/uv run python experiments/seed-stability/gausscopy.py prepare \
        >> $logs/gausscopy_prepare.log 2>&1
recipe="--temperature 0.00015 --lr 1e-5 --noise-k batchnorm --sd-k 0.3
        --s-hm 2e-6 --dict-init-scale 0.1 --cache $eval --mse-weights $weights
        --epochs 1 --adam-eps 1e-15 --max-to-keep 1 --checkpoint-each 50000
        --s-bcossim 0 --s-support 0 --s-l1f 0 --s-hm 0"

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

a=$out/alloff
b=$out/alloff_s43
/usr/bin/uv run python experiments/seed-stability/train_gauss.py $recipe \
    --out $a >> $logs/gausscopy_alloff.log 2>&1
/usr/bin/uv run python experiments/seed-stability/train_gauss.py $recipe \
    --seed 43 --out $b >> $logs/gausscopy_alloff_s43.log 2>&1

s=$out/scores
mkdir -p $s
score() {
    local name=$1; shift
    /usr/bin/uv run python "$@" > $s/$name.log 2>&1 || echo "$name failed" >> $s/$name.log
}
score partition experiments/ste-arm/partition.py --a $a --b $b --cache $eval
score headcontrib experiments/ste-arm/headcontrib.py --a $a --b $b --cache $eval \
    --mse-weights $weights
score subspace experiments/seed-stability/subspace.py --mode stack --a $a --b $b \
    --cache $eval --mse-weights $weights --out $s/subspace
score ratecurve experiments/rate-penalty/ratecurve.py --ckpt $a $b --cache $eval \
    --out $s/ratecurve
echo "done $(date '+%F %H:%M')" >> $s/done.log

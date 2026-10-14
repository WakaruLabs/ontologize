#!/usr/bin/env bash
# The pilot's Fano arm at s_pair 1e-3 against the control over the full
# training prefix: one epoch over the pairs of the first 4M rows (about
# 15,400 steps of 256 pairs), with a checkpoint kept every 2,500 steps.
# Then held-out slow-head kappa at each kept checkpoint, and FVU_w at the
# last. The question is whether the slow heads keep growing slower past
# the pilot's 5,000 steps or level off (notes.md). A finished run is
# skipped and an interrupted one resumes from its last checkpoint, so the
# script can be rerun.
set -eu
cd "$(dirname "$0")/../.."
# Allocator: train_ste.py's defaults (on demand), as in pilot.sh; see there
# for reserving a pool on a card nothing else needs.
D=data/out/sonar/next-sentence/long
mkdir -p $D
train() {   # out, term, s_pair
    grep -qs 'Training Complete' $1.log && return 0
    uv run python experiments/next-sentence/train_pair.py --epochs 1 \
        --keep-every 2500 --pair-term $2 --s-pair $3 --out $1 >> $1.log 2>&1
}
train $D/doc2_sp0_ph38 agree 0
train $D/doc2_fano_sp1e-3_ph38 fano 1e-3
runs="$D/doc2_sp0_ph38 $D/doc2_fano_sp1e-3_ph38"
for step in 2500 5000 7500 10000 12500 15000; do
    uv run python experiments/next-sentence/transitions.py --ckpt $runs \
        --step $step --slow-heads 38 --no-concentration \
        --out $D/transitions_$step > $D/transitions_$step.log 2>&1
done
uv run python experiments/next-sentence/eval_next.py --ckpt $runs \
    --out $D/eval.json > $D/eval.log 2>&1

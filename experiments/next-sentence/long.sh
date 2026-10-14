#!/usr/bin/env bash
# The pilot's Fano arm at s_pair 1e-3 against the control over the full
# training prefix: EPOCHS epochs (the argument, default 1) over the pairs of
# the first 4M rows, about 15,420 steps of 256 pairs each, with a checkpoint
# kept every 2,500 steps. Then held-out slow-head kappa at each kept
# checkpoint not yet measured, and FVU_w at the last (eval_e<EPOCHS>.json).
# A larger EPOCHS continues both runs from their last checkpoints. A run
# that has finished EPOCHS epochs leaves <run>.done_e<EPOCHS> and is
# skipped, and an interrupted one resumes, so the script can be rerun.
#
#   experiments/next-sentence/long.sh      # one epoch
#   experiments/next-sentence/long.sh 3    # continue both runs to three
set -eu
cd "$(dirname "$0")/../.."
# Allocator: train_ste.py's defaults (on demand), as in pilot.sh; see there
# for reserving a pool on a card nothing else needs.
E=${1:-1}
D=data/out/sonar/next-sentence/long
mkdir -p $D
train() {   # out, term, s_pair
    [ -e $1.done_e$E ] && return 0
    uv run python experiments/next-sentence/train_pair.py --epochs $E \
        --keep-every 2500 --pair-term $2 --s-pair $3 --out $1 >> $1.log 2>&1
    touch $1.done_e$E
}
train $D/doc2_sp0_ph38 agree 0
train $D/doc2_fano_sp1e-3_ph38 fano 1e-3
runs="$D/doc2_sp0_ph38 $D/doc2_fano_sp1e-3_ph38"
last=$(ls $D/doc2_fano_sp1e-3_ph38 | grep -E '^[0-9]+$' | sort -n | tail -1)
for step in $(seq 2500 2500 $last); do
    [ -e $D/transitions_$step ] && continue
    uv run python experiments/next-sentence/transitions.py --ckpt $runs \
        --step $step --slow-heads 38 --no-concentration \
        --out $D/transitions_$step > $D/transitions_$step.log 2>&1
done
uv run python experiments/next-sentence/eval_next.py --ckpt $runs \
    --out $D/eval_e$E.json > $D/eval_e$E.log 2>&1

#!/usr/bin/env bash
# Pilot sweep of the pair objective (pair_objective.md, "Evaluation" 1):
# the control and a log grid of s_pair around the weight at which the pair
# term pulls on the classifier as hard as the stock loss at initialization
# (train_pair.py --calibrate; notes.md). One seed, one epoch over the pairs
# of the first 1.3M rows (about 5,000 steps of 256 pairs), then held-out
# evaluation of every run. A run with a final checkpoint is skipped, so the
# script can be rerun after an interruption.
set -eu
cd "$(dirname "$0")/../.."
# The card is shared with other users' jobs. train_ste.py's default
# allocator (`platform`, on demand) lets another process take the memory a
# step needs mid-run, which ends the run; reserve a pool up front instead.
# An exported value still wins.
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-bfc}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-true}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}
D=data/out/sonar/next-sentence/pilot
grid="0 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3"
mkdir -p $D
for s in 0 1e-4 3e-4 1e-3 3e-5 1e-5 3e-3; do
    run=$D/doc2_sp${s}_ph38
    [ -e $run/5000 ] && continue
    uv run python experiments/next-sentence/train_pair.py --rows 1300000 \
        --epochs 1 --s-pair $s --out $run > $run.log 2>&1
done
runs=$(for s in $grid; do echo $D/doc2_sp${s}_ph38; done)
uv run python experiments/next-sentence/eval_next.py --ckpt $runs \
    --out $D/eval.json > $D/eval.log 2>&1
uv run python experiments/next-sentence/transitions.py --ckpt $runs \
    --slow-heads 38 --no-concentration --out $D/transitions \
    > $D/transitions.log 2>&1

#!/usr/bin/env bash
# Pilot sweep of the pair objective (pair_objective.md, "Evaluation" 1):
# a log grid of s_pair around the weight at which the pair term pulls on the
# classifier as hard as the stock loss at initialization (train_pair.py
# --calibrate; notes.md), beside the control (s_pair 0, shared by both
# terms). One seed, one epoch over the pairs of the first 1.3M rows (about
# 5,000 steps of 256 pairs), then held-out evaluation of every run. A run
# with a final checkpoint is skipped, so the script can be rerun after an
# interruption.
#
#   experiments/next-sentence/pilot.sh fano     # the default
#   experiments/next-sentence/pilot.sh agree
set -eu
cd "$(dirname "$0")/../.."
# Allocator: train_ste.py's defaults (`platform`, on demand), which share
# the card. On demand, another process can take the memory a step needs
# mid-run and end it; a rerun resumes from the last checkpoint. On a card
# nothing else needs, a reserved pool avoids that:
#   XLA_PYTHON_CLIENT_ALLOCATOR=bfc XLA_PYTHON_CLIENT_PREALLOCATE=true \
#   XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 experiments/next-sentence/pilot.sh
# but the pool is then closed to everyone else, which ends other runs
# sharing the card the same way. Don't reserve one on a shared card. There,
# BFC without preallocation, capped at the same fraction, grows only to
# what a step needs and never past the cap (about 9 GB with CUDA's own;
# these runs fit in it):
#   XLA_PYTHON_CLIENT_ALLOCATOR=bfc XLA_PYTHON_CLIENT_PREALLOCATE=false \
#   XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 experiments/next-sentence/pilot.sh
term=${1:-fano}
case $term in
    agree) tag=""; grid="1e-5 3e-5 1e-4 3e-4 1e-3 3e-3" ;;
    fano) tag="_fano"; grid="1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2" ;;
    *) echo "term: agree or fano, not $term" >&2; exit 2 ;;
esac
D=data/out/sonar/next-sentence/pilot
mkdir -p $D
train() {   # out, s_pair
    [ -e $1/5000 ] && return 0
    uv run python experiments/next-sentence/train_pair.py --rows 1300000 \
        --epochs 1 --pair-term $term --s-pair $2 --out $1 > $1.log 2>&1
}
train $D/doc2_sp0_ph38 0
for s in $grid; do
    train $D/doc2${tag}_sp${s}_ph38 $s
done
runs="$D/doc2_sp0_ph38 $(for s in $grid; do echo $D/doc2${tag}_sp${s}_ph38; done)"
uv run python experiments/next-sentence/eval_next.py --ckpt $runs \
    --out $D/eval${tag}.json > $D/eval${tag}.log 2>&1
uv run python experiments/next-sentence/transitions.py --ckpt $runs \
    --slow-heads 38 --no-concentration --out $D/transitions${tag} \
    > $D/transitions${tag}.log 2>&1

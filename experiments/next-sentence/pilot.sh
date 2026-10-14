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
# The card is shared with other users' jobs. train_ste.py's default
# allocator (`platform`, on demand) lets another process take the memory a
# step needs mid-run, which ends the run; reserve a pool up front instead.
# An exported value still wins.
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-bfc}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-true}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}
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

#!/usr/bin/env bash
# The planted tree under every condition and arm of the README, five
# seeds each, on CPU (about 16 s a seed). Conditions: the tree and its
# shuffle, at child scale 1 and 3. Arms: the two-layer stack, the stack
# without deep supervision, and one layer of two heads; the stack and the
# flat layer also at k = 11, the parent's level count, where a head has no
# entry to spare. A run that has a summary.json is skipped.
#
#   experiments/planted-hierarchy/grid.sh
set -eu
cd "$(dirname "$0")/../.."
T=experiments/planted-hierarchy/tree.py
O=experiments/planted-hierarchy/out
run() {   # name, flags...
    name=$1; shift
    [ -e $O/$name/summary.json ] && return 0
    mkdir -p $O
    uv run python $T --seeds 5 --name $name "$@" > $O/$name.log 2>&1
    grep "^tally" $O/$name.log | sed "s/^/$name: /"
}
for cs in 1 3; do
    for cond in tree shuffled; do
        s=""; [ $cond = shuffled ] && s="--shuffle-tree"
        for k in 16 11; do
            run stack_k${k}_${cond}_cs$cs --k $k --child-scale $cs $s
            run flat_k${k}_${cond}_cs$cs --l 1 --h 2 --k $k --child-scale $cs $s
        done
        run stack_nodeepsup_k16_${cond}_cs$cs --no-deepsup --child-scale $cs $s
    done
done
echo GRID_DONE

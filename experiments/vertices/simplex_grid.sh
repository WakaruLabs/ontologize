#!/usr/bin/env bash
# The planted-simplex toy taken toward the SONAR softmax model one knob at a
# time: which of more heads, more layers, a layer-0 classifier without its
# constant coordinate, and unit-norm rows puts the vertices outside the
# data, as resid_nc's are, where one head on simplex data puts them on the
# hull. The last configuration turns every knob on together.
#
#   experiments/vertices/simplex_grid.sh
#
# ALPHA, SEEDS and ARMS in the environment override the defaults. Each
# configuration writes data/out/vertices/simplex/<name>, the name
# simplex.py gives it, and logs to experiments/logs/. CPU.
set -eu
cd "$(dirname "$0")/../.."
alpha=${ALPHA:-0.2}
seeds=${SEEDS:-5}
arms=${ARMS:-"softmax ste sparsemax_fixed schedule_short"}
mkdir -p experiments/logs

configs=(
    ""
    "--h 4"
    "--h 16"
    "--l 3 --deepsup --resid-norm"
    "--no-const0"
    "--unit"
    "--no-const0 --unit"
    "--h 16 --l 3 --deepsup --resid-norm --no-const0 --unit"
)
for c in "${configs[@]}"; do
    tag=$(echo "base$c" | tr -c 'a-z0-9\n' '_')
    echo "== ${c:-base}" >&2
    # shellcheck disable=SC2086
    /usr/bin/uv run python experiments/vertices/simplex.py --alpha "$alpha" \
        --seeds "$seeds" --arms $arms $c \
        > "experiments/logs/vertices_simplex_grid_a${alpha}_${tag}.log" 2>&1
done

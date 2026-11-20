#!/usr/bin/env bash
# Phase 1's readings once both LCM seeds have trained (plan.md). Gate A is
# already in each run's log.jsonl; this adds:
# - Gate B (stateinput.py) on each seed's blocks 4 and 8;
# - tail-only harvests of each seed and of its untrained init (step 0), at
#   both blocks, into HARVEST_DIR (8 files, about 0.5 GB each);
# - stateagree.py between the two seeds and between the two inits (the
#   floor), per block.
# Outputs go to data/out/lcm-organism/phase1. A harvest already written is
# kept, so a rerun picks up where it stopped.
#
#   experiments/lcm-organism/phase1.sh [HARVEST_DIR]
set -euo pipefail
cd "$(dirname "$0")/../.."
RUNS=data/out/lcm-organism
OUT=$RUNS/phase1
H=${1:-$OUT/tail}
mkdir -p "$OUT" "$H"
py() { uv run python -u "$@"; }

for s in 42 43; do
  [ -e "$OUT/stateinput_s$s.json" ] ||
    py experiments/lcm-organism/stateinput.py --run "$RUNS/lcm_l8_d1024_s$s" --blocks 4 8 \
       --out "$OUT/stateinput_s$s.json"
done
for s in 42 43; do
  for b in 4 8; do
    for st in latest 0; do
      tag=$([ "$st" = 0 ] && echo init || echo trained)
      step=$([ "$st" = 0 ] && echo 0 || echo -1)
      f="$H/s${s}_b${b}_$tag.npy"
      [ -e "$f" ] || py experiments/lcm-organism/harvest_lcm.py --run "$RUNS/lcm_l8_d1024_s$s" \
        --block "$b" --step "$step" --prefix-rows 0 --out "$f"
    done
  done
done
for b in 4 8; do
  for tag in trained init; do
    JAX_PLATFORMS=cpu py experiments/lcm-organism/stateagree.py --a "$H/s42_b${b}_$tag.npy" \
      --b "$H/s43_b${b}_$tag.npy" --out "$OUT/agree_b${b}_$tag.json"
  done
done

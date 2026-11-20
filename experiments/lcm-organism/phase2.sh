#!/usr/bin/env bash
# Phase 2 on one GPU (plan.md): dictionaries on LCM seed 42's block-4 states
# and on its untrained init's, by the SONAR control's recipe as long.sh
# trains doc2_sp0_ph38 (train_pair.py, three epochs over the pairs of the
# first 4M rows, --s-pair 0), at two dictionary seeds each, then scored the
# way the SONAR control is.
#
#   experiments/lcm-organism/phase2.sh harvest [HARVEST_DIR]
#   experiments/lcm-organism/phase2.sh train   [HARVEST_DIR]
#   experiments/lcm-organism/phase2.sh score   [HARVEST_DIR]
#
# - harvest: the two full harvests (rows below 4.4M and the tail, 9.5 GB
#   each) into HARVEST_DIR, default data/activations/lcm-organism (a
#   symlink there puts them on another disk);
# - train: the four runs, into data/out/lcm-organism/phase2. Each keeps only
#   its last checkpoint (--max-to-keep 1 --keep-every 46250), which changes
#   no training, where long.sh kept one every 2,500 steps (15 GB a run);
# - score: seedmatch.py between each arm's two seeds, with within-language
#   NMI on the last 262,144 rows as for the SONAR control (notes.md of
#   next-sentence, 2026-11-10), the same for the SONAR control's seeds 42
#   and 43, and eval_next.py's held-out FVU_w.
# Every step skips what it has already written, and a run that stopped
# resumes from its checkpoint, so a stage can be rerun, and the stages can
# run as a systemd unit with Restart=on-failure.
#
# Before each run and before scoring, it waits for the GPU (where nvidia-smi
# exists): utilization at most 10% and at least GPU_FREE_MIB (default 8192)
# MiB free, at three checks GPU_WAIT_S (default 60) seconds apart. The memory
# matters as much as the load: the trainers allocate on demand, so another
# process's idle server can end a run with an out-of-memory error.
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=${1:?usage: phase2.sh harvest|train|score [HARVEST_DIR]}
H=${2:-data/activations/lcm-organism}
RUNS=data/out/lcm-organism
LCM=$RUNS/lcm_l8_d1024_s42
D=$RUNS/phase2
SONAR=data/out/sonar/next-sentence/long
mkdir -p "$H" "$D"
py() { uv run python -u "$@"; }
step_of() { [ "$1" = init ] && echo 0 || echo -1; }
run_of() { echo "$D/${1}_b4_doc2_sp0_ph38_s$2"; }
wait_gpu() {
  command -v nvidia-smi > /dev/null || return 0
  local calm=0 u f
  while [ "$calm" -lt 3 ]; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$u" -le 10 ] && [ "$f" -ge "${GPU_FREE_MIB:-8192}" ]; then
      calm=$((calm + 1))
    else
      calm=0
      echo "$(date '+%F %T') waiting for the GPU: ${u}% busy, ${f} MiB free"
    fi
    if [ "$calm" -lt 3 ]; then sleep "${GPU_WAIT_S:-60}"; fi
  done
}

case $stage in
  harvest)
    for arm in lcm init; do
      f=$H/${arm}_b4.npy
      [ -e "$f" ] || py experiments/lcm-organism/harvest_lcm.py --run "$LCM" --block 4 \
        --step "$(step_of $arm)" --out "$f"
    done ;;
  train)
    for arm in lcm init; do
      for s in 42 43; do
        out=$(run_of $arm $s)
        [ -e "$out.done_e3" ] && continue
        wait_gpu
        py experiments/next-sentence/train_pair.py --cache "$H/${arm}_b4.npy" \
          --mse-weights "$H/${arm}_b4.mse_weights_matched.npy" --epochs 3 --seed $s \
          --pair-term agree --s-pair 0 --keep-every 46250 --max-to-keep 1 --out "$out" \
          >> "$out.log" 2>&1
        touch "$out.done_e3"
      done
    done ;;
  score)
    wait_gpu
    match() {   # name, a, b, cache, labels
      [ -e "$D/seedmatch_$1.json" ] && return 0
      py experiments/next-sentence/seedmatch.py --a "$2" --b "$3" --cache "$4" \
        --labels "$5" --label-rows 262144 --min-rows 1000 --out "$D/seedmatch_$1.json" \
        > "$D/seedmatch_$1.log" 2>&1
    }
    for arm in lcm init; do
      match ${arm}_b4 "$(run_of $arm 42)" "$(run_of $arm 43)" "$H/${arm}_b4.npy" "$H/${arm}_b4.langs.npy"
      [ -e "$D/eval_${arm}_b4.json" ] ||
        py experiments/next-sentence/eval_next.py --ckpt "$(run_of $arm 42)" "$(run_of $arm 43)" \
          --cache "$H/${arm}_b4.npy" --mse-weights "$H/${arm}_b4.mse_weights_matched.npy" \
          --out "$D/eval_${arm}_b4.json" > "$D/eval_${arm}_b4.log" 2>&1
    done
    match sonar "$SONAR/doc2_sp0_ph38" "$SONAR/doc2_sp0_ph38_s43" \
      data/sonar_embeddings/mc4_sent16.npy data/sonar_embeddings/mc4_sent16.langs.npy ;;
  *) echo "unknown stage $stage"; exit 2 ;;
esac

#!/usr/bin/env bash
# Phase 2 of the comparison suite (after sae_ladder.sh): remaining training
# runs (k5120 finalization, bilinear, seed-stability replica) followed by
# the full eval battery -- pareto, headstruct, splitting (incl. seed
# stability), textfid, langprobe, refit -- over every rung + the
# Ontologizer. Driven by the sae-evals systemd user unit.
#
# Idempotent: every step is guarded by its output artifact, sae.py runs
# resume from snapshots, and the unit restarts on failure -- so crash
# reboots and GPU contention only cost the step in flight. The done marker
# is only written after a fully clean pass.
set -u
cd "$(dirname "$0")"

DONE=data/out/sonar/sae/.evals_done
[ -e "$DONE" ] && { echo "evals already complete ($DONE)"; exit 0; }

SAE=data/out/sonar/sae
ONTO=data/out/sonar/multilingual/resid_nc
FAIL=0

wait_gpu() {
    while :; do
        free=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) \
               - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) ))
        [ "$free" -ge 6000 ] && return
        sleep 120
    done
}

step() {  # step <artifact-guard> <cmd...>
    local guard=$1; shift
    [ -f "$guard" ] && return 0
    echo "EVALS: $*"
    wait_gpu
    "$@" || { echo "EVALS: step failed: $*" >&2; FAIL=1; }
}

# k5120 died at step 22k (already at its FVU floor) before writing
# params.npz; a short resume replays to 22k, evaluates, and finalizes
step $SAE/m11264_k5120/params.npz \
    uv run python sae.py --m 11264 --topk 5120 --epochs 23

step $SAE/m5120_k32_bl/params.npz \
    uv run python sae.py --m 5120 --topk 32 --enc bilinear --lr 4e-4 --epochs 150
step $SAE/m5120_k32_s43/params.npz \
    uv run python sae.py --m 5120 --topk 32 --lr 4e-4 --epochs 150 --seed 43

# cheap and idempotent: refresh every pass so k5120 lands in the table
echo "EVALS: pareto refresh"
wait_gpu
uv run python pareto.py || FAIL=1

step $SAE/m11264_k160/headstruct/groups.csv \
    uv run python headstruct.py --sae $SAE/m11264_k160/params.npz --rows 1048576

SPLIT=data/out/sonar/splitting
step $SPLIT/m5120_k32__m11264_k32/parents.csv \
    uv run python splitting.py --a $SAE/m5120_k32/params.npz --b $SAE/m11264_k32/params.npz
# seed stability: same architecture, seeds 42 vs 43
step $SPLIT/m5120_k32__m5120_k32_s43/parents.csv \
    uv run python splitting.py --a $SAE/m5120_k32/params.npz --b $SAE/m5120_k32_s43/params.npz

RUNS="m5120_k32 m5120_g160top1 m5120_g160softmax m5120_k32_p5 \
      m5120_k32_bl m11264_k32 m11264_k160 m11264_k5120"
for run in $RUNS; do
    step data/out/sonar/textfid/$run/summary.json \
        uv run python textfid.py --model $SAE/$run/params.npz \
            --device cuda --b-decode 64
    step data/out/sonar/langprobe/$run/langs.csv \
        uv run python langprobe.py --model $SAE/$run/params.npz
done
step data/out/sonar/textfid/resid_nc_372600/summary.json \
    uv run python textfid.py --model $ONTO --device cuda --b-decode 64
step data/out/sonar/langprobe/resid_nc/langs.csv \
    uv run python langprobe.py --model $ONTO

step data/out/sonar/refit/refit.csv \
    uv run python refit.py --onto $ONTO

if [ "$FAIL" -eq 0 ]; then
    touch "$DONE"
    echo "EVALS: complete"
else
    echo "EVALS: pass finished with failures; will retry" >&2
    exit 1
fi

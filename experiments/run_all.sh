#!/usr/bin/env bash
# Run the full experiment program in PLAN.md order.
#
# Run FROM THE REPO ROOT on the machine that holds the data:
#     bash experiments/run_all.sh
#
# Behaviour:
#   - phases run in dependency order; each stage logs to experiments/logs/
#   - missing inputs SKIP a stage with a reason instead of dying
#   - a FAIL from verify-mse-weights halts everything (its contract: the
#     whitening weights underpin every other number)
#   - SKIP_LONG=1 skips the GPU-day phases (retrains, seed replica, HSIC
#     arms, scaling sweep); default runs everything, as ordered
#
# Costs are per the experiment READMEs; the long phases are honestly
# multiple GPU-days in total.

set -u

cd "$(dirname "$0")/.." || exit 1
[ -f sonar.py ] || { echo "run from the repo root (sonar.py not found)"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

LOGDIR=experiments/logs
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.txt"
: > "$SUMMARY"

CACHE=data/sonar_embeddings/mc4_4M.npy
WEIGHTS=data/out/sonar/mse_weights.npy
RUN_HM=data/out/sonar/multilingual/resid_nc_hm
RUN_NC=data/out/sonar/multilingual/resid_nc
SKIP_LONG="${SKIP_LONG:-0}"
# The hsic 'aux' arm retrains the live resid_nc_hm configuration for strict
# code-path comparability. The existing reference run already IS that
# configuration, so the arm defaults off to save ~a GPU-day; set
# STRICT_AUX_ARM=1 to train it anyway.
STRICT_AUX_ARM="${STRICT_AUX_ARM:-0}"

note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }

# Disk discipline: a finished run only needs its final checkpoint plus
# loss.csv/log.jsonl for analysis. Prune numbered step dirs down to the
# latest once the run's own diagnostic has consumed the history.
# KEEP_CKPTS=1 disables pruning entirely.
KEEP_CKPTS="${KEEP_CKPTS:-0}"
prune_ckpts() {
  [ "$KEEP_CKPTS" = "1" ] && return 0
  local base="$1" dir keep d
  for dir in "$base" "$base/checkpoints"; do
    [ -d "$dir" ] || continue
    keep=$(ls -1 "$dir" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
    [ -n "$keep" ] || continue
    for d in $(ls -1 "$dir" | grep -E '^[0-9]+$'); do
      [ "$d" != "$keep" ] && rm -rf "${dir:?}/$d"
    done
    note "PRUNE $dir (kept step $keep)"
  done
}

run_stage() {
  # run_stage <name> <log> -- cmd...
  local name="$1" log="$2"; shift 2
  [ "$1" = "--" ] && shift
  note "RUN   $name"
  if "$@" > "$LOGDIR/$log" 2>&1; then
    note "OK    $name"
    return 0
  else
    note "ERROR $name (see $LOGDIR/$log)"
    return 1
  fi
}

skip() { note "SKIP  $1 ($2)"; }

# ---- phase 0: background the CPU/network-bound corpus skip -----------------
# encode_fresh's stream-skip is hours of CPU and network; start it now so
# the GPU never idles waiting for it. Its GPU-encode tail runs when the
# skip completes; we wait for the whole thing before pareto_fresh.

FRESH_PID=""
if [ -f "$CACHE" ]; then
  note "BG    fresh-eval encode (CPU/network skip phase; GPU tail at the end)"
  uv run python experiments/fresh-eval/encode_fresh.py > "$LOGDIR/fresh-encode.log" 2>&1 &
  FRESH_PID=$!
fi

# ---- phase 1: cheap gates --------------------------------------------------

if [ -f "$CACHE" ] && [ -f "$WEIGHTS" ]; then
  if ! run_stage "verify-mse-weights" verify-mse-weights.log -- \
      uv run python experiments/verify-mse-weights/verify_mse_weights.py; then
    if [ -f experiments/verify-mse-weights/ACKNOWLEDGED ]; then
      note "WARN  verify-mse-weights FAILED but explicitly acknowledged (experiments/verify-mse-weights/ACKNOWLEDGED); shipped weights remain authoritative"
    else
      note "HALT  verify-mse-weights FAILED: the whitening weights are wrong or mismatched; nothing downstream is trustworthy. Fix, or record the reason in experiments/verify-mse-weights/ACKNOWLEDGED, before rerunning."
      exit 1
    fi
  fi
else
  skip "verify-mse-weights" "missing $CACHE or $WEIGHTS"
fi

if [ -d "$RUN_NC" ]; then
  run_stage "freeze-diag (resid_nc)" freeze-diag-nc.log -- \
    uv run python experiments/layer4-freeze/freeze_diag.py \
      --ckpt "$RUN_NC" --out experiments/layer4-freeze/diag_resid_nc
else
  skip "freeze-diag (resid_nc)" "missing $RUN_NC"
fi

if [ -d "$RUN_HM" ]; then
  run_stage "freeze-diag (resid_nc_hm)" freeze-diag-hm.log -- \
    uv run python experiments/layer4-freeze/freeze_diag.py \
      --ckpt "$RUN_HM" --out experiments/layer4-freeze/diag_resid_nc_hm
else
  skip "freeze-diag (resid_nc_hm)" "missing $RUN_HM"
fi

# ---- phase 3: the decisive pair -------------------------------------------

if [ -d "$RUN_HM" ] && [ -f "$CACHE" ]; then
  run_stage "steering-overlay" steering-overlay.log -- \
    uv run python experiments/steering-overlay/steer_overlay.py \
      --model "$RUN_HM" --device cuda
  STEERS=$(ls experiments/steering-overlay/out/*/steers.jsonl 2>/dev/null | head -1)
  if [ -n "${STEERS:-}" ]; then
    run_stage "task-naturalness" task-naturalness.log -- \
      uv run python experiments/task-naturalness/naturalness.py \
        --steers "$STEERS" --device cuda
  else
    skip "task-naturalness" "no steers.jsonl produced by the overlay"
  fi
else
  skip "steering-overlay + naturalness" "missing $RUN_HM or $CACHE"
fi

# ---- phase 4: cheap analyses on existing runs ------------------------------

if [ -d "$RUN_HM" ] && [ -f data/.stage_hm_history_done ]; then
  run_stage "devinterp-tracking" devinterp.log -- \
    uv run python experiments/devinterp-tracking/track.py --ckpt "$RUN_HM"
else
  skip "devinterp-tracking" "checkpoint history still syncing; rerun after hm_history marker"
fi

if [ -d "$RUN_NC" ] && [ -f "$CACHE" ]; then
  run_stage "partition-autointerp harvest" headinterp-harvest.log -- \
    uv run python experiments/partition-autointerp/headinterp.py harvest \
      --ckpt "$RUN_NC" --cache "$CACHE"
  run_stage "partition-autointerp texts" headinterp-texts.log -- \
    uv run python autointerp.py texts \
      --features experiments/partition-autointerp/out/heads.npz \
      --out data/out/sonar/autointerp --cache "$CACHE"
  if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    run_stage "partition-autointerp describe" headinterp-describe.log -- \
      uv run python experiments/partition-autointerp/headinterp.py describe --mode both
    run_stage "partition-autointerp score" headinterp-score.log -- \
      uv run python experiments/partition-autointerp/headinterp.py score --null
  else
    skip "partition-autointerp describe/score" "ANTHROPIC_API_KEY not set"
  fi
else
  skip "partition-autointerp" "missing $RUN_NC or $CACHE"
fi

# ---- phase 2 (deferred): the honest eval set -------------------------------
# Collect the backgrounded encode, then run the Pareto against it.

if [ -n "$FRESH_PID" ]; then
  note "WAIT  fresh-eval encode (backgrounded at start)"
  if wait "$FRESH_PID"; then
    note "OK    fresh-eval encode"
  else
    note "ERROR fresh-eval encode (see $LOGDIR/fresh-encode.log)"
  fi
  if [ -f experiments/fresh-eval/mc4_fresh.npy ] && [ -d "$RUN_HM" ]; then
    run_stage "fresh-eval pareto" fresh-pareto.log -- \
      uv run python experiments/fresh-eval/pareto_fresh.py
  else
    skip "fresh-eval pareto" "fresh cache or $RUN_HM missing"
  fi
else
  skip "fresh-eval" "missing $CACHE"
fi

# ---- phase 5: the long runs (GPU-days) -------------------------------------

if [ "$SKIP_LONG" = "1" ]; then
  skip "long phases" "SKIP_LONG=1 (seed replica, retrains, HSIC arms, sweep, RL)"
else
  if [ -f "$CACHE" ] && [ -f "$WEIGHTS" ]; then
    run_stage "seed-stability train (full run)" seed-train.log -- \
      uv run python experiments/seed-stability/train_seed43.py
    if [ -d "${RUN_HM}_s43" ]; then
      run_stage "seed-stability match" seed-match.log -- \
        uv run python experiments/seed-stability/match_heads.py \
          --a "$RUN_HM" --b "${RUN_HM}_s43"
      prune_ckpts "${RUN_HM}_s43"
    else
      skip "seed-stability match" "replica run dir missing"
    fi

    for v in baseline slow_anneal drop_ramp sdk_floor; do
      run_stage "layer4 retrain ($v, full run)" "retrain-$v.log" -- \
        uv run python experiments/layer4-freeze/retrain_variants.py --variant "$v"
      if [ -d "experiments/layer4-freeze/runs/l4fix_$v" ]; then
        run_stage "layer4 diag ($v)" "retrain-diag-$v.log" -- \
          uv run python experiments/layer4-freeze/freeze_diag.py \
            --ckpt "experiments/layer4-freeze/runs/l4fix_$v" \
            --out "experiments/layer4-freeze/diag_$v"
        prune_ckpts "experiments/layer4-freeze/runs/l4fix_$v"
      fi
    done

    HSIC_ARMS="hsic both"
    if [ "$STRICT_AUX_ARM" = "1" ]; then
      HSIC_ARMS="aux $HSIC_ARMS"
    else
      skip "hsic arm (aux)" "existing resid_nc_hm serves as the aux reference; STRICT_AUX_ARM=1 to retrain it"
    fi
    for arm in $HSIC_ARMS; do
      run_stage "hsic arm ($arm, full run)" "hsic-$arm.log" -- \
        uv run python experiments/hsic-bottleneck/train_hsic.py --arm "$arm" --s-hsic-heads 1e-4
      if [ -d "experiments/hsic-bottleneck/runs/$arm" ]; then
        run_stage "hsic head-health ($arm)" "hsic-diag-$arm.log" -- \
          uv run python experiments/layer4-freeze/freeze_diag.py \
            --ckpt "experiments/hsic-bottleneck/runs/$arm" \
            --out "experiments/hsic-bottleneck/diag_$arm"
        prune_ckpts "experiments/hsic-bottleneck/runs/$arm"
      fi
    done

    run_stage "scaling sweep queue (GPU-days)" scaling-queue.log -- \
      bash experiments/scaling-sweep/queue.sh
    run_stage "scaling sweep analysis" scaling-analyze.log -- \
      uv run python experiments/scaling-sweep/analyze.py
    for d in data/out/sonar/multilingual/scaling/*/; do
      [ -d "$d" ] && prune_ckpts "${d%/}"
    done
  else
    skip "long training phases" "missing $CACHE or $WEIGHTS"
  fi

  if [ -d "$RUN_HM" ] && [ -f "$CACHE" ]; then
    run_stage "rl-classifications (embed tier)" rl-train.log -- \
      uv run python experiments/rl-classifications/rl_reinforce.py train \
        --ckpt "$RUN_HM" --cache "$CACHE" --reward embed
  else
    skip "rl-classifications" "missing $RUN_HM or $CACHE"
  fi
fi

# ---- phase 6: needs prerequisites ------------------------------------------

skip "transcoder" "needs a paired target cache (second encode_corpus pass) + target-space mse_weights first; see experiments/transcoder/DESIGN.md"

note "DONE. Full log directory: $LOGDIR/"
echo
echo "==== summary ===="
cat "$SUMMARY"

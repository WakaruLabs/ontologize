#!/bin/bash
# LAPTOP side: one TPU VM for this repository's jobs (README.md).
#   tpu.sh stage code            git archive of HEAD -> $ONTO_BUCKET/stage/code/
#   tpu.sh stage data PATH...    files or directories under data/ -> $ONTO_BUCKET/stage/data/
#   tpu.sh stage jobs DIR        DIR's queue files (q.txt, or q0.txt, q1.txt, ...) -> $ONTO_BUCKET/stage/jobs/,
#                                replacing the staged ones; each is parsed first
#   tpu.sh up                    request the node (flex-start, spot or on-demand) unless it exists
#   tpu.sh wait                  until the node is READY and answers ssh
#   tpu.sh start                 run restart.sh on it: pull code, data and queues, start the runners
#   tpu.sh smoke                 run smoke.sh on it (after start, or RESTART_NO_RUN=1 tpu.sh start)
#   tpu.sh status                the node's state, the runners' status and the jobs' markers
#   tpu.sh ssh [COMMAND...]      a shell, or a command, on the node
#   tpu.sh down [--yes]          delete the node and its queued resource (asks first); the bucket stays
# Settings are read from $ONTO_CLOUD_ENV (default ~/.config/ontologize/cloud.env, outside the
# repository; cloud.env.example lists them). Required: ONTO_PROJECT, ONTO_ZONE, ONTO_BUCKET.
set -euo pipefail
ENVF=${ONTO_CLOUD_ENV:-$HOME/.config/ontologize/cloud.env}
# shellcheck disable=SC1090
[ -f "$ENVF" ] && source "$ENVF"
: "${ONTO_PROJECT:?set ONTO_PROJECT (in $ENVF)}"
: "${ONTO_ZONE:?set ONTO_ZONE (in $ENVF)}"
: "${ONTO_BUCKET:?set ONTO_BUCKET, gs://bucket/prefix (in $ENVF)}"
NODE=${ONTO_NODE:-onto-tpu}
ACC=${ONTO_ACCELERATOR:-v5litepod-4}
PROV=${ONTO_PROVISIONING:-flex}
HOURS=${ONTO_HOURS:-24}
case $ACC in
  v5litepod-*) RT=v2-alpha-tpuv5-lite ;;
  v6e-*) RT=v2-alpha-tpuv6e ;;
  v5p-*) RT=v2-alpha-tpuv5 ;;
  *) RT= ;;
esac
RUNTIME=${ONTO_RUNTIME:-$RT}
: "${RUNTIME:?set ONTO_RUNTIME for accelerator $ACC}"
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
SCOPES=https://www.googleapis.com/auth/cloud-platform
LABELS=purpose=ontologize

T() { local verb=$1; shift; gcloud compute tpus tpu-vm "$verb" "$@" --project="$ONTO_PROJECT" --zone="$ONTO_ZONE"; }
Q() { local verb=$1; shift; gcloud alpha compute tpus queued-resources "$verb" "$@" --project="$ONTO_PROJECT" --zone="$ONTO_ZONE"; }
# a describe that fails for any reason other than "not found" stops the script
describe() {
  local out
  if out=$("$@" 2>&1); then echo "$out"
  elif grep -qiE "not.?found|NOT_FOUND" <<<"$out"; then echo MISSING
  else echo "describe failed: $out" >&2; exit 1; fi
}
node_state() { describe T describe "$NODE" --format='value(state)'; }
qr_state() { describe Q describe "$NODE" --format='value(state.state)'; }
env_for_vm() {
  printf 'ONTO_BUCKET=%q ONTO_RAM_GB=%q ONTO_DATA_SPARE_GB=%q ONTO_PRECISION=%q ONTO_SYNC_EVERY=%q RESTART_NO_RUN=%q RESTART_FORCE=%q' \
    "$ONTO_BUCKET" "${ONTO_RAM_GB:-}" "${ONTO_DATA_SPARE_GB:-10}" "${ONTO_PRECISION:-high}" \
    "${ONTO_SYNC_EVERY:-120}" "${RESTART_NO_RUN:-}" "${RESTART_FORCE:-}"
}

cmd=${1:-}; shift || true
case $cmd in
  stage)
    what=${1:-}; shift || true
    case $what in
      code)
        cd "$REPO"
        git diff --quiet HEAD -- || echo "note: uncommitted changes are not in the archive (git archive HEAD)"
        A=$(mktemp -d)
        name=ontologize-$(git rev-parse --short HEAD)-$(date +%Y%m%dT%H%M%S).tar.gz
        git archive --format=tar.gz -o "$A/$name" HEAD
        gcloud storage cp "$A/$name" "$ONTO_BUCKET/stage/code/$name"
        rm -rf "$A" ;;
      data)
        cd "$REPO"
        [ $# -gt 0 ] || { echo "usage: tpu.sh stage data PATH..."; exit 2; }
        for p in "$@"; do
          # a path under data/ may be a symlink to another disk, a file
          # (f16cache.py --bulk) or a directory: its name under data/ (-s)
          # names the upload, and what it resolves to is read, since rsync
          # skips a source that is itself a symlink. --no-ignore-symlinks
          # follows the ones inside a directory
          rel=$(realpath -s --relative-to="$REPO/data" "$p")
          [[ $rel != ..* && $rel != /* ]] || { echo "$p is not under data/"; exit 2; }
          src=$(realpath "$p")
          if [ -d "$src" ]; then gcloud storage rsync --recursive --no-ignore-symlinks "$src" "$ONTO_BUCKET/stage/data/$rel"
          else gcloud storage cp "$src" "$ONTO_BUCKET/stage/data/$rel"; fi
        done ;;
      jobs)
        dir=${1:?usage: tpu.sh stage jobs DIR}
        shopt -s nullglob; qs=("$dir"/q*.txt)
        [ ${#qs[@]} -gt 0 ] || { echo "no q*.txt in $dir"; exit 2; }
        for q in "${qs[@]}"; do python3 -I "$HERE/runq.py" --check --queue "$q" >/dev/null; done
        A=$(mktemp -d); cp "${qs[@]}" "$A"/
        gcloud storage rsync --recursive --delete-unmatched-destination-objects "$A" "$ONTO_BUCKET/stage/jobs"
        rm -rf "$A" ;;
      *) echo "usage: tpu.sh stage code | data PATH... | jobs DIR"; exit 2 ;;
    esac ;;
  up)
    s=$(node_state)
    if [ "$s" != MISSING ]; then echo "$NODE: $s (tpu.sh down, then up, to recreate it)"; exit 0; fi
    if [ "$PROV" = flex ]; then
      q=$(qr_state)
      if [ "$q" != MISSING ]; then echo "queued resource $NODE: $q (tpu.sh down to clear it)"; exit 0; fi
      Q create "$NODE" --node-id="$NODE" --accelerator-type="$ACC" --runtime-version="$RUNTIME" \
        --provisioning-model=flex-start --max-run-duration="${HOURS}h" --scopes="$SCOPES" --labels="$LABELS"
      echo "requested: $ACC for ${HOURS}h under flex-start; the node is deleted when the time is up"
    else
      SPOT=(); [ "$PROV" = spot ] && SPOT=(--spot)
      [ "$PROV" = spot ] || [ "$PROV" = ondemand ] || { echo "ONTO_PROVISIONING: flex | spot | ondemand"; exit 2; }
      T create "$NODE" --accelerator-type="$ACC" --version="$RUNTIME" "${SPOT[@]}" \
        --scopes="$SCOPES" --labels="$LABELS"
    fi ;;
  wait)
    while true; do
      s=$(node_state)
      [ "$PROV" = flex ] && s="$s (queued resource: $(qr_state))"
      echo "$(date +%T) $NODE: $s"
      [[ $s == READY* ]] && break
      [[ $s == *FAILED* || $s == *SUSPENDED* || $s == PREEMPTED* ]] && { echo "not coming up; tpu.sh down"; exit 1; }
      sleep 60
    done
    for _ in $(seq 1 60); do T ssh "$NODE" --command=true >/dev/null 2>&1 && break; sleep 10; done
    echo "ssh ok" ;;
  start|smoke)
    T scp "$HERE/restart.sh" "$NODE":onto-restart.sh
    T ssh "$NODE" --command="$(env_for_vm) bash onto-restart.sh"
    if [ "$cmd" = smoke ]; then T ssh "$NODE" --command="bash onto/ontologize/experiments/cloud/smoke.sh"; fi ;;
  status)
    echo "$NODE: $(node_state)"
    [ "$PROV" = flex ] && echo "queued resource: $(qr_state)"
    gcloud storage cat "$ONTO_BUCKET/status/*.json" 2>/dev/null || echo "(no runner status yet)"
    gcloud storage cat "$ONTO_BUCKET/queue/*.json" 2>/dev/null || echo "(no job markers yet)" ;;
  ssh)
    if [ $# -gt 0 ]; then T ssh "$NODE" --command="$*"; else T ssh "$NODE"; fi ;;
  down)
    if [ "${1:-}" != --yes ]; then
      read -r -p "delete $NODE in $ONTO_ZONE ($ONTO_PROJECT)? [y/N] " a
      [[ $a == [yY]* ]] || { echo "kept"; exit 0; }
    fi
    if [ "$(qr_state)" != MISSING ]; then Q delete "$NODE" --force --quiet
    elif [ "$(node_state)" != MISSING ]; then T delete "$NODE" --quiet
    else echo "$NODE: nothing to delete"; fi ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac

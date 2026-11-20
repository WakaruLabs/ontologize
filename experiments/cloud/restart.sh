#!/bin/bash
# VM side: an idempotent (re)start on a fresh or recreated VM. Everything is
# pulled from the bucket ($ONTO_BUCKET, a gs:// prefix or a local directory):
#   stage/code/ontologize-<sha>-<time>.tar.gz  git archives (tpu.sh stage code); the newest is used
#   stage/data/...                             -> data/... (tpu.sh stage data)
#   stage/jobs/q<i>.txt or q.txt               queues (tpu.sh stage jobs): q<i> runs on chip i alone,
#                                              q.txt on the whole VM; not both
# then the environment (bootstrap.sh, once), then one runq.py per queue,
# detached. runq.py pulls each job's state from $ONTO_BUCKET/runs, skips
# finished jobs and resumes stopped ones.
# usage on the VM (tpu.sh start copies this here and runs it):
#   ONTO_BUCKET=gs://bucket/prefix bash restart.sh [dir=~/onto]
#   ONTO_RAM_GB=N        keep data/ in a dedicated tmpfs of N GB (not /dev/shm), for caches
#                        the boot disk reads too slowly
#   ONTO_PRECISION=high  JAX matmul precision for jobs that don't set it (high | highest | default)
#   ONTO_SYNC_EVERY=120  seconds between syncs of a running job
#   ONTO_DATA_SPARE_GB=10  GiB data/ must keep free after the pull, for the outputs runs write
#                        (data/out is on the same filesystem)
#   RESTART_NO_RUN=1     inputs and environment only
#   RESTART_FORCE=1      start even though runners are already running
# The pull is refused, before anything is copied, unless what it adds fits:
#   bash restart.sh --fit-check STAGE_DATA DATA SPARE_GB   that check alone, changing nothing
set -euo pipefail
is_gs() { [[ $1 == gs://* ]]; }

# Bytes a pull of stage/data ($1, gs:// or a directory) into data/ ($2) adds:
# each staged file's size, less the size of the copy data/ already holds.
# Files data/ holds that the stage doesn't name stay, so they count as used.
added() {
  { if is_gs "$1"; then
      gcloud storage ls -l "$1/**" | awk -v n=$(( ${#1} + 1 )) \
        '$1 ~ /^[0-9]+$/ && $3 ~ /^gs:/ { u = $0; sub(/^ *[0-9]+ +[^ ]+ +/, "", u); print "S", $1, substr(u, n + 1) }'
    else
      find "$1" -type f -printf 'S %s %P\n'
    fi
    if [ -d "$2" ]; then find -H "$2" -type f -printf 'L %s %P\n'; fi
  } | awk '{ k = $0; sub(/^[SL] [0-9]+ /, "", k) }
           $1 == "L" { have[k] = $2; next }
           { staged[k] = $2 }
           END { for (k in staged) { d = staged[k] - have[k]; if (d > 0) s += d }; printf "%d\n", s }'
}
# Whether that fits beside $3 GiB kept free for outputs; says how much either way
fits() {
  local add avail
  add=$(added "$1" "$2")
  avail=$(df -B1 --output=avail "$2/" | tail -1 | tr -d ' ')
  if [ $(( add + $3 * 1073741824 )) -gt "$avail" ]; then
    awk -v a="$add" -v f="$avail" -v s="$3" 'BEGIN { printf "stage/data does not fit: the pull adds %.1f GiB (%d bytes), %.1f GiB is free under data/, and %d GiB is kept for outputs (ONTO_DATA_SPARE_GB). Prune stage/data or raise ONTO_RAM_GB\n", a / 2^30, a, f / 2^30, s }'
    return 1
  fi
  awk -v a="$add" -v f="$avail" 'BEGIN { printf "stage/data fits: the pull adds %.1f GiB (%d bytes) of %.1f GiB free\n", a / 2^30, a, f / 2^30 }'
}
if [ "${1:-}" = --fit-check ]; then shift; fits "$@"; exit; fi

: "${ONTO_BUCKET:?set ONTO_BUCKET: gs://bucket/prefix, or a local directory}"
RD=${1:-$HOME/onto}
REPO=$RD/ontologize
S=$ONTO_BUCKET/stage
mkdir -p "$RD" "$REPO"
if pgrep -f "experiments/cloud/runq.py" >/dev/null && [ -z "${RESTART_FORCE:-}" ]; then
  echo "runners already running:"; pgrep -af "experiments/cloud/runq.py"; exit 0
fi
# a VM's service account may lack storage.buckets.get, which parallel composite
# uploads need; they then fail with a 403 after small files have gone through
command -v gcloud >/dev/null && gcloud config set storage/parallel_composite_upload_enabled False >/dev/null 2>&1 || true
tree() { mkdir -p "$2"; if is_gs "$1"; then gcloud storage rsync --recursive "$1" "$2"; else rsync -a "$1/" "$2/"; fi; }
getf() { if is_gs "$1"; then gcloud storage cp "$1" "$2"; else cp "$1" "$2"; fi; }

# code: the newest archive
if is_gs "$S"; then
  NEW=$(gcloud storage ls -l "$S/code/*.tar.gz" | grep -v TOTAL | sort -k2 | tail -1 | awk '{print $3}')
else
  NEW=$(ls -t "$S"/code/*.tar.gz 2>/dev/null | head -1)
fi
[ -n "$NEW" ] || { echo "no code archive in $S/code (tpu.sh stage code)"; exit 1; }
if [ "$(cat "$REPO/.code_archive" 2>/dev/null)" != "$NEW" ]; then
  getf "$NEW" "$RD/code.tar.gz"
  tar -xzf "$RD/code.tar.gz" -C "$REPO"
  echo "$NEW" > "$REPO/.code_archive"
  echo "code: $NEW"
fi

# data, optionally in RAM
if [ -n "${ONTO_RAM_GB:-}" ]; then
  if ! mountpoint -q /mnt/onto; then
    # interleaved, so a cache larger than one NUMA node's memory doesn't fill it
    sudo mkdir -p /mnt/onto
    sudo mount -t tmpfs -o "size=${ONTO_RAM_GB}G,mpol=interleave" tmpfs /mnt/onto
    sudo chown "$USER" /mnt/onto
  fi
  sudo loginctl enable-linger "$USER" || true
  mkdir -p /mnt/onto/data
  if [ ! -L "$REPO/data" ]; then
    [ ! -e "$REPO/data" ] || { echo "$REPO/data is on disk; move it or unset ONTO_RAM_GB"; exit 1; }
    ln -s /mnt/onto/data "$REPO/data"
  fi
fi
if is_gs "$S" && gcloud storage ls "$S/data/" >/dev/null 2>&1 || [ -d "$S/data" ]; then
  mkdir -p "$REPO/data/"
  fits "$S/data" "$REPO/data" "${ONTO_DATA_SPARE_GB:-10}"
  tree "$S/data" "$REPO/data"
fi

# queues, which only starting runners needs: a smoke test may have none
rm -rf "$RD/jobs"
if is_gs "$S" && gcloud storage ls "$S/jobs/" >/dev/null 2>&1 || [ -d "$S/jobs" ]; then
  tree "$S/jobs" "$RD/jobs"
fi
shopt -s nullglob
QS=("$RD"/jobs/q*.txt)
if [ -e "$RD/jobs/q.txt" ] && [ ${#QS[@]} -gt 1 ]; then
  echo "q.txt (the whole VM) and per-chip queues would share chips; stage one or the other"; exit 1
fi

cd "$REPO"
if [ ! -f experiments/cloud/env.sh ]; then bash experiments/cloud/bootstrap.sh; fi
source experiments/cloud/env.sh
if [ -n "${RESTART_NO_RUN:-}" ]; then echo "RESTART_READY (inputs and environment; no runners)"; exit 0; fi
[ ${#QS[@]} -gt 0 ] || { echo "no queues in $S/jobs (tpu.sh stage jobs)"; exit 1; }
mkdir -p experiments/logs/cloud
for q in "${QS[@]}"; do
  base=$(basename "$q" .txt)
  chip=${base#q}
  CHIP=()
  if [[ $chip =~ ^[0-9]+$ ]]; then CHIP=(--chip "$chip"); fi
  "$ONTO_PY" -I experiments/cloud/runq.py --check --queue "$q" >/dev/null
  echo "$(date) restart: $base (code $(cat .code_archive))" >> experiments/logs/cloud/restarts.log
  nohup setsid "$ONTO_PY" -u -I experiments/cloud/runq.py --queue "$q" --bucket "$ONTO_BUCKET" \
    "${CHIP[@]}" --precision "${ONTO_PRECISION:-high}" --sync-every "${ONTO_SYNC_EVERY:-120}" \
    >> "experiments/logs/cloud/$base.runner.log" 2>&1 < /dev/null &
  echo "runner $base: pid $!"
done
echo "logs: $REPO/experiments/logs/cloud/"

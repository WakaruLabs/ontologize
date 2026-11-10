#!/usr/bin/env bash
# Another seed of long.sh's two arms, matched against the earlier seeds.
# Waits for an idle GPU (as recipe-ladder/ladder.sh does), trains one epoch
# and continues to three, as seeds 42 and 43 were trained, then runs
# seedmatch.py between the new seed and each of 42 and 43, for the Fano arm
# and the control alike, with the slow pairs' NMI within each language
# (languages with at least 1,000 of the last 262,144 rows, as for 42-43).
# long.sh skips finished runs and resumes interrupted ones, and an existing
# match is skipped, so the script can be rerun. A failed match is logged and
# does not stop the others.
#
#   systemd-run --user --unit replica-s44 -p Restart=on-failure -p RestartSec=120 \
#       --working-directory=$PWD \
#       -p StandardOutput=append:$PWD/experiments/logs/replica_s44.log \
#       -p StandardError=append:$PWD/experiments/logs/replica_s44.log \
#       /usr/bin/bash experiments/next-sentence/replica.sh 44
set -eu
cd "$(dirname "$0")/../.."
S=${1:-44}
D=data/out/sonar/next-sentence/long
LANGS=data/sonar_embeddings/mc4_sent16.langs.npy

# idle: utilization at most 10% at five samples two seconds apart
idle() {
    for _ in 1 2 3 4 5; do
        u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        [ "$u" -le 10 ] || return 1
        sleep 2
    done
}
calm=0
while [ "$calm" -lt 3 ]; do
    if idle; then calm=$((calm + 1)); else calm=0; fi
    if [ "$calm" -lt 3 ]; then sleep 300; fi
done
echo "$(date -Is) seed $S: GPU idle, training"

experiments/next-sentence/long.sh 1 "$S"
experiments/next-sentence/long.sh 3 "$S"

run() {   # arm, seed -> run directory
    if [ "$2" = 42 ]; then echo "$D/doc2_$1_ph38"; else echo "$D/doc2_$1_ph38_s$2"; fi
}
for arm in fano_sp1e-3 sp0; do
    for prev in 42 43; do
        out=$D/seedmatch_${arm}_s${prev}_s$S.json
        [ -e "$out" ] && continue
        uv run python experiments/next-sentence/seedmatch.py \
            --a "$(run $arm $prev)" --b "$(run $arm "$S")" \
            --labels $LANGS --label-rows 262144 --min-rows 1000 --out "$out" \
            > "${out%.json}.log" 2>&1 \
            || echo "$(date -Is) seedmatch $arm s$prev-s$S failed; see ${out%.json}.log"
    done
done
echo "$(date -Is) seed $S: done"

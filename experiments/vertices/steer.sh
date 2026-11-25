#!/usr/bin/env bash
# The decode-based steering comparison (steer.py) on the softmax model: one
# head per row forced to another entry's vertex, its cell's trimmed extreme
# or its centroid, at their own lengths and at matched ones, decoded through
# M2M100 and read back through SONAR and the model.
#
# It waits for the unit named by AFTER (default kmstart-seeds) to be down at
# three checks five minutes apart, then for an idle GPU as kmstart_seeds.sh
# does, and runs steer.py once: JAX on the CPU, M2M100 and SONAR's encoder
# on the GPU. A rerun reads the same rows and makes the same draws, so a
# failed run is started again by hand, as a systemd unit without restarts
# (README.md).
set -eu
cd "$(dirname "$0")/../.."
out=data/out/sonar/vertices/steer
logs=experiments/logs
after=${AFTER:-kmstart-seeds}

down=0
while [ "$down" -lt 3 ]; do
    case "$(systemctl --user show -p ActiveState --value "$after")" in
        active|activating|deactivating|reloading) down=0 ;;
        *) down=$((down + 1)) ;;
    esac
    if [ "$down" -lt 3 ]; then sleep 300; fi
done

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

mkdir -p $out $logs
JAX_PLATFORMS=cpu /usr/bin/uv run python experiments/vertices/steer.py \
    --ckpt data/out/sonar/multilingual/resid_nc --queries 512 \
    --device cuda --out $out >> $logs/vertices_steer.log 2>&1
echo "done $(date '+%F %H:%M')" >> $out/done.log

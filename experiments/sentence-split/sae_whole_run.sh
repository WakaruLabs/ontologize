#!/usr/bin/env bash
# The k32 SAE on the whole of mc4_sent16 under the masked MSE weights
# (`cachestats.py mask`), with block-shuffled batches (`sae.py
# --block-rows`: 672 whole blocks of 65,536 rows, 16 to a shuffle buffer),
# for sae_conv's 144,000 steps of 4,096, 13.4 epochs. Waits until the
# `sent16-full` unit (full_runs.sh's Ontologizers, which read the cache at
# random) has been down for three checks in a row, 15 minutes: that unit
# restarts after a crash within 2 minutes, passing briefly through states
# that are not "active", so a single check can mistake a restart for the
# end. sae.py resumes from its state snapshot, so this runs as a systemd
# unit with Restart=on-failure.
set -u
cd "$(dirname "$0")/../.."
down=0
while [ "$down" -lt 3 ]; do
    state=$(systemctl --user show -p ActiveState --value sent16-full)
    case "$state" in
        active|activating|deactivating|reloading) down=0 ;;
        *) down=$((down + 1)) ;;
    esac
    [ "$down" -lt 3 ] && sleep 300
done
exec /usr/bin/uv run python sae.py --m 5120 --topk 32 \
    --cache data/sonar_embeddings/mc4_sent16.npy \
    --mse-weights data/out/sonar/sentcache/mse_weights_no837.npy \
    --block-rows 65536 --buffer-blocks 16 --epochs 14 --max-steps 144000 \
    --out data/out/sonar/sae_sent16/m5120_k32_no837_whole \
    >> experiments/logs/sae_sent16_m5120_k32_no837_whole.log 2>&1

#!/usr/bin/env bash
# Start encode_sentences.py's encode stage (TF32 matmuls) once the split of
# mc4_sent16 is complete. Waits while the `sent-split` unit is running or
# restarting itself; if the split stops without completing, exits without
# encoding. Run as a systemd unit with Restart=on-failure, so a crash during
# the encode resumes it (the split is complete by then, so the wait passes).
set -u
cd "$(dirname "$0")/../.."
prog=data/sonar_embeddings/mc4_sent16.split.json
until python3 -c "import json, sys; sys.exit(0 if json.load(open('$prog')).get('complete') else 1)" 2>/dev/null; do
    state=$(systemctl --user show -p ActiveState --value sent-split)
    case "$state" in
        active|activating|deactivating|reloading) sleep 60 ;;
        *) echo "split incomplete and sent-split is $state: not encoding"; exit 0 ;;
    esac
done
exec /usr/bin/uv run python encode_sentences.py encode --tf32

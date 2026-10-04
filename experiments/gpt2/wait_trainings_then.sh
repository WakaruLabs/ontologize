#!/bin/bash
# Wait for orphaned training jobs to finish, then run the given queues.
cd "$(dirname "$0")/../.."
while pgrep -f '^\.venv/bin/python experiments/gpt2/train_onto\.py' >/dev/null; do sleep 30; done
exec experiments/gpt2/drive.sh "$@"

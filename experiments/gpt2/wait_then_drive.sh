#!/bin/bash
# Wait for the first driver (started with "q_hard") to finish, then run the given queues.
cd "$(dirname "$0")/../.."
while pgrep -f '^/bin/bash experiments/gpt2/drive\.sh q_hard' >/dev/null; do sleep 30; done
exec experiments/gpt2/drive.sh "$@"

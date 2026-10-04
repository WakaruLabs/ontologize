#!/bin/bash
# Wait for any running drive.sh to finish, then run the given queues.
cd "$(dirname "$0")/../.."
while pgrep -f '^/bin/bash experiments/gpt2/drive\.sh' >/dev/null; do sleep 60; done
exec experiments/gpt2/drive.sh "$@"

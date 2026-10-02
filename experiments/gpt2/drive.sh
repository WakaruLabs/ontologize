#!/bin/bash
# Run job queues sequentially after the SAE baselines finish.
# Usage: drive.sh q_a q_b ...   (files data/out/gpt2/logs/<q>.txt)
cd "$(dirname "$0")/../.."
while pgrep -f '^/bin/bash experiments/gpt2/sae_runs.sh' >/dev/null; do sleep 30; done
for q in "$@"; do
  experiments/gpt2/runq.sh data/out/gpt2/logs/$q.txt > data/out/gpt2/logs/$q.out 2>&1
done
echo DRIVER DONE

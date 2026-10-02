#!/bin/bash
cd "$(dirname "$0")/../.."
L=data/out/gpt2/logs
until grep -q "AFTER QUEUE DONE" $L/after_queue.out 2>/dev/null; do sleep 60; done
exec experiments/gpt2/drive.sh q_private40

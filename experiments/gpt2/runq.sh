#!/bin/bash
# Job queue for the GPT-2 experiments. Usage: runq.sh JOBFILE
# Each line is either "DIAG <run>" (diagnose an existing run) or
# "<name> <train_onto.py args...>". Training jobs run two at a time; each
# pair is diagnosed (diagnose.py --ce) one run at a time after it finishes,
# since JAX keeps its GPU pool and the splice eval needs room for GPT-2.
cd "$(dirname "$0")/../.."
L=data/out/gpt2/logs
diag() { .venv/bin/python experiments/gpt2/diagnose.py data/out/gpt2/$1 --ce > $L/$1.diag.log 2>&1; }
train() { local n=$1; shift
  .venv/bin/python experiments/gpt2/train_onto.py --name $n --overwrite "$@" > $L/$n.log 2>&1; }
pending=()
flush() {
  for p in "${pending[@]}"; do train $p & done; wait
  for p in "${pending[@]}"; do diag ${p%% *}; done
  pending=()
}
while read -r line; do
  [ -z "$line" ] && continue
  if [[ $line == DIAG* ]]; then diag ${line#DIAG }; continue; fi
  pending+=("$line")
  [ ${#pending[@]} -eq 2 ] && flush
done < "$1"
[ ${#pending[@]} -gt 0 ] && flush
echo QUEUE DONE

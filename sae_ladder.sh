#!/usr/bin/env bash
# Converged structural-ablation ladder (see sae.py docstring): all rungs at
# m=5120, b=4096, lr 4e-4, ~145k steps (epochs 150) -- ~4x the optimization
# distance that converged m11264_k32 at lr 1e-4, so each rung ends flat.
#
#   rung 1  m5120_k32           plain top-k baseline (resumes the 48k-step
#                               lr 1e-4 run; remainder trains at 4e-4)
#   rung 2a m5120_g160top1      160 competing groups, hard winner
#   rung 2b m5120_g160softmax   160 competing groups, soft (DictEnc-like)
#   rung 3  m5120_k32_p5        top-k + 5 nested prefix losses (deepsup)
#
# then refreshes pareto.csv and runs headstruct on each rung (trained
# partition for the group runs, discovery for the others).
#
# Run via the sae-ladder systemd user unit: every sae.py run resumes from
# its state.npz, so crash-reboots only cost the interval since the last
# --save-each snapshot. Idempotent: completed runs fast-forward and exit;
# the done marker skips everything on later boots.
set -u
cd "$(dirname "$0")"

DONE=data/out/sonar/sae/.ladder_done
[ -e "$DONE" ] && { echo "ladder already complete ($DONE)"; exit 0; }

wait_gpu() {
    while :; do
        free=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) \
               - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) ))
        [ "$free" -ge 6000 ] && return
        sleep 120
    done
}

run() {
    local tries=0
    until uv run python sae.py --lr 4e-4 --epochs 150 "$@"; do
        tries=$((tries + 1))
        if [ "$tries" -ge 20 ]; then
            echo "LADDER: giving up on: $*" >&2
            return 1
        fi
        echo "LADDER: run failed (GPU contention?); retrying: $*" >&2
        sleep 300
        wait_gpu
    done
}

wait_gpu
run --m 5120 --topk 32                                || true
run --m 5120 --topk 0 --groups 160                    || true
run --m 5120 --topk 0 --groups 160 --group-fn softmax || true
run --m 5120 --topk 32 --prefixes 5                   || true

# evals (idempotent, minutes)
uv run python pareto.py || true
HS="uv run python headstruct.py --rows 1048576"
$HS --sae data/out/sonar/sae/m5120_g160top1/params.npz --trained-groups || true
# softmax latents always fire; 2/32 = "above resting share" (autointerp's
# tag threshold)
$HS --sae data/out/sonar/sae/m5120_g160softmax/params.npz --trained-groups \
    --fire-thr 0.0625 || true
$HS --sae data/out/sonar/sae/m5120_k32/params.npz || true
$HS --sae data/out/sonar/sae/m5120_k32_p5/params.npz || true

touch "$DONE"
echo "LADDER: complete"

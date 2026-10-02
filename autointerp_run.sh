#!/usr/bin/env bash
# The auto-interp campaign (protocol 1 of the interpretability battery):
# harvest -> texts -> describe (plain + contrastive) -> score over the six
# models where the description-mode comparison answers a designed question:
#
#   onto resid_nc      params (E[p]-origin tag decode) vs acts
#   m5120_k32          params (W_dec row decode) vs acts -- the baseline
#   m5120_k32_bl       params vs eig (eigenfeature decode) vs acts
#   m5120_g160top1     params vs acts -- does trained head structure help?
#   m5120_g160softmax  onto's closest competitor; same dense-mixture code
#   m11264_k5120       reconstruction-optimal near-dense control (interp floor)
#
# Every model also gets mode "cacts": the contrastive describe (top texts vs
# random corpus draws), which blocks the degenerate "noisy non-English web
# scrape" answer the plain prompt invites on mC4.
#
# Driven by the autointerp-run systemd user unit. Idempotent: harvests are
# artifact-guarded, texts/describe/score resume from their own done-sets,
# so crash reboots only cost the item in flight. The mC4 streaming (texts)
# is the long pole; judge = haiku via the Anthropic Messages API.
set -u
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
# the judge bills the API key (the unit's EnvironmentFile provides it),
# never the interactive subscription -- autointerp.py calls the Messages
# API directly and refuses to run with the key unset.
# ANTHROPIC_API_KEY may name a key file instead of embedding the key
[ -f "${ANTHROPIC_API_KEY:-}" ] && ANTHROPIC_API_KEY=$(< "$ANTHROPIC_API_KEY")
export ANTHROPIC_API_KEY
[ -n "${ANTHROPIC_API_KEY:-}" ] || {
    echo "AI: ANTHROPIC_API_KEY is empty; the judge cannot run" >&2
    exit 1
}
# judge pacing: plain-prompt-equivalent units per hour across describe/score
# (autointerp.call_weight charges big prompts more). On API billing this only
# smooths the request rate under API limits -- there is no session quota;
# 1000 ~= one plain call per 3.6s shared across score's 4 jobs
RATE=${AI_RATE:-1000}

DONE=data/out/sonar/autointerp/.campaign_done
[ -e "$DONE" ] && { echo "campaign already complete ($DONE)"; exit 0; }

AI=data/out/sonar/autointerp
SAE=data/out/sonar/sae
ONTO=data/out/sonar/multilingual/resid_nc
SAES="m5120_k32 m5120_k32_bl m5120_g160top1 m5120_g160softmax m11264_k5120"
DIRS="onto/onto m5120_k32/sae m5120_k32_bl/sae m5120_g160top1/sae m5120_g160softmax/sae m11264_k5120/sae"
FAIL=0

wait_gpu() {
    while :; do
        free=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) \
               - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) ))
        [ "$free" -ge 6000 ] && return
        sleep 120
    done
}

run() {
    echo "AI: $*"
    "$@" || { echo "AI: step failed: $*" >&2; FAIL=1; }
}

wait_gpu
[ -f $AI/onto/onto/features.npz ] || run uv run python autointerp.py harvest \
    --model onto --ckpt $ONTO --out $AI/onto
for m in $SAES; do
    [ -f $AI/$m/sae/features.npz ] || run uv run python autointerp.py harvest \
        --model sae --ckpt $SAE/$m/params.npz --out $AI/$m
done

# one shared streaming pass recovers text for every model's rows
run uv run python autointerp.py texts --out $AI --features \
    $AI/onto/onto/features.npz \
    $(for m in $SAES; do echo $AI/$m/sae/features.npz; done)

run uv run python autointerp.py describe --dir $AI/onto/onto --out $AI \
    --mode both --device cuda --rate $RATE
for m in $SAES; do
    mode=both; [ "$m" = m5120_k32_bl ] && mode=all
    run uv run python autointerp.py describe --dir $AI/$m/sae --out $AI \
        --mode $mode --device cuda --rate $RATE
done

for d in $DIRS; do
    run uv run python autointerp.py describe --dir $AI/$d --out $AI \
        --mode cacts --rate $RATE
done

for d in $DIRS; do
    run uv run python autointerp.py score --dir $AI/$d --out $AI \
        --judge haiku --jobs 4 --rate $RATE --null --null-n 100
done

if [ "$FAIL" -eq 0 ]; then
    touch "$DONE"
    echo "AI: complete"
else
    echo "AI: pass finished with failures; will retry" >&2
    exit 1
fi

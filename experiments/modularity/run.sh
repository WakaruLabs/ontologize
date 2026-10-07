#!/usr/bin/env bash
# The modularity runs recorded in notes.md: 1M cache rows (half discovery,
# half scoring), 20 size-matched nulls, kNN-15 heat-kernel graphs.
# Existing headstruct/ outputs of the SAE runs are left alone; these
# write to headstruct_mod/.
set -euo pipefail
cd "$(dirname "$0")/../.."

COMMON="--rows 1048576 --nulls 20 --modularity --knn 15"
ONTO=data/out/sonar/multilingual/bl_ctl_full
SAE=data/out/sonar/sae_conv

run() {  # run <out dir> <headstruct args...>
    local out=$1; shift
    mkdir -p "$out"
    uv run python headstruct.py "$@" $COMMON --out "$out" 2>&1 \
        | grep -v -i warn | tee "$out/run.log"
}

# Ontologizer heads, each layer on its own input's graph
run $ONTO/headstruct --onto $ONTO
# matched-shape SAE: 160 trained softmax heads of 32, graph of X
run $SAE/m5120_g160softmax/headstruct_mod \
    --sae $SAE/m5120_g160softmax/params.npz --trained-groups
# flat trained-head SAE: 160 top-1 heads of 32, graph of X
run $SAE/m5120_g160top1/headstruct_mod \
    --sae $SAE/m5120_g160top1/params.npz --trained-groups
# plain top-k SAE, discovered groups, graph of X
run $SAE/m5120_k32/headstruct_mod --sae $SAE/m5120_k32/params.npz
# Matryoshka SAE, prefix blocks as layers on prefix-residual graphs
run $SAE/m5120_k32_p5/headstruct_mod \
    --sae $SAE/m5120_k32_p5/params.npz --prefix-layers
# trained-head Matryoshka SAE: m5120_g160top1 plus 5 nested prefixes,
# trained like the sae_ladder.sh rungs (lr 4e-4, 150 epochs, ~144k steps):
#   uv run python sae.py --m 5120 --topk 0 --groups 160 --prefixes 5 \
#       --lr 4e-4 --epochs 150 --out $SAE/m5120_g160top1_p5
run $SAE/m5120_g160top1_p5/headstruct_mod \
    --sae $SAE/m5120_g160top1_p5/params.npz --trained-groups --prefix-layers

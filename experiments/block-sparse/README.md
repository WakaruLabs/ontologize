# Block-sparse featurizer baseline

The vanilla block-sparse featurizer of Fel et al. (2026), arXiv:2606.25234
(`sae.py --block`; the model is `ontologize/blocksparse.py`), trained on
`mc4_4M` at the settings of the converged top-k baseline
`data/out/sonar/sae_conv/m5120_k32`:
- m = 5,120 latents, batch 4,096, learning rate 1e-4, seed 42;
- 150 epochs (144,300 steps);
- the whitened MSE, with the cache tail held out.

A block is the opposite unit to a head. It is chosen sparsely and varies
continuously inside its span, where a head is always active and names one
of its entries. These runs ask what a code built on that unit costs and
buys on SONAR, at the top-k baseline's coefficient count.

## Runs

Each run sends 32 coefficients per row, as `--topk 32` does, grouped into
blocks of 2, 4 or 8. The support bits name which blocks are active
(`pareto.sae_index_bits`); TopK-32 pays 276.5 to name its 32 of 5,120
latents.

| run | blocks | active | support bits |
|---|---|---|---|
| `m5120_bsf2_k16` | 2,560 of 2 | 16 | 136.8 |
| `m5120_bsf4_k8` | 1,280 of 4 | 8 | 67.2 |
| `m5120_bsf8_k4` | 640 of 8 | 4 | 32.7 |

The paper's description length on DINOv3 was lowest at blocks of 2 to 4;
8 lies past that.

Each run is a systemd user unit that restarts after a crash; `sae.py`
resumes from its state snapshot.

```bash
for spec in "2 16" "4 8" "8 4"; do
  set -- $spec; B=$1; L=$2; name=m5120_bsf${B}_k${L}
  systemd-run --user --unit sae-conv-bsf${B}-k${L} -p Restart=on-failure -p RestartSec=120 \
    --working-directory=$PWD \
    -p StandardOutput=append:$PWD/experiments/logs/sae_conv_${name}.log \
    -p StandardError=append:$PWD/experiments/logs/sae_conv_${name}.log \
    /usr/bin/uv run python sae.py --m 5120 --block $B --topk $L --aux-k 0 --epochs 150 \
      --out data/out/sonar/sae_conv/$name
done
```

## Evaluation

`pareto.py` places the runs beside the converged SAEs: it reads each
run's block size from its `meta.json` and counts the support per block.

```bash
uv run python pareto.py --sae data/out/sonar/sae_conv/m5120_bsf*/params.npz \
    data/out/sonar/sae_conv/m5120_k32/params.npz
```

`quantrate.py` and `refit.py` skip block runs: neither has a block code's
support yet. Every other consumer of `params.npz` raises on one rather than
read `topk` as latents.

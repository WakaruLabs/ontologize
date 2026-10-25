# adam-eps

Does Adam's eps slow the classifiers?

Adam steps by m̂/(√v̂ + ε). For a parameter whose gradient scale √v̂ is well
below ε, the step is about √v̂/ε of a full one, so ε rather than the learning
rate sets how fast that parameter trains. The losses here are small in
absolute terms (a whitened MSE), and the bilinear classifiers start with small
logits, so their gradients can sit orders of magnitude below optax's default
ε = 1e-8 while the dictionaries' do not. Barron, Gnanapragasam & Tschannen
(2026, *A Fix for Softmax and Cross-Entropy*, §5) analyse the same mechanism:
ε decides where gradients stop counting.

## Code

- `train_eps.py` runs `experiments/ste-arm/train_ste.py` with Adam's ε set by
  `--adam-eps`. It hands ε to optax through a subclass of whichever
  `Hyperparams` class `train_ste.py`'s flags choose (`with_eps`; with
  `--dict-init-scale` that class is `DictWDHyperparams`). Every other flag is
  `train_ste.py`'s, so an arm and its control differ in ε alone. `log.jsonl`
  records `adam_eps` with the other hyperparameters.
- `throttle.py` reads the second moments a checkpoint saves with its optimizer
  state and prints, per layer, the median share of a full Adam step the
  classifier and the dictionary take, √v̂/(√v̂ + ε) with the run's own ε. It
  restores arrays as numpy, so it runs on CPU beside a GPU job. Given an
  `sae.py` run directory instead of `RUN:STEP`, it reads the run's
  `state.npz` per parameter, under the ε its `meta.json` records.
- `tests/test_adam_eps.py` checks that ε reaches the optimizer, the throttle
  arithmetic, and the reading of `sae.py`'s saved state.

## Usage

Two arms, the stock ε and a lowered one, on the GPT-2 organism (`ste_h76`'s
configuration, which includes `--dict-init-scale 1.0`), run one after the
other with a small share of the GPU:

```bash
mkdir -p data/out/gpt2_l8/adam_eps
for eps in 1e-8 1e-15; do
  ONTO_MEM_FRACTION=0.1 uv run python experiments/adam-eps/train_eps.py \
      --base gpt2_l8 --dict-init-scale 1.0 --epochs 2 --adam-eps $eps \
      --out data/out/gpt2_l8/adam_eps/ste_h76_eps$eps \
      > data/out/gpt2_l8/adam_eps/ste_h76_eps$eps.log 2>&1 || break
done
```

Compare the arms on:
- held-out whitened FVU (`pareto.py --ckpt <run>`);
- the layer-0 logit scale (`experiments/ste-arm/logitscale.py`);
- the Adam throttle per layer (`gpt2_l8` saves every 250 steps and keeps the
  last ten saves plus every 25,000th step, so a two-epoch arm has 25000 and
  28500-30750):

  ```bash
  JAX_PLATFORMS=cpu uv run python experiments/adam-eps/throttle.py \
      data/out/gpt2_l8/adam_eps/ste_h76_eps1e-8:30750 \
      data/out/gpt2_l8/adam_eps/ste_h76_eps1e-15:30750
  ```

`notes.md` has the results.

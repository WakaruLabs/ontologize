# adam-eps: notes

## 2026-10-24: on GPT-2, the stock ε held the classifiers to 0.1% of a step; at 1e-15 held-out FVU_w falls 6-fold

**Setup.** `ste_h76`'s configuration on the GPT-2 organism (`--base gpt2_l8 --dict-init-scale 1.0`),
two epochs (30,993 steps at batch 256), one seed per arm, run one after the other at
`ONTO_MEM_FRACTION=0.1` (the loop in `README.md`).
- **The two arms** differ in ε alone: optax's default 1e-8 (the control) and 1e-15.
- **The third reference** is the original `data/out/gpt2_l8/ste_h76` (371,900 steps, ε 1e-8).
- **What each arm kept:** checkpoints at 25,000 and 28,500-30,750 (the last ten saves, plus every
  25,000), and `loss.csv` to step 30,750. The last 243 steps came after the last save.
- **Where scores come from:** held-out scores use the 32,890-row whole-document tail that training
  held out. On that tail, the original's final FVU_w reproduces as 0.1934.

**Adam's step share** (`throttle.py`). Each entry is the median √v̂/(√v̂ + ε) per layer, where 1 is a
full step:

| run | step | classifiers, layers 0-4 | dictionaries, layers 0-4 |
|---|---|---|---|
| original | 10,000 | 0.001 0.001 0.000 0.000 0.000 | 0.980 0.968 0.960 0.943 0.899 |
| original | 30,000 | 0.001 0.001 0.000 0.000 0.000 | 0.965 0.952 0.945 0.924 0.864 |
| original | 371,900 | 0.008 0.007 0.002 0.000 0.000 | 0.756 0.695 0.761 0.670 0.551 |
| ε 1e-8 | 30,750 | 0.001 0.001 0.000 0.000 0.000 | 0.964 0.951 0.944 0.924 0.863 |
| ε 1e-15 | 30,750 | 1.000 1.000 1.000 1.000 1.000 | 1.000 1.000 1.000 1.000 1.000 |

**Held-out FVU_w** (`pareto.py --eval-rows 32890 --temperature 1.0`). Under `ste` the soft and hard
codes are the same one-hot code, so one column covers both.

| step | original | ε 1e-8 | ε 1e-15 |
|---|---|---|---|
| 25,000 | | 0.5698 | 0.0944 |
| 30,000 | 0.5369 | 0.5482 | 0.0905 |
| 30,750 | | 0.5467 | **0.0900** |
| 371,900 | 0.1934 | | |

- **The m=1 deviation code** (380 coefficients) at step 30,750 scores 0.6800 in the control and
  0.1757 at 1e-15. The original scores 0.3319 at step 371,900.
- **SAEs on the same rows.** The `sae.py` TopK SAEs score:
  - `m12160_k32`: 0.1874, from 32 coefficients and 317 index bits;
  - `m12160_k64`: 0.1515;
  - `m12160_k128`: 0.1179, from 128 coefficients and 1,020 index bits.
- **The SAEs weren't throttled.** Their Adam step shares, read from `state.npz`, are 0.994-1.000
  for every parameter.

**Training MSE** (`loss.csv` column `MSE`, each value the mean of the 250 steps ending there):

| step | original | ε 1e-8 | ε 1e-15 | 1e-15 / 1e-8 |
|---|---|---|---|---|
| 250 | 1.589e-3 | 1.588e-3 | 1.585e-3 | 0.998 |
| 1,000 | 6.035e-4 | 6.032e-4 | 4.245e-4 | 0.704 |
| 5,000 | 6.161e-4 | 6.153e-4 | 2.122e-4 | 0.345 |
| 10,000 | 5.980e-4 | 5.978e-4 | 1.719e-4 | 0.288 |
| 30,750 | 5.021e-4 | 5.022e-4 | 1.399e-4 | 0.279 |

- **The control flatlines.** It sits near 6e-4 from step 500 to 10,000; the 1e-15 arm falls
  throughout.
- **The original's endpoint.** It reaches 2.04e-4 at step 371,900. The 1e-15 arm is below that from
  about step 5,500.

**Layer-0 logit scale** (`experiments/ste-arm/logitscale.py`, 2,048 tail rows). The closed-form logit
SD at initialization is 0.0026. The original's step-371,900 row comes from the earlier run in
`data/out/gpt2_l8/logitscale`.

| run | step | within-head SD | margin/t | row norm | noise flips |
|---|---|---|---|---|---|
| original | 30,000 | 0.0338 | 0.009 | 1.023 | 0.682 |
| original | 371,900 | 1.175 | 1.445 | 3.268 | 0.012 |
| ε 1e-8 | 30,750 | 0.0352 | 0.010 | 1.024 | 0.673 |
| ε 1e-15 | 30,750 | 1.147 | 1.361 | 8.450 | 0.011 |

**Code use** (`experiments/ste-arm/codeuse.py`, the 32,890-row tail):

| run | step | realized bits of 1,900 | live entries per head | heads under 1.5 bits | layer-0 prefix FVU_w |
|---|---|---|---|---|---|
| original | 30,000 | 1,008 (53.1%) | 11.8 | 32 | 0.837 |
| original | 371,900 | 1,403 (73.8%) | 19.6 | 1 | 0.481 |
| ε 1e-8 | 30,750 | 987 (51.9%) | 11.6 | 33 | 0.832 |
| ε 1e-15 | 30,750 | 1,840 (96.9%) | 31.1 | 0 | 0.411 |

*Reading:*
- **ε, not the learning rate, set how fast the classifiers trained.** A step share of 0.001 at
  ε = 1e-8 puts the classifiers' √v̂ near 1e-11. Under the stock ε, three things follow:
  - **The rows barely move.** After 30,000 steps the classifier rows are within 3% of their
    initial norm.
  - **Noise picks the code.** The logits reach about a thirtieth of the scale the 1e-15 arm reaches,
    and the training logit noise (sd_K 0.02) decides two thirds of the argmaxes.
  - **Most entries go unused.** About two thirds of the entries are dead.

  The dictionaries meanwhile train at nearly the full rate, learning a code whose assignments are
  mostly random.
- **Removing the throttle fixes all of these, and fast.** In 1/12 of the original's steps:
  - its logit scale and noise flips match the original's endpoint;
  - nearly every entry is live;
  - realized bits reach 97% of nominal;
  - held-out FVU_w is less than half the original's final value (0.090 against 0.193).

  On the same rows that is also below TopK-128's 0.118. Whether the hard code wins at matched rate
  depends on what the SAEs' coefficients cost (`quantrate.py`), which hasn't been run on this arm.
- **The stock run slows itself as it converges.** As the loss falls, the gradients shrink with it.
  By step 371,900 the dictionaries are throttled too, at 0.55-0.76.
  - **Why the loss scale matters:** with the loss scaled by c, Adam's update m̂/(√v̂ + ε) equals
    the unscaled update with ε/c. So ε acts as a floor in the loss's own units. The rescaled
    `mse_weights` in `gpt2_l8.py` keep `sonar.py`'s loss weights meaningful, but they also put this
    loss near 6e-4.
- **The rows grew more than the logits did.** The 1e-15 arm's classifier rows are 2.6 times the
  original's final norm, at about the same logit SD.
- **The softmax fix in `classify.py` isn't needed here.** The mean top-1 margin over t is 1.36. That
  is far from where float32 softmax loses gradient, at a margin of about 16.6.

*What this does not settle:*
- **The classifier and the dictionary weren't separated.** ε changed for every parameter: the
  classifiers' share went from 0.001 to 1, the dictionaries' from 0.86-0.96 (at step 30,750) to 1. The classifiers are
  the likely cause, but an ε per parameter group (`optax.multi_transform`) would separate them.
- **One seed per arm.** At step 30,000 the control scores 2.1% above the original (0.5482 against
  0.5369), which is the scale of single-seed differences here. The ε effect is sixfold.
- **Only one lowered ε and one length.** Only 1e-15 was tested, and only for two epochs.
  - **A side effect of 1e-15:** Adam ignores the gradient's scale, so a parameter whose gradient is
    pure noise still takes a step of about the learning rate.
  - **Untested:** an intermediate ε (1e-10, 1e-12), and a run at the original's length.
- **Not run on SONAR.** The stock SONAR runs are less throttled.
  - **`ste_h76_init01`:** its classifiers take 0.66-0.80 of a full step at 369,500.
  - **`resid_nc`:** its classifiers in layers 2-4 drop to 0.005-0.08 from step 190,000 on, and layer
    4's dictionary to 0.07-0.10. That happens just before that run's layer-4 heads froze
    (`experiments/layer4-freeze`).
- **Every GPT-2 Ontologizer result to date was trained under this throttle**, the writeup's
  included.

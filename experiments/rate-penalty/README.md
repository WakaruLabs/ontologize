# rate-penalty

Does the hard stack reach a lower FVU at a given entropy-coded rate if it is
trained on that rate?

The hard stack's code is categorical: each head sends one of k entries. Coded
by each head's own usage, a row then costs the sum over heads of the entropy
of that head's entries (the realized bits `quantrate.py` reports), not the
nominal h·log₂k per layer. Trained only on its error, the stack spends nearly
every nominal bit: the recipe ladder's best hard-stack arm realizes 1,891 of
1,900 (`experiments/recipe-ladder/notes.md`, 2026-11-04). A code trained on
distortion plus λ times its realized rate is entropy-constrained VQ (Chou,
Lookabaugh & Gray 1989), per head. Soft-to-hard VQ (Agustsson et al. 2017)
trains the same trade end to end. The Binary Autoencoder (Cho et al. 2025)
trains a one-bit SAE against a batch entropy of its code
(`scratch/literature/entropy-coding.md`).

## The penalty

`s_Hm` already weighs `KL_m` (`DictBlock.hmean_kl`), which is, per layer and
averaged over heads, log₂k − H(p̄): the batch-mean classification's distance
from uniform. A positive `s_Hm` is a load-balancing bonus. A negative one
charges H(p̄), each head's usage entropy over the batch, which for hard
assignments is its realized bits per row. With h heads the loss gains
λ·Σ_{layer, head} H(p̄) up to a constant, where λ = −`s_Hm`/h, so no code
changes. The flag needs the `=` form, `--s-hm=-2e-8`, since argparse reads
`-2e-8` alone as an option.

**Setting λ.** The stack trains under deep supervision, so its MSE is the
mean over its l = 5 prefixes, and the last layer's bits change only the last
prefix's error. The control's last layer is worth 1.6×10⁻⁴ FVU per realized
bit on the tail (`ratecurve.py`). One unit of FVU is 7.0×10⁻⁶ in loss units
under the `med837` weights (the tail's mean weighted variance). A penalty at
par with the last layer's average bit is therefore
λ = 1.6×10⁻⁴ × 7.0×10⁻⁶ / 5 ≈ 2.2×10⁻¹⁰, or `s_Hm` ≈ −1.7×10⁻⁸. Earlier
layers' bits enter more prefixes and are worth more per bit, so a penalty
near this value bites mainly on the last layers.

## Arms

`arms.sh` trains the ladder's `nobcos_nosupp_nol1f_nohm` recipe (every
auxiliary weight off; the three training noises on; 50,000 steps of 256 over
the head of `mc4_sent16`, `med837` weights, Adam ε 10⁻¹⁵) with
`--s-hm=-2e-8`, `-8e-8` and `-5e-9`, in that order: about 1.2×, 4.7× and 0.3×
par. The control and its seed-43 replica are the ladder's runs of the same
recipe at `s_Hm` = 0. One seed per arm.

## Scoring

`ratecurve.py`: on the tail's 32,765 rows, over the 1023 free coordinates
(the `no837` weights, as `coordfvu.py`), each prefix's FVU against its
realized bits, with the gains pinned to their means in the forward ("pinned",
the code is the entries alone) and as computed ("float", gains unpriced).
The penalty helps if an arm's points sit below the curve the control's
prefixes trace at the same realized bits. `quantrate.py` prices the gains,
for the full code, if an arm warrants it.

## Usage

```bash
systemd-run --user --unit rate-penalty -p Restart=on-failure -p RestartSec=120 \
    --working-directory=$PWD \
    -p StandardOutput=append:$PWD/experiments/logs/rate_penalty.log \
    -p StandardError=append:$PWD/experiments/logs/rate_penalty.log \
    /usr/bin/bash experiments/rate-penalty/arms.sh

JAX_PLATFORMS=cpu uv run python experiments/rate-penalty/ratecurve.py \
    --ckpt <run dirs> --out data/out/sonar/rate_penalty/ratecurve
```

`tests/test_ratecurve.py` checks the realized bits of even, constant and
skewed heads, and the prefix FVU's weighting.

`notes.md` has the results.

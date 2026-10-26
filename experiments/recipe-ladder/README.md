# recipe-ladder

Which parts of the Ontologizer's training recipe matter, taken one at a
time?

The structural ladder (the findings' appendix; `sae.py --groups`,
`--prefixes`) adds Ontologizer structure to an SAE one commitment at a time.
This does the same for the training recipe. The live `mc4_sent16` recipe
has three training noises and four auxiliary weights:
- each is removed from both Ontologizers on its own, against a shared
  control;
- the noises are added to the k32 SAE, one at a time and together.

Why these:
- At a matched rate, the ε 1e-15 SAEs reach the hard stack's FVU at about
  a fifth of its bits (`experiments/sentence-split/notes.md`, 2026-10-25).
- The recipe's weights are fixed in absolute units, and they were not
  rescaled for `mc4_sent16`, whose whitened variance is 1/85 of `mc4_4M`'s.
- Two of the weights, `s_support` and `s_bcossim`, carry the pull that
  makes the direct run's coordinate 837 drift. Two others, `s_bcossim` and
  `s_L1F`, appear in no earlier result.

## The recipe and its SAE analogs

| piece | Ontologizer (live value) | SAE analog (`sae.py`) |
|---|---|---|
| logit noise | `--sd-k 0.3 --noise-k batchnorm`: Gaussian noise of 0.3× each head's logit RMS on the logits before selection (`DictEnc.withStats`) | `--sd-pre`: noise of `sd_pre`× the sample's pre-activation RMS on the pre-activations that choose the top k. The chosen latents keep their clean values |
| winner dropout | `--p-drop 0.1`: per head per sample, the winner is masked and the runner-up chosen (`DictBlock.drop_winners`) | `--p-drop`: each chosen latent, with probability p, is masked and the runner-up takes its place |
| feature noise | `--sd-f 0.1` (`noise_F="featvar"`): noise of 0.1× each coordinate's batch std on each head's output before pooling (`DictBlock.withStats`) | `--sd-out`: the sum over latents of that noise on each latent's output, drawn as Gaussian noise on the reconstruction with variance `sd_out² Σ_j var_b(z_j) W_dec[q, j]²` |
| sample similarity | `s_bcossim` 1e-5 | — |
| support overlap | `s_support` 1e-6 | — |
| output L1 | `s_L1F` 1e-9 | — |
| mean-entropy bonus | `s_Hm` 2e-6 | — |

Under the straight-through estimator the Ontologizer's hard code carries
no logit magnitude. Its logit noise therefore changes only which entry is
chosen, which is why the SAE's chosen latents keep their clean values.

What to expect:
- **Feature noise.** Its scale is stop-gradiented, so for a layer's own
  gradient it is unbiased noise. It acts through Adam's second moment, and
  as augmentation of the residual the layers above classify.
- **The SAE's output noise** has no layers above, so it is the
  pure-variance case.
- **Logit noise and winner dropout** change which entries train, so they
  can move the fit and the collapse.

## Code

- **`experiments/ste-arm/train_ste.py`** takes `--sd-f` and `--s-l1f`
  besides its other noise and weight flags. It also takes `--max-to-keep`
  and `--checkpoint-each` for its checkpoint retention.
  `experiments/adam-eps/train_eps.py` inherits all four.
- **`sae.py --sd-pre/--p-drop/--sd-out`** sets the fields of the same
  names on `ontologize/sae.py`'s `SAE`.
  - `SAE.loss` applies them only when given an rng, so encoding and
    evaluation never see them.
  - They need a flat linear top-k SAE.
  - They draw from their own key, folded with the step. A noised run
    therefore shares the stock run's initialization and batch order, and a
    resume draws what the uninterrupted run would have.
  - Run names add `_sdpre<g>`, `_pdrop<g>` and `_sdout<g>`, and `meta.json`
    records the three.
- **Tests:** `tests/test_sae.py` covers the noises and `tests/test_train_ste.py`
  the flags.
- **`ladder.sh WEIGHTS`** trains the Ontologizer arms.
  - Both architectures: the hard stack, and the direct k = 128 stack.
  - The ε 1e-15 full runs' flags, at `mask_runs.sh`'s schedule (50,000
    steps of 256 on the cache's first 12.8M rows).
  - The arms, in this order: `ctrl`, `ctrl_s43` (a seed-43 replica, for
    the seed-to-seed spread), `nosdk`, `nosdf`, `nopdrop`, `nonoise`,
    `nobcos`, `nosupp`, `nol1f`, `nohm`.
  - `WEIGHTS` is `no837` or `med837` (`cachestats.py mask`). Runs go to
    `data/out/sonar/recipe_ladder/<arch>_<WEIGHTS>_<arm>`, and each keeps
    only its final checkpoint.
- **`sae_noise_runs.sh`** trains the SAE arms: `sae_rate_runs.sh`'s k32 at
  ε 1e-15 with `--sd-pre 0.3`, `--p-drop 0.1`, `--sd-out 0.1`, all three,
  and a seed-43 replica. They go to
  `data/out/sonar/sae_sent16/m5120_k32_lr4e-4_no837_whole_eps1e-15_<arm>`,
  and the stock run of that name is the control. It starts once unit
  `sent16-med` has finished.

## Usage

Each script runs as a systemd unit that waits for an idle GPU and resumes
after a crash:

```bash
systemd-run --user --unit recipe-ladder -p Restart=on-failure -p RestartSec=120 \
    --working-directory=$PWD \
    -p StandardOutput=append:$PWD/experiments/logs/recipe_ladder.log \
    -p StandardError=append:$PWD/experiments/logs/recipe_ladder.log \
    /usr/bin/bash experiments/recipe-ladder/ladder.sh no837
```

`sae_noise_runs.sh` launches the same way, as unit `sae-noise`.

**Disk.** A 50k-step arm at the trainer's default retention takes 12 GB
(direct) or 5.9 GB (hard stack). With only the final checkpoint kept, that
falls to about 1.1 and 0.6 GB, so about 17 GB for the 20 arms.

## Scoring

Each arm is compared with its control, beside the seed pair's difference.
One seed per arm.
- **FVU over the 1023 free coordinates**, whatever weights the arm trained
  under:

  ```bash
  uv run python experiments/ste-arm/coordfvu.py \
      --cache data/sonar_embeddings/mc4_sent16.npy \
      --mse-weights data/out/sonar/sentcache/mse_weights_no837.npy \
      --rows 32765 --coord 837 --models <runs>
  ```
- **Collapse:** `cossim_k_max` and `KL_m` from each run's `loss.csv`.
- **The rate:** `quantrate.py`.
- **The auxiliary arms:** `auxpull.py --coord 837` and
  `coordtrace.py --coord 837` as well.

`notes.md` has the results.

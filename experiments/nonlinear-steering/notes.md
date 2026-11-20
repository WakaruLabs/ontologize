# nonlinear-steering: results

## 2026-11-07: rescaling the layer-8 residual (`rescale.py`): what the gain-shape code can't see costs little at natural scales

**Why.** A gain-shape layer classifies the direction of its input and multiplies its output by the input's
norm. With an unbiased encoder, decoder and classifier (all `gpt2_l8` Ontologizers here), the code is then
exactly invariant to rescaling the whole input. So whatever GPT-2 does differently with c·h than with h, no
code can name. This measures how much that is, against orthogonal steps of the same length, which change the
residual's direction instead of its size.

**Setup.**
- **Data:** 214 held-out windows, the full-length documents of the `gpt2_l8` tail, with 126 scored
  predictions each.
- **Site:** `blocks.8.resid_post`. The hook matches the harvested rows to a relative error of 1.2e-6.
- **Edits:** positions ≥ 1 of every window, either all of them (`all`) or alternate ones (`alt`).
  - `rescale`: h → c·h.
  - Sideways steps of length |c − 1|·|h| orthogonal to h: `orth` along a random direction, `orth_data` along
    another held-out activation (centred).
- **Behaviour:** KL from the unedited next-token distribution (nats per prediction), the change in
  cross-entropy on the actual next token, and top-1 agreement.
- **Codes:** 4,096 held-out rows under the same edits, one random and one data-shaped direction per row
  across c. Seed 0.

**How much the norm varies naturally.** The median |h| is 121.9. Relative to its own window's median, a
position's norm has percentiles 1, 5, 25, 75, 95 and 99 at 0.84, 0.89, 0.96, 1.05, 1.12 and 1.19. Its rank
correlation with position is −0.17. So 90% of positions sit within about ±11% of their window's median, and
c = 0.9 and 1.1 are the natural range.

**Behaviour, every position:**

| c | step | rescale KL | sideways KL, random | sideways KL, data | rescale ÷ random | ΔCE rescale | ΔCE random | top-1 agree, rescale / random |
|---|---|---|---|---|---|---|---|---|
| 0.9 | 0.10 | 0.0029 | 0.0089 | 0.0154 | 0.32 | +0.0046 | +0.0086 | 0.963 / 0.941 |
| 1.1 | 0.10 | 0.0022 | 0.0089 | 0.0154 | 0.25 | +0.0008 | +0.0086 | 0.967 / 0.941 |
| 0.8 | 0.20 | 0.0136 | 0.0361 | 0.0656 | 0.38 | +0.0173 | +0.0344 | 0.922 / 0.883 |
| 1.25 | 0.25 | 0.0116 | 0.0568 | 0.1071 | 0.20 | +0.0087 | +0.0543 | 0.926 / 0.853 |
| 0.67 | 0.33 | 0.0474 | 0.1009 | 0.2050 | 0.47 | +0.0540 | +0.0967 | 0.859 / 0.808 |
| 1.5 | 0.50 | 0.0365 | 0.2433 | 0.6251 | 0.15 | +0.0322 | +0.2350 | 0.872 / 0.714 |
| 0.5 | 0.50 | 0.1652 | 0.2433 | 0.6251 | 0.68 | +0.1742 | +0.2350 | 0.756 / 0.714 |
| 2.0 | 1.00 | 0.1013 | 1.0386 | 3.3547 | 0.10 | +0.0967 | +1.0152 | 0.796 / 0.476 |

- The standard error over windows of the rescale KL is at most 0.0027.
- `alt` gives about half of each effect. Its rescale ÷ random ratios are within 0.05 of these (0.09–0.63
  against 0.10–0.68), so changing a position's norm relative to its neighbours adds nothing visible.

**Codes.** Share of head decisions that change:

| Ontologizer | float floor (× 1.000001) | rescale, c from 0.67 to 1.5 | rescale, c = 0.5 or 2 | sideways 0.1 | sideways 0.2 | sideways 1.0 |
|---|---|---|---|---|---|---|
| `ste_h20_cat128` @371,900 | 0.0007 | 0.0023–0.0030 | 0 | 0.446 | 0.585 | 0.922 |
| `ste_h76` @371,900 | 0.0031 | 0.0211–0.0218 | 0 | 0.376 | 0.466 | 0.731 |
| `ste_h76_sgn` @371,900 | 0.0012 | 0.0086–0.0092 | 0 | 0.424 | 0.520 | 0.789 |
| `ste-sigmoid` @432,900 | 0.0018 | 0.0132–0.0151 | 0 | 0.561 | 0.667 | 0.924 |

The sideways columns are the random direction; data-shaped steps change 1–3 points more.

SAEs, Jaccard distance between a row's active set before and after:

| SAE | rescale 0.9 / 1.1 | rescale 0.8 / 1.25 | rescale 0.5 / 2 | sideways 0.1 | sideways 0.2 / 0.25 | sideways 0.5 / 1.0 |
|---|---|---|---|---|---|---|
| `m12160_k32` | 0.087 / 0.075 | 0.178 / 0.152 | 0.531 / 0.354 | 0.142 | 0.261 / 0.317 | 0.556 / 0.805 |
| `m12160_k128` | 0.115 / 0.093 | 0.241 / 0.188 | 0.646 / 0.392 | 0.213 | 0.376 / 0.442 | 0.667 / 0.844 |
| `m12160_l10.01` | 0.156 / 0.163 | 0.290 / 0.372 | 0.654 / 0.819 | 0.187 | 0.354 / 0.431 | 0.725 / 0.923 |
| `m12160_l10.01_gated` | 0.192 / 0.184 | 0.357 / 0.382 | 0.716 / 0.750 | 0.208 | 0.390 / 0.472 | 0.759 / 0.927 |

**Readings.**
1. **GPT-2 depends far less on the residual's size than on its direction.** In the natural range (c = 0.9,
   1.1), rescaling moves the next-token distribution by KL 0.002–0.003. That's a quarter to a third of a
   random sideways step of the same length (0.009), and a fifth to a seventh of a step toward another real
   activation (0.015). Top-1 agreement is 96–97% against 94%, and cross-entropy rises by 0.001–0.005 against
   0.009.
2. **Shrinking costs more than growing.** Halving the residual costs KL 0.165, two thirds of an equal-length
   sideways step. Doubling it costs 0.101, a tenth of one, although the step is twice as long. One reading:
   block 9 normalizes its input, so a rescale changes how much blocks 9–11's writes weigh against the
   layer-8 state, and shrinking hands them the output. That mechanism is not tested here.
3. **The gain-shape codes are exactly invariant, up to float32 rounding.**
   - Power-of-two scales, which round exactly, flip no decisions.
   - Other scales flip 0.2–2.2%, flat across c, i.e. a rounding effect, not a dependence on scale.
   - Layer 0 never flips, and flips grow with depth, as near-ties tip in the cancelling residuals and cascade.
   - A rescale by 1 + 10⁻⁶ flips 0.07–0.31%.
   - `ste_h76` has the most knife-edge decisions: about 2% flip under rounding alone, so they would flip
     under any perturbation.
4. **SAEs encode scale.** Their encoder biases and top-k competition make the active set follow the norm.
   Rescaling by 0.8 changes 18–36% of it. That is 0.64–0.68 of what a sideways step of the same length
   does for the top-k SAEs, and 0.82–0.92 for the L1 and gated ones (26–39%). The L1 and gated SAEs respond to doubling almost as much as to a sideways step of the same
   length (0.75–0.82 against 0.92–0.93).
5. **So the blind spot is cheap where activations live.** Gain-shape discards a variable GPT-2 barely uses at
   natural scales, while SAEs spend code changes on it. The blind spot costs something only under large
   shrinking (c ≤ 0.67), well outside the 1–99% range of natural norms.

**Caveats.**
- One site and one model.
- Rescales apply to all positions ≥ 1 or to alternate ones, never to one position in isolation.
- Equal length is not equal naturalness, so the two sideways controls bracket the comparison rather than
  pinning it.
- Behaviour here is GPT-2 alone; no dictionary reconstruction is spliced in.
- The code rows are a 4,096-row sample of 214 windows.

```bash
uv run python experiments/nonlinear-steering/rescale.py
```

## 2026-11-07: pilot (`pilot.py`): criteria, fixed before the run

GPT-2 small (124M) is far smaller than the 1.7B–14B models Hewitt used, and his efficacy fell with size
(+0.29 at 1.7B against +0.76 at 14B). So before building the robustness measurements on these
constructions, the pilot asks whether GPT-2 can host them at the dictionaries' site.

The pass rule, coded in `verdict` before any run:
- **Each steer** must reach at least half of its independent control's own-task efficacy (NLL drop on held-out
  windows, nats/token).
- **Its crosstalk**, the mean KL on the other tasks' held-out windows, must be at most a quarter of its own
  efficacy.
- **Its control's** own efficacy must be positive.
- **Intensity is not scaling** passes if all four `shared:<task>` steers pass.
- **Composition is not addition** passes if `v1`, `v2` and `v1 + v2` pass.

## 2026-11-07: pilot, seed 0: both constructions fail, and both fail linearly

**Setup.** As in the README: 1,200 steps, 8 windows of 64 tokens per text per step, Adam at 0.02, KL weight
0.2. There are 4,604, 4,800, 904 and 1,244 training windows for french, code, recipes and sports, and 114–128
held-out windows per text. The run took 17 minutes on the GPU. The site's median norm is 122.8.

**Held-out efficacy**, the NLL drop in nats/token. Rows are steers, columns are texts; the diagonal is each
steer's own task.

| steer | french | code | recipes | sports | neutral |
|---|---|---|---|---|---|
| shared: french (α 0.47) | 0.004 | 0.067 | 0.014 | 0.016 | 0.009 |
| shared: code (α 4.46) | −0.126 | 0.684 | −0.009 | 0.059 | −0.022 |
| shared: recipes (α 1.15) | 0.003 | 0.177 | 0.029 | 0.037 | 0.019 |
| shared: sports (α 1.41) | −0.000 | 0.224 | 0.033 | 0.044 | 0.022 |
| single: french | 0.123 | −0.029 | −0.026 | 0.005 | −0.013 |
| single: code | −0.176 | 0.705 | −0.056 | −0.023 | −0.062 |
| single: recipes | −0.005 | 0.051 | 0.094 | 0.094 | 0.053 |
| single: sports | −0.004 | 0.067 | 0.051 | 0.185 | 0.074 |
| v1 (→ french) | 0.081 | 0.237 | −0.056 | −0.025 | −0.040 |
| v2 (→ recipes) | −0.126 | 0.387 | 0.069 | 0.051 | 0.020 |
| v1 + v2 (→ code) | −0.057 | 0.684 | −0.012 | 0.018 | −0.035 |

**Against the criteria:**
- shared: french reaches 3% of its control's efficacy, recipes 31%, sports 24%. shared: code reaches 97%, with
  crosstalk at 23% of its efficacy.
- v1 reaches 66% and v2 73%, but their crosstalk (KL 0.066 and 0.100) is three to six times the bound.
- v1 + v2 reaches 97% of the code control.
- Intensity: fail. Composition: fail.

The scales started at 1, 3, 5 and 7 and ended at 0.47, 4.46, 1.15 and 1.41. The code scale rose from the
first steps and was the largest by step 200, while the others drifted down.

**Readings.**
1. **The shared direction is the code direction, and its scale works as intensity.** It has cosine 0.92 with
   the independent code vector, and the other three controls sit at 0.29–0.33. Its smaller scales mostly make
   code text likelier (+0.18 and +0.22 for the recipe and sports scales) rather than steering their own tasks.
   The search found the linear solution: one behaviour, graded by scale.
2. **The composition is additive.** v1 and v2 each carry part of the code direction: their projections onto
   the code control's unit vector are 24.0 and 29.2, which sum to 53.2 against that control's length of 55.1.
   Their separate code effects (+0.24 and +0.39) add to 91% of their sum's (+0.68). Each also carries some of
   its own task (cosine 0.67 with the French control, 0.49 with recipes). In the construction Hewitt describes,
   neither part does the sum's task; here each does a share of it.
3. **The task set is weak, and the objective is unbalanced.** These are confounds, so the pilot does not show
   that GPT-2 small can't host the constructions.
   - The recipe and sports controls point in similar directions (cosine 0.60). The recipe vector helps sports
     text exactly as much as recipe text (0.094 each), and both help neutral C4 text (+0.05, +0.07). Much of
     what they learn adapts GPT-2 from WebText to C4 rather than to a topic.
   - Under the crosstalk rule, the controls for French (KL 0.051 against a bound of 0.031), recipes (0.047
     against 0.024) and sports (0.059 against 0.046) would fail too. Only code passes, narrowly (0.156 against
     0.176).
   - Code's attainable NLL drop (0.7 nats/token) is six times the others', so the summed NLL objective pulls
     the shared direction toward code first.
   - Crosstalk here is KL on other tasks' text, which counts any distribution change. That is stricter than
     Hewitt's behaviour-score crosstalk. It doesn't decide the verdict, though: the intensity steers fail on
     efficacy alone, and v1 and v2 also fail on that score, since each alone makes code text likelier.

**Next.** Fix the task set and the balance before concluding anything about GPT-2:
- screen candidate behaviours by training single-task controls first (four minutes each), and keep only those
  whose controls are selective, with more distinct candidates such as German, digits and tables, markup, and
  dialogue;
- weight each task's NLL by the inverse of its control's efficacy;
- run two or three seeds and starting scales, since the search has its own biases.

If balanced, selective tasks still give linear solutions, the robustness study needs a larger model with
dictionaries.

```bash
uv run python experiments/nonlinear-steering/pilot.py --seed 0
```

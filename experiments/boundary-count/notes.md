# boundary-count: notes

## 2026-11-11: the step that holds other heads' margins

Three hard-code models at their last checkpoints: the binary stack
`ste_k2_h380` (5 × 380, k = 2, step 739,000), the 32-way stack
`ste_h76_init01` (5 × 76, k = 32, step 369,500) and the flat arm
`ste_l1_h380_i01` (1 × 380, k = 32, step 369,500). 128 rows of the
`mc4_4M` tail, one random target head per layer per row, so 128 targets
per layer. One seed per model. The target entry is the head's runner-up,
the easiest one to reach; a step to an arbitrary entry is longer and is
not measured here.

```bash
for m in ste_k2_h380 ste_h76_init01 ste_l1_h380_i01; do
  uv run python experiments/boundary-count/nullstep.py \
      --model data/out/sonar/multilingual/$m --rows 128
done
```

The frozen forward reproduces the module's logits to a relative error of
2e-7, 6e-6 and 4e-7. Selections differ between two evaluations of one
input in differently sized batches for about 0.1% of heads on the flat
arm (the `do` row's in-layer 0.001, which is zero by construction), so
0.001 is the floor of every share below.

### Whether an exact step exists

The count decides it, as predicted. Share of the target's gradient the
constraints leave free (median), and how often the step exists:

| constraint set | binary stack | 32-way stack | flat |
|---|---|---|---|
| the layer's other heads, every margin | 379 margins: 0.61–0.98, always | 2,325: never | 11,749: never |
| the layer's other heads, runner-up margins | the same set | 75: 0.93–0.95, always | 379: 0.62, always |
| the layer and those below, runner-up | layer 1 (759): 0.07, always; layers 2–3 (1,139; 1,519): never | 151–303: 0.71–0.87, always | — |
| the whole stack, runner-up | 1,899: never | 379: 0.63–0.64, always | the layer's set |

### What a single flip costs before any step

`do` sets the target head to its runner-up and touches nothing else. The
layers above reclassify what it writes. Share of the stack's other heads
that change, by the target's layer:

| target layer | binary stack | 32-way stack |
|---|---|---|
| 0 | 0.064 | 0.338 |
| 1 | 0.033 | 0.184 |
| 2 | 0.014 | 0.080 |
| 3 | 0.004 | 0.021 |
| 4 | 0.000 | 0.000 |

This is the collateral any input step that realizes the flip must cause.
One layer-0 head's flip in the 32-way stack changes a third of the other
379 heads; in the binary stack, where a layer has five times as many
heads each writing less, 6%.

### Steps at their own length

Each cell is `col_stack`, the share of the stack's other heads that
changed, with the step's length as a share of |x| and its hit rate in
brackets. "Random" is a random step of the constrained step's length.

**32-way stack.**

| target layer | `do` | `grad` | `null_stack` | `soft_stack` | random |
|---|---|---|---|---|---|
| 0 | 0.338 | 0.364 (0.017; 1.00) | 0.342 (0.021; 1.00) | 0.335 (0.023; 1.00) | 0.288 |
| 1 | 0.184 | 0.273 (0.009; 0.75) | 0.195 (0.011; 0.99) | 0.187 (0.012; 0.99) | 0.251 |
| 2 | 0.080 | 0.220 (0.006; 0.57) | 0.109 (0.008; 0.94) | 0.096 (0.009; 0.96) | 0.232 |
| 3 | 0.021 | 0.205 (0.006; 0.45) | 0.067 (0.007; 0.89) | 0.053 (0.008; 0.95) | 0.221 |
| 4 | 0.000 | 0.167 (0.005; 0.39) | 0.050 (0.006; 0.84) | 0.040 (0.006; 0.82) | 0.200 |

**Binary stack.** No exact step holds the whole stack, so the weighted
step stands in; it is several times longer than the plain crossing.

| target layer | `do` | `grad` | `soft_stack` | random |
|---|---|---|---|---|
| 0 | 0.064 | 0.084 (0.046; 1.00) | 0.081 (0.185; 1.00) | 0.142 |
| 1 | 0.033 | 0.079 (0.028; 0.65) | 0.048 (0.102; 0.98) | 0.111 |
| 2 | 0.014 | 0.074 (0.022; 0.47) | 0.030 (0.068; 0.99) | 0.098 |
| 3 | 0.004 | 0.068 (0.018; 0.26) | 0.019 (0.051; 0.95) | 0.091 |
| 4 | 0.000 | 0.058 (0.012; 0.27) | 0.014 (0.034; 0.84) | 0.077 |

`null_layer` at layer 0 has an in-layer collateral of 0.000 (random:
0.017) and a stack collateral of 0.074. `null_upto` at layer 1 holds
layers 0–1: in-layer 0.007 (random: 0.073), stack 0.059.

**Flat arm** (in-layer collateral; there is no stack).

| step | length | hit | collateral | random of that length |
|---|---|---|---|---|
| `grad` | 0.010 | 1.00 | 0.019 | 0.018 |
| `null_ru` | 0.013 | 1.00 | 0.003 | 0.023 |
| `soft_layer` | 0.014 | 1.00 | 0.001 | 0.025 |

### Steps at 0.25 |x|

The strength `steerembed.py` reports at. `col_stack`, with a random step
of that length beside it, averaged over target layers:

| step | binary stack | 32-way stack | flat |
|---|---|---|---|
| `grad` | 0.176 / 0.171 | 0.652 / 0.655 | 0.404 / 0.390 |
| `null_stack` (runner-up) | — | 0.613 / 0.659 | 0.288 / 0.385 |
| `soft_stack` | 0.083 / 0.171 | 0.592 / 0.657 | 0.156 / 0.385 |

### Reading

- **The shortest step that flips a head is about 1% of the input's
  length**: 0.5–1.7% on the 32-way stack, 1.0% on the flat arm, 1.2–4.6%
  on the binary stack. The 0.25 at which collateral is usually reported
  is 15 to 50 times that.
- **The gradient step is no better than a random one**, at its own length
  and at 0.25 (0.176 against 0.171; 0.652 against 0.655; 0.404 against
  0.390). This reproduces `steerembed.py`'s result, 17–19% and 66%,
  through a different script.
- **A step chosen against the other heads' margins is much better.** At
  its own length its collateral sits close to the intervention's own:
  within 0.05 of `do` at every layer of the 32-way stack, where the
  gradient step adds up to 0.18, and within 0.02 on the binary stack,
  where it adds up to 0.06. On the flat arm it is 0.001–0.003 against
  0.02. So collateral
  is not blind to direction. The directions tried before were not chosen
  with the other heads in view.
- **Most of what remains in a stack is the cascade of the flip itself.**
  For a layer-0 target in the 32-way stack every step, however chosen,
  changes a third of the stack, because the intervention does.
- **The count decides existence and little else.** Holding every margin
  of a layer is possible only on the binary arm, as the count says. But
  holding only each head's nearest boundary is always possible on the
  32-way arms and works about as well at the step's own length. The
  count does not explain why the binary stack is the hardest to disturb:
  that is its smaller cascade (0.064 against 0.338 for a layer-0 flip)
  and its lower first-order flip rate at a given length (`lin_layer` for
  a random 0.25 step: 0.135, against 0.458 and 0.385).
- **At 0.25 the constrained directions lose most of their advantage on
  the 32-way stack** (0.59–0.61 against 0.66) and keep it on the flat arm
  (0.16 against 0.39) and the binary stack (0.08 against 0.17). A step
  that long is far outside the range where the margins move linearly.
- **Hit rates fall with depth for the unconstrained step** (1.00 at layer
  0, 0.39 and 0.27 at layer 4) because the layers below flip first and
  change the target layer's input. Holding their margins restores it
  (0.84–1.00).

### Limits

- The target is the runner-up entry. Steering to a chosen entry is the
  case a steering claim needs, and it is longer.
- The constrained step needs every head's margin Jacobian at the row. It
  is a measurement of what the geometry allows, not a direction the
  model's parameters offer by themselves.
- One seed per model, 128 targets per layer. The binary arm was trained
  at half the batch size of the others.
- `lin_layer` uses the frozen-code Jacobian of the target's layer, so for
  targets above layer 0 it ignores flips below.

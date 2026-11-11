# boundary-count: is collateral a matter of direction?

`experiments/ste-arm/steerembed.py` steps an input along a direction and
counts the other heads that change. Every direction it tries flips them at
a random step's rate. None of those directions is chosen with the other
heads in view, so that result leaves open whether a better direction
exists. This experiment builds it.

## The question as a count

A head with `k` entries has `k - 1` margins, its winner's logit minus each
other entry's, and changes when one of them crosses zero. Between flips
every margin is a smooth function of the input, so holding `c` of them
fixed is `c` linear constraints on a `d`-dimensional step. A step that
moves the target head and holds the rest exists, to first order, only
while `c < d`:

| margins to hold | 5 × 76, k = 32 | 5 × 380, k = 2 | 1 × 380, k = 32 |
|---|---|---|---|
| one layer, all | 2,325 | 379 | 11,749 |
| one layer, runner-up only | 75 | 379 | 379 |
| the stack, runner-up only | 379 | 1,899 | 379 |
| dimensions | 1,024 | 1,024 | 1,024 |

## Code

`nullstep.py` does the whole experiment for one checkpoint.

- **The Jacobian.** `frozen_logits` is the hard forward with every code
  held at its clean value, as a differentiable function of the input. It
  is rebuilt from the module's parts because the module stops the gradient
  at each layer's residual input and at its gain, and a step moves both
  whether or not a code changes. `main` checks it against the module's own
  logits before using it, and refuses a model it does not rebuild
  (`check_model`: residual forward with gain and the constant coordinate,
  hard selection, no router, no fibers).
- **The steps** (`steps_for`, on `constrained_step` and `weighted_step`).
  The target is a head's runner-up entry. `grad` is the shortest step that
  crosses that boundary. `null_layer`, `null_ru`, `null_upto` and
  `null_stack` are the shortest that also hold a set of other margins
  exactly; a set with at least `d` margins is skipped. `soft_layer` and
  `soft_stack` minimize the sum of squared relative margin changes
  instead, so they exist for any set.
- **Scoring.** Each step is applied, the input renormalized to its length
  as `steerembed.py` does, and the model rerun. A random step of the same
  length is scored beside every step. `lin_layer` is what the first-order
  model predicts, so the gap between it and `col_layer` is curvature plus
  the cascade.

Outputs go to `data/out/sonar/boundary/<run>/`: `steps.csv`, one row per
row, target, step and strength, and `summary.json`.

```bash
uv run python experiments/boundary-count/nullstep.py \
    --model data/out/sonar/multilingual/ste_k2_h380
```

The pure helpers are tested in `tests/test_boundary_count.py`, with the
frozen forward's Jacobian checked against finite differences above
layer 0.

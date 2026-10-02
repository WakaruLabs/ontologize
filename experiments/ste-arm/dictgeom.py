"""Exact within-head row geometry, read from the dictionary weights.

Use this rather than the logged `cossim_k` whenever an arm predates the
current stats row. `rowcos` has itself changed: for the 11-column-era
sweep arms the logged column is not today's statistic, and two of them
exceed 1 after the per-layer divide, so it is not a mean cosine there at
all. For arms on the 18-wide row the two agree to four decimals, which is
what validates reading the cheap column for those.

Needs a GPU: the checkpoints are sharded, so a CPU restore raises a
topology mismatch. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.10` is enough and
leaves a training run undisturbed.

Reports, per arm, averaged over heads and layers:
  c        mean within-head pairwise row cosine
  eff      participation ratio of the Gram's eigenvalues, (sum L)^2 / sum L^2
  rank99   singular values needed for 99% of the squared Frobenius norm
Both rank measures run on the rows as the forward pass sees them: `abs`,
then unit-normalized, so `c` is the quantity the orthant floor bounds.
Normalizing discards the row norms, so `eff` is directional diversity
only, while `DictBlock.fwd` uses the unnormalized `dicts()`.

Two things about `eff` that are easy to get wrong. Unit rows make
`sum L = k`, so `eff = k^2 / ||G||_F^2 = k / (1 + (k-1) RMS(c)^2)`: it
reads the mean SQUARE cosine, and a dictionary with `c ~ 0` but spread
is not orthogonal. The signed arms sit at `c = 0.0009` with RMS 0.0738,
which is exactly their eff of 27.4 rather than 32.

And the per-layer spread is wide enough to qualify any aggregate: in a
`resid` cascade collinearity falls monotonically with depth (abs/1536
runs eff 2.95, 4.24, 9.45, 11.60, 17.50 for a mean of 9.15), so arms
that differ on the mean can overlap layer for layer. The closed form
above matches an arm only when its layers are homogeneous.
"""
import sys
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp

from ontologize.training.serialize import migrate_spec

ROOT = Path("/home/keira/flock/ontologize/data/out/sonar/multilingual")


BASE_MSE = 0.000593134   # whitened target variance, SONAR and gpt2_l8 alike


def arm_path(arm):
    return Path(arm).resolve() if "/" in arm else (ROOT / arm).resolve()


def steps(arm):
    return sorted(int(d.name) for d in arm_path(arm).iterdir()
                  if d.name.isdigit())


def fvu(arm, step=None, win=200):
    """Windowed train FVU_w at `step` (default: the end of the log)."""
    a = np.loadtxt(arm_path(arm) / "loss.csv", delimiter=",", usecols=1,
                   ndmin=1)
    end = len(a) if step is None else min(step, len(a))
    return float(a[max(0, end - win):end].mean()) / BASE_MSE


def step_at_fvu(arm, target, win=200):
    """The checkpointed step whose windowed FVU is CLOSEST to `target`, or
    None if the arm never reaches it. Matching on FVU rather than on step
    count is what makes arms of different selection rules comparable: they
    converge at very different rates. Closest rather than first-past
    because checkpoints are 10k apart and a fast arm overshoots a target by
    more than the spread being measured."""
    a = np.loadtxt(arm_path(arm) / "loss.csv", delimiter=",", usecols=1,
                   ndmin=1) / BASE_MSE
    sm = np.convolve(a, np.ones(win) / win, "valid")
    if not (sm <= target).any():
        return None
    return min(steps(arm), key=lambda s: abs(fvu(arm, s, win) - target))


def load_dicts(arm, step=None):
    """Returns the per-layer (h, k, e_dec) dictionaries and the spec."""
    p = arm_path(arm)
    man = ocp.CheckpointManager(
        p, checkpointers={"state": ocp.PyTreeCheckpointer(),
                          "spec": ocp.PyTreeCheckpointer()})
    step = step or man.latest_step()
    spec = migrate_spec(man.restore(step, items={"spec": None})["spec"])
    state = man.restore(step, items={"state": None})["state"]
    params = state["params"] if "opt_state" in state else state
    while "params" in params:
        params = params["params"]
    Ws = []
    for i in range(spec["l"]):
        Ws.append(np.asarray(params[f"dictencs_{i}"]["dict"]["weights"]))
    return Ws, spec, step


def geometry(W, signed=False):
    """(c, eff_rank, rank99) for one layer's (h, k, d) dictionary."""
    A = W if signed else np.abs(W)
    A = A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)
    h, k, _ = A.shape
    G = np.einsum("hkd,hjd->hkj", A, A)
    off = G[:, ~np.eye(k, dtype=bool)].mean()
    lam = np.linalg.eigvalsh(G).clip(0)                 # (h, k)
    eff = (lam.sum(-1) ** 2 / (lam ** 2).sum(-1)).mean()
    cum = np.cumsum(np.sort(lam, -1)[:, ::-1], -1) / lam.sum(-1, keepdims=True)
    rank99 = (cum < 0.99).sum(-1).mean() + 1
    return float(off), float(eff), float(rank99)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = next((float(a.split("=")[1]) for a in sys.argv[1:]
                   if a.startswith("--fvu=")), None)
    at_steps = next((a.split("=")[1] for a in sys.argv[1:]
                     if a.startswith("--steps=")), None)
    if "--per-layer" in sys.argv:
        # in a `resid` cascade layer 0 classifies the input and later layers
        # classify residuals, so the aggregate hides a depth trend
        print(f"{'arm':>22} {'select':>8} {'stat':>5}  " +
              "  ".join(f"{'L'+str(i):>6}" for i in range(5)) +
              f"  {'mean':>6}")
        print("-" * 82)
        for arm in args:
            Ws, spec, step = load_dicts(arm)
            g = np.array([geometry(W, spec.get("signed", False))
                          for W in Ws])
            short = arm.rstrip("/").split("/")[-1]
            for name, col, fmt in (("c", 0, "+6.3f"), ("eff", 1, "6.2f")):
                vals = "  ".join(f"{v:{fmt}}" for v in g[:, col])
                print(f"{short:>22} {spec['select']:>8} {name:>5}  {vals}"
                      f"  {g[:, col].mean():{fmt}}")
        return
    if at_steps:
        # trace: the same arm at several steps, to separate "how long it
        # trained" from "which selection rule it used"
        print(f"{'arm':>20} {'select':>8} {'step':>8} {'c':>8} {'eff':>7} "
              f"{'FVU_w':>8}")
        print("-" * 64)
        for arm in args:
            avail = steps(arm)
            for s in [int(x) for x in at_steps.split(",")]:
                s = min(avail, key=lambda a: abs(a - s))
                Ws, spec, s = load_dicts(arm, s)
                g = np.array([geometry(W, spec.get("signed", False))
                              for W in Ws]).mean(0)
                print(f"{arm:>20} {spec['select']:>8} {s:>8} {g[0]:>+8.4f} "
                      f"{g[1]:>7.2f} {fvu(arm, s):>8.4f}")
        return
    if target is not None:
        print(f"matched at FVU_w <= {target}, not at matched step\n")
    print(f"{'arm':>20} {'select':>8} {'k':>4} {'c':>8} {'eff':>7} "
          f"{'rank99':>7} {'step':>8} {'FVU_w':>8}")
    print("-" * 78)
    for arm in args:
        at = None
        if target is not None:
            at = step_at_fvu(arm, target)
            if at is None:
                print(f"{arm:>20} never reaches FVU_w {target} "
                      f"(best {fvu(arm):.4f})")
                continue
        try:
            Ws, spec, step = load_dicts(arm, at)
        except Exception as e:  # noqa: BLE001
            print(f"{arm:>20} FAILED {type(e).__name__}: {e}"[:110])
            continue
        g = np.array([geometry(W, spec.get("signed", False)) for W in Ws])
        c, eff, r99 = g.mean(0)
        print(f"{arm:>20} {spec['select']:>8} {spec['k']:>4} {c:>+8.4f} "
              f"{eff:>7.2f} {r99:>7.1f} {step:>8} {fvu(arm, step):>8.4f}")


if __name__ == "__main__":
    main()

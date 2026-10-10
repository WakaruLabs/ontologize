"""Layer-0 classifier logit scale against the temperature and the training
noise.

The straight-through argmax is scale-invariant, so the forward does not set
the classifier's logit scale. Its size against the surrogate temperature t
sets how informative the surrogate gradient is, and against the training
logit noise how often the noise rather than the classifier picks a head's
entry. Layer 0 reads the (encoded) cache row, so its logits need no
upstream layer. Per run:

  closed form     the logit SD at initialization. With w, v drawn
                  lecun_normal (variance 1/fan_in), a bilinear logit
                  (w.u)(v.u) has SD |u|^2/fan_in, the product of two
                  independent zero-mean factors of variance |u|^2/fan_in.
                  On the unit shape plus its constant coordinate
                  (|u|^2 = 2, fan_in = d + 1) that is 2/(d+1); on RMSNorm's
                  sphere of radius sqrt(d) it would be 1. A sigmoid gate on
                  the first factor scales the rest by the gate's RMS, about
                  1/2 on the unit shape, so SD ~ sqrt(|u|^2/fan_in)/2 at
                  order 2; order n multiplies n factors. Mean over rows.
  init            the within-head SD of the run's own classifier freshly
                  initialized (`--init-seed`), on the unit shape and with
                  the shape scaled to radius sqrt(d).
  per checkpoint  the within-head SD (the SD over a head's k logits, mean
                  over rows and heads) and the pooled SD; SD/t and the mean
                  top-1 margin/t at the run's temperature; the mean norm of
                  a classifier row; and `flips`, the share of (row, head)
                  argmaxes changed by the run's own training logit noise,
                  drawn `--draws` times through the classifier's noise
                  function at the logged sd_K, beside the median ratio of
                  the noise's RMS over a head's entries to its top-1
                  margin. Winner dropout is not included.

t, sd_K and their schedules are read from the run's log.jsonl
(`env_config`, the last one starting at or before the checkpoint) and
evaluated at each checkpoint's step by `ontostate.schedules`; the noise
kind is the one the restored classifier applies.

  uv run python experiments/ste-arm/logitscale.py \\
      --model data/out/sonar/multilingual/ste_h76_init01
  uv run python experiments/ste-arm/logitscale.py \\
      --cache data/activations/gpt2_l8.npy --out data/out/gpt2_l8/logitscale \\
      --model data/out/gpt2_l8/ste_h76 data/out/gpt2_l8/ste_h20_cat128

Writes <out>/init.csv (one row per run) and <out>/steps.csv (one row per
checkpoint).
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from pareto import load_onto
from ontologize.ontologizer import Ontologizer
from ontologize.training.ontostate import schedules


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", required=True,
                   help="one or more Ontologizer run dirs")
    p.add_argument("--steps", type=int, nargs="*", default=None,
                   help="checkpoints to score, of those each run has on "
                        "disk (default: all of them)")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--rows", type=int, default=2048,
                   help="cache tail rows the logits are measured on")
    p.add_argument("--draws", type=int, default=4,
                   help="noise draws per checkpoint")
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="noise draws")
    p.add_argument("--out", default="data/out/sonar/logitscale")
    return p.parse_args()


# ---------- statistics ----------

def within_head_sd(K: Float[np.ndarray, "b h k"]) -> float:
    """SD over a head's k logits, mean over rows and heads."""
    return float(K.std(-1).mean())


def top1_margin(K: Float[np.ndarray, "b h k"]) -> Float[np.ndarray, "b h"]:
    """Largest minus second-largest logit of each head."""
    top2 = np.sort(K, -1)[..., -2:]
    return top2[..., 1] - top2[..., 0]


def closed_form_sd(u_sq: Float[np.ndarray, "b"], fan_in: int, n: int,
                   gate: str, activation: str) -> float:
    """Mean over rows of the logit SD at a lecun_normal initialization, for
    classifier inputs of squared norm `u_sq`: `s^n` for an order-n product
    of independent zero-mean factors of SD `s = sqrt(u_sq / fan_in)`, and
    `s^(n-1) sqrt(E[sigmoid(s z)^2])`, z standard normal, when a sigmoid
    gates the first factor (Gauss-Hermite; about `s^(n-1) / 2` for small
    `s`, where the gate sits near 1/2). NaN for an output activation,
    another gate, or a gated order below 2, which have no such form."""
    s = np.sqrt(u_sq / fan_in)
    if activation != "none":
        return float("nan")
    if gate == "none":
        return float(np.mean(s ** n))
    if gate == "sigmoid" and n >= 2:
        z, w = np.polynomial.hermite_e.hermegauss(64)
        g2 = (w / (1 + np.exp(-np.outer(s, z))) ** 2).sum(-1) / w.sum()
        return float(np.mean(s ** (n - 1) * np.sqrt(g2)))
    return float("nan")


def hyper_at(entries: Sequence[Dict], step: int) -> Dict:
    """The `hyper` block of the last `env_config` entry of a run's log
    whose training started at or before `step` (a resumed run logs one per
    start, `meta.resume_from` naming the step it resumed from)."""
    best, start_best = None, -1
    for e in entries:
        if e.get("type") != "env_config":
            continue
        start = e.get("meta", {}).get("resume_from") or 0
        start = start if isinstance(start, int) else 0
        if start_best <= start <= step:
            best, start_best = e["hyper"], start
    if best is None:
        raise ValueError(f"no env_config at or before step {step}")
    return best


def temperature_noise(hyper: Dict, step: int) -> Tuple[float, float]:
    """`(t, sd_K)` in force at `step`, through the training schedules."""
    T, _, sd_K, _ = schedules(
        step, hyper.get("anneal_steps", 0), hyper["temperature"],
        hyper.get("temperature_end"), hyper.get("p_drop", 0.0),
        hyper.get("p_drop_start"), hyper.get("sd_K", 0.0),
        hyper.get("sd_K_end"))
    return float(T), float(sd_K)


def read_log(run: str) -> List[Dict]:
    with open(Path(run) / "log.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_steps(run: str) -> List[int]:
    """Checkpoint steps on disk, ascending."""
    return sorted(int(p.name) for p in Path(run).iterdir()
                  if p.name.isdigit() and (p / "state").exists())


# ---------- model ----------

def layer0(module: Ontologizer, X: Float[Array, "b d_in"], scale: float
           ) -> Tuple[Float[Array, "b g"], Float[Array, "b h k"]]:
    """Layer 0's classifier input and logits, with the shape part of the
    input (all but the trailing constant coordinates) scaled by `scale`;
    1 is the model's own input. Without gain-shape the input is the
    encoded row itself and `scale` must be 1."""
    de = module.dictencs[0]
    E, _ = module.encode(X, 0.0, None)
    U, _ = de.gainshape_in(module.constinput(E))
    d = U.shape[-1] - de.n_const
    U = jnp.concatenate([U[..., :d] * scale, U[..., d:]], -1)
    return U, de.classifier(U)


def layer0_noise(module: Ontologizer, K: Float[Array, "b h k"], sd: float,
                 key: jax.Array) -> Float[Array, "b h k"]:
    """`K` with the classifier's training logit noise at scale `sd`."""
    return module.dictencs[0].classifier.addnoise(K, sd, key)[0]


def logits(model: Ontologizer, params: Dict, X: Float[np.ndarray, "b d_in"],
           scale: float = 1.0
           ) -> Tuple[Float[np.ndarray, "b g"], Float[np.ndarray, "b h k"]]:
    U, K = jax.jit(lambda p, x: model.apply(p, x, scale, method=layer0))(
        {"params": params}, jnp.asarray(X))
    return np.asarray(U, np.float64), np.asarray(K, np.float64)


def noise_flips(model: Ontologizer, params: Dict,
                K: Float[np.ndarray, "b h k"], sd: float, draws: int,
                key: jax.Array) -> Tuple[float, float]:
    """Share of (row, head) argmaxes the noise changes, mean over draws,
    and the median ratio of the noise's RMS over a head's entries to the
    head's top-1 margin."""
    f = jax.jit(lambda p, k_, s, r: model.apply(p, k_, s, r,
                                                method=layer0_noise))
    Kj = jnp.asarray(K, jnp.float32)
    clean = K.argmax(-1)
    flips, sq = [], np.zeros(K.shape[:-1])
    for i in range(draws):
        Kn = np.asarray(f({"params": params}, Kj, sd,
                          jax.random.fold_in(key, i)), np.float64)
        flips.append(float((Kn.argmax(-1) != clean).mean()))
        sq += ((Kn - K) ** 2).mean(-1)
    rms = np.sqrt(sq / draws)
    margin = top1_margin(K)
    ratio = np.where(margin > 0, rms / np.maximum(margin, 1e-30), np.inf)
    return float(np.mean(flips)), float(np.median(ratio))


def main() -> None:
    cfg = parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float32)
    d = X.shape[1]
    key = jax.random.PRNGKey(cfg.seed)
    init_rows, step_rows = [], []

    for run in cfg.model:
        name = Path(run).name
        entries = read_log(run)
        on_disk = run_steps(run)
        steps = [s for s in cfg.steps if s in on_disk] if cfg.steps \
            else on_disk
        if not steps:
            raise ValueError(f"{run}: none of --steps is on disk")
        model, _, _ = load_onto(run, steps[-1])
        params0 = model.init(jax.random.PRNGKey(cfg.init_seed),
                             jnp.zeros((1, model.d_in), jnp.float32))["params"]
        U0, K0 = logits(model, params0, X)
        u_sq = (U0 ** 2).sum(-1)
        fan_in = U0.shape[-1]
        closed = closed_form_sd(u_sq, fan_in, model.n, model.gate,
                                model.activation_cl)
        gain = model.resid_gain
        if gain:
            Ur, K0r = logits(model, params0, X, float(np.sqrt(d)))
            closed_r = closed_form_sd((Ur ** 2).sum(-1), fan_in, model.n,
                                      model.gate, model.activation_cl)
            init_r = within_head_sd(K0r)
        else:
            closed_r = init_r = float("nan")
        t0, _ = temperature_noise(hyper_at(entries, 0), 0)
        noise_log = hyper_at(entries, steps[-1]).get("noise_K")
        if noise_log is not None and noise_log != model.noise_K:
            print(f"  WARNING: log noise_K {noise_log} != model noise_K "
                  f"{model.noise_K}; using the model's")
        init_rows.append({
            "run": name, "d": d, "fan_in": fan_in,
            "u_sq": round(float(u_sq.mean()), 6), "n": model.n,
            "gate": model.gate, "closed_sd": closed,
            "init_within_sd": within_head_sd(K0),
            "init_pooled_sd": float(K0.std()),
            "closed_sd_sqrtd": closed_r, "init_within_sd_sqrtd": init_r,
            "t0": t0, "closed_over_t": closed / t0})
        print(f"=== {name}  (d={d}, fan_in={fan_in}, |u|^2={u_sq.mean():.4f},"
              f" gate={model.gate}, noise {model.noise_K})")
        print(f"  init: closed-form SD {closed:.6f} (/t {closed / t0:.3f}), "
              f"within-head {within_head_sd(K0):.6f}, pooled {K0.std():.6f}"
              + (f"; sqrt(d) sphere: closed-form {closed_r:.4f}, within-head"
                 f" {init_r:.4f}" if gain else ""))
        print(f"  {'step':>7} {'t':>8} {'sd_K':>6} {'within SD':>10} "
              f"{'/closed':>8} {'SD/t':>8} {'margin/t':>9} {'|row|':>7} "
              f"{'flips':>6} {'noise/margin':>12}")
        for step in steps:
            model, params, step = load_onto(run, step)
            _, K = logits(model, params, X)
            T, sd_K = temperature_noise(hyper_at(entries, step), step)
            sd = within_head_sd(K)
            margin = float(top1_margin(K).mean())
            W = np.asarray(params["dictencs_0"]["classifier"]["weight"],
                           np.float64)
            row = float(np.linalg.norm(W, axis=-1).mean())
            flips, ratio = noise_flips(model, params, K, sd_K, cfg.draws,
                                       jax.random.fold_in(key, step))
            step_rows.append({
                "run": name, "step": int(step), "t": T,
                "noise": model.noise_K, "sd_K": sd_K, "within_sd": sd,
                "pooled_sd": float(K.std()), "sd_over_closed": sd / closed,
                "sd_over_t": sd / T, "margin_over_t": margin / T,
                "row_norm": row, "flips": flips,
                "noise_over_margin": ratio})
            print(f"  {step:>7} {T:8.2g} {sd_K:6.3g} {sd:10.6f} "
                  f"{sd / closed:8.3f} {sd / T:8.3f} {margin / T:9.3f} "
                  f"{row:7.4f} {flips:6.3f} {ratio:12.2f}")
        print()

    for fname, rows in (("init.csv", init_rows), ("steps.csv", step_rows)):
        with open(out / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

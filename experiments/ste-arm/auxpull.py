"""Which auxiliary losses are actually doing anything, and at what weight?

A penalty's share of the loss VALUE says nothing about whether it moves
the model; the optimizer sees gradients. Two live terms in this repo
read as reasonable by value and sit one to two orders of magnitude below
the weight at which they apply any force, and one arm was trained at a
weight chosen from the value share before this was noticed.

For each statistic this reports, per parameter group:

  ratio          |grad stat| / |grad MSE|, the force the statistic
                 carries per unit weight
  equal-pull w   the weight at which the penalty matches the objective's
                 own gradient, i.e. 1 / ratio
  applied        the configured weight times the ratio -- what the term
                 is currently worth. Below ~0.01 it is decoration.

Every gate is forced on for the measurement (a statistic is
stop-gradiented unless its weight is nonzero, so a term configured at 0
would otherwise report a gradient of exactly 0 and tell you nothing).
The configured weights are read from the checkpoint's own spec and the
`Hyperparams` passed, so "applied" describes the run as configured.

`--cos` adds the pairwise gradient cosines between the terms, which is
the question of whether two penalties are pulling the same direction.
Near-zero means they are independent levers, not substitutes.

The head-independence penalty is included, under whichever estimator
`--hsic-estimator` names. The two are not interchangeable here: the
biased estimator's value is dominated by a floor that independent heads
alone produce at a training batch, so a weight calibrated against it is
calibrated against the floor's gradient rather than the dependence's.

MSE's gradient shrinks over training while most penalties' do not, so
the answer is not constant: measure at the checkpoint you care about.
`--step 0` is available where an early checkpoint was kept.

  uv run python experiments/ste-arm/auxpull.py \\
      --model data/out/sonar/multilingual/ste_h76 --temperature 0.00015
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

import ontologize.fns.hsic as hsic

from pareto import load_onto

# statistic -> the Hyperparams weight that scales it. The gate names
# differ from the weight names in two places, so they are spelled out:
# cossim_b is gated by `bcossim_loss` and support by `support_loss`.
TERMS = [("L1_K", "s_L1K"), ("L1_F", "s_L1F"), ("entropy", "s_H"),
         ("cossim_b", "s_bcossim"), ("support", "s_support"),
         ("cossim_k", "s_kcossim"),
         ("KL_m", "s_Hm"), ("hsic_heads", "s_hsic_heads")]
GATES = dict(sparse_K=True, sparse_F=True, entropy_loss=True,
             bcossim_loss=True, support_loss=True, kcossim_loss=True,
             hmean_loss=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint dir")
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy",
                   help="'' scores unweighted MSE")
    p.add_argument("--b", type=int, default=256,
                   help="batch; the batch statistics are batch-size "
                        "sensitive, so use the training batch")
    p.add_argument("--weights", default=None,
                   help="JSON of configured weights, e.g. "
                        "'{\"s_bcossim\": 2e-4}'. Default: sonar.py's")
    p.add_argument("--groups", nargs="+",
                   default=["classifier", "dict", "decoder"])
    p.add_argument("--hsic-sigma2", type=float, default=1.0,
                   help="RBF bandwidth^2 for the head grams; 0 = median")
    p.add_argument("--hsic-estimator", default="unbiased",
                   choices=["biased", "unbiased"])
    p.add_argument("--cos", action="store_true",
                   help="also print pairwise gradient cosines")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    model, raw, step = load_onto(cfg.model, cfg.step)
    # a stat is stop-gradiented unless its loss flag is set, so every
    # gate goes on or the measurement reads its own configuration back
    probe_model = dataclasses.replace(model, **GATES)
    params = {"params": raw}

    mm = np.load(cfg.cache, mmap_mode="r")
    X = jnp.asarray(np.asarray(mm[-cfg.b:], np.float32))
    rw = (jnp.sqrt(jnp.asarray(np.load(cfg.mse_weights)))
          if cfg.mse_weights else 1.0)

    import sonar as base
    conf = {name: getattr(base, name, 0.0) for _, name in TERMS}
    if cfg.weights:
        conf.update(json.loads(cfg.weights))

    def forward(p: PyTree, X: Float[Array, "b d_in"]
                ) -> Tuple[Dict[str, Float[Array, ""]],
                           Float[Array, "b d_out"]]:
        """Every logged statistic, summed over layers, plus the decode."""
        def probe(module, X: Float[Array, "b d_in"]
                  ) -> Tuple[Dict[str, Float[Array, ""]],
                             Float[Array, "b d_out"]]:
            E, _ = module.encode(X, 0.0, None)
            R = module.resid(E)
            Ein = module.constinput(E)
            acc = {k: 0.0 for k, _ in TERMS}
            for i, de in enumerate(module.dictencs):
                U, G = de.gainshape_in(Ein)
                K = de.classifier(U)
                P = de.dict.cluster(K, cfg.temperature)
                # the router gain enters the logged stats as in
                # DictBlock.withStats: base output and S, no fiber
                S = de.scale(U) if de.scaled else None
                Fs = de.dict.hfwd(P, S)
                acc["L1_K"] += de.classifier.l1(K)
                acc["L1_F"] += de.dict.l1(Fs)
                acc["entropy"] += de.dict.entropy(P, S)
                acc["cossim_b"] += de.dict.bcossim_tags(P, S)
                acc["support"] += de.dict.support_overlap(P)
                acc["cossim_k"] += de.dict.rowcos().mean()
                acc["KL_m"] += de.dict.hmean_kl(P)
                # the head-independence penalty is computed in the loss
                # rather than the layer, so it has no gate to force on
                acc["hsic_heads"] += hsic.pairwise_head_cka(
                    P, cfg.hsic_sigma2, cfg.hsic_estimator)
                R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
                if i < module.l - 1:
                    Ein = module.nextinput(X, R, P.reshape(X.shape[0], -1))
            return acc, module.decode(R)
        return probe_model.apply(p, X, method=probe)

    def gnorm(tree: PyTree, group: str) -> float:
        """L2 norm of one parameter group's gradient."""
        leaves = [v for path, v in jax.tree_util.tree_leaves_with_path(tree)
                  if group in jax.tree_util.keystr(path)]
        return float(jnp.sqrt(sum(jnp.sum(v * v) for v in leaves))) if leaves else 0.0

    def gvec(tree: PyTree, group: str) -> Float[Array, "p"]:
        """One parameter group's gradient, flattened in a stable order."""
        leaves = [v.ravel() for path, v in
                  sorted(jax.tree_util.tree_leaves_with_path(tree),
                         key=lambda kv: str(kv[0]))
                  if group in jax.tree_util.keystr(path)]
        return jnp.concatenate(leaves) if leaves else jnp.zeros(1)

    mse = lambda p: (((forward(p, X)[1] - X) * rw) ** 2).mean()
    g_mse = jax.grad(mse)(params)
    vals, _ = forward(params, X)
    print(f"{Path(cfg.model).name} step {step}, batch {cfg.b}, "
          f"T={cfg.temperature:g}, whitened MSE "
          f"{float(mse(params)):.3e}\n")

    grads = {name: jax.grad(lambda p, n=name: forward(p, X)[0][n])(params)
             for name, _ in TERMS}
    rows = {}
    for group in cfg.groups:
        m = gnorm(g_mse, group)
        if m == 0.0:
            continue
        print(f"--- {group} (|grad MSE| {m:.3e}) ---")
        print(f"{'statistic':10s} {'value':>9} {'|grad|':>10} {'ratio':>9} "
              f"{'equal-pull w':>13} {'configured':>11} {'applied':>9}")
        for name, wname in TERMS:
            b = gnorm(grads[name], group)
            w = float(conf.get(wname, 0.0) or 0.0)
            if b == 0.0:
                print(f"{name:10s} {float(vals[name]):9.3f} {b:10.3e}"
                      f"{'  (no gradient to this group)':>44}")
                continue
            applied = w * b / m
            flag = "  inert" if applied < 0.01 else ""
            print(f"{name:10s} {float(vals[name]):9.3f} {b:10.3e} "
                  f"{b/m:9.1f} {m/b:13.2e} {w:11.1e} {applied:9.3f}{flag}")
            rows[(group, name)] = dict(value=float(vals[name]), gnorm=b,
                                       ratio=b/m, equal_pull=m/b,
                                       configured=w, applied=applied)
        print()

    if cfg.cos:
        for group in cfg.groups:
            V = {n: gvec(grads[n], group) for n, _ in TERMS}
            live = [n for n, _ in TERMS if float(jnp.linalg.norm(V[n])) > 0]
            if len(live) < 2:
                continue
            print(f"pairwise gradient cosine on {group} "
                  f"(0 = independent levers, 1 = the same lever):")
            print("           " + " ".join(f"{n[:8]:>8}" for n in live))
            for a in live:
                cells = [f"{float(V[a] @ V[b] / (jnp.linalg.norm(V[a]) * jnp.linalg.norm(V[b]))):8.2f}"
                         for b in live]
                print(f"{a:10s} " + " ".join(cells))
            print()

    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out).write_text(json.dumps(
            {"model": str(cfg.model), "step": int(step), "b": cfg.b,
             "temperature": cfg.temperature,
             "rows": {f"{g}/{n}": v for (g, n), v in rows.items()}}, indent=2))
        print(f"\n-> {cfg.out}")


if __name__ == "__main__":
    main()

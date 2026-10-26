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
The configured weights are the run's own, from the last `hyper` record in
its `log.jsonl` (sonar.py's for a run without one), and `--weights`
overrides them, so "applied" describes the run as configured.

`--coord C` restricts every gradient to the parameters that write output
coordinate C (a decoder-free model's atoms, else the decoder kernel's row
for C; coordfvu.py's `output_params`). It is for a coordinate the MSE
weights do not reach, where the statistics' gradients are all there is.
Per statistic it reports:

  |grad| on C    the norm of the gradient on C's parameters
  / median       that over the same statistic's on the median coordinate's
  pull vs MSE    the configured weight times |grad| on C, over the MSE's
                 |grad| on the median coordinate's parameters
  net share      its share of the net weighted pull on C, the projection
                 of its weighted gradient on their sum (the shares sum to 1)

and the net pull's root-mean-square per parameter, to set beside Adam's
eps. `--batches N` averages every gradient over N consecutive batches of
the tail, as the optimizer's first moment does over steps.

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
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

import ontologize.fns.hsic as hsic

from coordfvu import output_params
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
    p.add_argument("--batches", type=int, default=1,
                   help="consecutive tail batches to average each gradient over")
    p.add_argument("--coord", type=int, default=None,
                   help="also report each statistic's pull on the parameters "
                        "that write this output coordinate")
    p.add_argument("--weights", default=None,
                   help="JSON of configured weights, e.g. "
                        "'{\"s_bcossim\": 2e-4}'. Default: the run's own "
                        "(log.jsonl), else sonar.py's")
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


def in_group(path: Tuple, group: str) -> bool:
    """Whether a parameter's tree path has `group` as one of its keys: the
    DictBlock's `dict` is a key under each `dictencs_<i>`, so matching the
    path's text instead would put a layer's classifier in it too."""
    return any(getattr(k, "key", None) == group for k in path)


def run_weights(model: str) -> Optional[Dict[str, float]]:
    """The statistics' weights the run trained with, from the last `hyper`
    record in its log.jsonl; None for a run without one."""
    log = Path(model) / "log.jsonl"
    if not log.exists():
        return None
    hypers = [r["hyper"] for r in map(json.loads, log.read_text().splitlines())
              if r.get("hyper")]
    if not hypers:
        return None
    return {name: float(hypers[-1].get(name, 0.0) or 0.0) for _, name in TERMS}


def coord_pull(grads: Dict[str, Float[np.ndarray, "r d"]], g_mse: Float[np.ndarray, "r d"],
               w: Dict[str, float], c: int) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Per statistic, its gradient on coordinate c's parameters (column c
    of the (parameters, coordinates) matrices `grads`): the norm, that over
    the median coordinate's, the weighted norm over the MSE's on the median
    coordinate, and its share of the net weighted gradient on c; and the
    net gradient's norm and root-mean-square per parameter."""
    ref = float(np.median(np.linalg.norm(g_mse, axis=0)))
    net = sum(w[n] * G[:, c] for n, G in grads.items())
    nn2 = float(net @ net)
    rows = {}
    for n, G in grads.items():
        col = np.linalg.norm(G, axis=0)
        med = float(np.median(col))
        rows[n] = {"gnorm": float(col[c]),
                   "over_median": float(col[c] / med) if med > 0 else float("nan"),
                   "pull_vs_mse": w[n] * float(col[c]) / ref if ref > 0 else float("nan"),
                   "net_share": w[n] * float(G[:, c] @ net) / nn2 if nn2 > 0 else float("nan")}
    return rows, {"net_gnorm": float(np.sqrt(nn2)), "net_rms": float(np.sqrt(nn2 / len(net))),
                  "mse_gnorm": float(np.linalg.norm(g_mse[:, c])), "mse_median_gnorm": ref}


def main() -> None:
    cfg = parse_args()
    model, raw, step = load_onto(cfg.model, cfg.step)
    # a stat is stop-gradiented unless its loss flag is set, so every
    # gate goes on or the measurement reads its own configuration back
    probe_model = dataclasses.replace(model, **GATES)
    params = {"params": raw}

    mm = np.load(cfg.cache, mmap_mode="r")
    tail = np.asarray(mm[-cfg.b * cfg.batches:], np.float32)
    batches = [jnp.asarray(tail[i * cfg.b:(i + 1) * cfg.b]) for i in range(cfg.batches)]
    rw = (jnp.sqrt(jnp.asarray(np.load(cfg.mse_weights)))
          if cfg.mse_weights else 1.0)

    conf = run_weights(cfg.model)
    if conf is None:
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
                  if in_group(path, group)]
        return float(jnp.sqrt(sum(jnp.sum(v * v) for v in leaves))) if leaves else 0.0

    def gvec(tree: PyTree, group: str) -> Float[Array, "p"]:
        """One parameter group's gradient, flattened in a stable order."""
        leaves = [v.ravel() for path, v in
                  sorted(jax.tree_util.tree_leaves_with_path(tree),
                         key=lambda kv: str(kv[0]))
                  if in_group(path, group)]
        return jnp.concatenate(leaves) if leaves else jnp.zeros(1)

    def mean(f):
        """f(X) averaged over the batches, summed as it goes so that one
        gradient tree per batch is never held at once."""
        acc = None
        for X in batches:
            v = f(X)
            acc = v if acc is None else jax.tree_util.tree_map(jnp.add, acc, v)
        return jax.tree_util.tree_map(lambda v: v / len(batches), acc)

    mse = lambda p, X: (((forward(p, X)[1] - X) * rw) ** 2).mean()
    g_mse = mean(lambda X: jax.grad(mse)(params, X))
    vals = mean(lambda X: forward(params, X)[0])
    print(f"{Path(cfg.model).name} step {step}, {cfg.batches} batch(es) of {cfg.b}, "
          f"T={cfg.temperature:g}, whitened MSE "
          f"{float(mean(lambda X: mse(params, X))):.3e}\n")

    grads = {name: mean(lambda X, n=name: jax.grad(lambda p: forward(p, X)[0][n])(params))
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

    coord = None
    if cfg.coord is not None:
        d = batches[0].shape[1]
        w = {n: float(conf.get(wn, 0.0) or 0.0) for n, wn in TERMS}
        crow, net = coord_pull({n: output_params(grads[n]["params"], d) for n, _ in TERMS},
                               output_params(g_mse["params"], d), w, cfg.coord)
        c = cfg.coord
        print(f"--- the parameters that write coordinate {c} ---")
        print(f"MSE |grad| on {c}: {net['mse_gnorm']:.3e}, on the median coordinate: "
              f"{net['mse_median_gnorm']:.3e}")
        print(f"{'statistic':10s} {'configured':>11} {'|grad| on c':>12} {'/ median':>9} "
              f"{'pull vs MSE':>12} {'net share':>10}")
        for n, _ in TERMS:
            r = crow[n]
            print(f"{n:10s} {w[n]:11.1e} {r['gnorm']:12.3e} {r['over_median']:9.2f} "
                  f"{r['pull_vs_mse']:12.3e} {r['net_share']:10.3f}")
        print(f"net weighted pull on {c}: |grad| {net['net_gnorm']:.3e}, "
              f"root-mean-square per parameter {net['net_rms']:.3e}\n")
        coord = {"coord": c, "rows": crow, **net}

    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out).write_text(json.dumps(
            {"model": str(cfg.model), "step": int(step), "b": cfg.b,
             "batches": cfg.batches, "temperature": cfg.temperature,
             "weights": conf, "coord": coord,
             "rows": {f"{g}/{n}": v for (g, n), v in rows.items()}}, indent=2))
        print(f"\n-> {cfg.out}")


if __name__ == "__main__":
    main()

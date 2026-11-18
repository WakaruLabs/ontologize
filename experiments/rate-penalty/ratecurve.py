"""A hard code's rate--distortion curve over its layer prefixes, with the
rate read as the entries' realized entropy rather than the nominal
h * log2(k) bits per layer.

On the cache tail, for each run and each prefix of its layers:

  - FVU_w, the whitened weighted error over the tail's variance, as
    coordfvu.py computes it (a weight of 0 drops a coordinate);
  - realized bits: the sum over the prefix's heads of the plug-in entropy
    of each head's argmax entry over the tail rows, Miller-Madow corrected
    (quantrate.plugin_bits), which is what an entropy coder that codes
    each head's entry by its own usage would spend per row;
  - nominal bits, and the entries no tail row selects.

Two codes per run. "pinned": every layer's gain is fixed to its mean over
the tail rows, in the forward itself, so later layers see the residual the
pinned code leaves and the code is the entries alone (errspectrum.py's
convention). "float": the gains as the model computes them, sent at no
cost, a lower bound on what sending them would cost; quantrate.py prices
them.

A run trained with a rate penalty on its batch-mean usage entropy
(README.md) and one trained without should be compared on these curves:
the penalty can only help if its points sit below the curve the unpenalized
run's prefixes trace at the same realized bits.

  JAX_PLATFORMS=cpu uv run python experiments/rate-penalty/ratecurve.py \\
      --ckpt <run dir> ... --out data/out/sonar/rate_penalty/ratecurve
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from pareto import load_onto
from quantrate import plugin_bits


def usage_counts(E: Int[np.ndarray, "n h"], k: int) -> Int[np.ndarray, "h k"]:
    """Per head, how many rows select each entry."""
    h = E.shape[1]
    C = np.zeros((h, k), np.int64)
    for j in range(h):
        C[j] = np.bincount(E[:, j], minlength=k)
    return C


def head_bits(C: Int[np.ndarray, "h k"]) -> Float[np.ndarray, "h"]:
    """Each head's realized entropy in bits, from its usage counts."""
    n = int(C[0].sum())
    return np.array([plugin_bits(c, n) for c in C])


def prefix_curve(err: Float[np.ndarray, "l d"], var: Float[np.ndarray, "d"],
                 w: Float[np.ndarray, "d"], bits: Float[np.ndarray, "l"]
                 ) -> Tuple[Float[np.ndarray, "l"], Float[np.ndarray, "l"]]:
    """Per prefix, FVU_w from its per-coordinate mean squared error, and
    the cumulative bits of the layers it holds."""
    fvu = (err * w).mean(1) / (var * w).mean()
    return fvu, np.cumsum(bits)


def forward(ckpt: str, step: int, T: float, X: Float[np.ndarray, "n d"],
            batch: int) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict]:
    """For the float and the pinned code: each prefix's per-coordinate
    mean squared error over X, and every layer's argmax entries."""
    model, raw, step = load_onto(ckpt, step)
    params = {"params": raw}

    def fwd(module, Xb: Float[Array, "b d"], const: Float[Array, "l"], pin: bool
            ) -> Tuple[Float[Array, "l b d"], Int[Array, "l b h"], Float[Array, "l b"]]:
        E, _ = module.encode(Xb, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        outs, idx, gains = [], [], []
        for i, de in enumerate(module.dictencs):
            U, G = de.gainshape_in(Ein)
            P = de.dict.cluster(de.classifier(U), T)
            if G is None:
                g, gains_i = None, jnp.ones(Xb.shape[:1])
            else:
                g = jnp.where(pin, jnp.full_like(G, const[i]), G)
                gains_i = G[..., 0]
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), g)
            outs.append(module.decode(R))
            idx.append(P.reshape(Xb.shape[0], module.h, module.k).argmax(-1))
            gains.append(gains_i)
            if i < module.l - 1:
                Ein = module.nextinput(Xb, R, P.reshape(Xb.shape[0], -1))
        return jnp.stack(outs), jnp.stack(idx), jnp.stack(gains)

    f = jax.jit(lambda p, x, c, pin: model.apply(p, x, c, pin, method=fwd),
                static_argnums=3)

    def run(const: Array, pin: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        err = np.zeros((model.l, X.shape[1]))
        idx, gains = [], []
        for i in range(0, len(X), batch):
            Xb = X[i:i + batch]
            Y, I, G = f(params, jnp.asarray(Xb, jnp.float32), const, pin)
            err += ((np.asarray(Y, np.float64) - Xb[None]) ** 2).sum(1)
            idx.append(np.asarray(I))
            gains.append(np.asarray(G))
        return err / len(X), np.concatenate(idx, 1), np.concatenate(gains, 1)

    err_f, idx_f, G = run(jnp.zeros(model.l), False)
    gmean = G.mean(1)
    err_p, idx_p, _ = run(jnp.asarray(gmean, jnp.float32), True)
    info = {"ckpt": ckpt, "step": step, "l": model.l, "h": model.h, "k": model.k,
            "gain_means": [float(g) for g in gmean]}
    return {"float": (err_f, idx_f), "pinned": (err_p, idx_p)}, info


def score(codes: Dict[str, Tuple[np.ndarray, np.ndarray]], info: Dict,
          var: Float[np.ndarray, "d"], w: Float[np.ndarray, "d"]) -> List[Dict]:
    """One row per code and prefix."""
    k, h = info["k"], info["h"]
    rows = []
    for code, (err, idx) in codes.items():
        counts = [usage_counts(idx[i], k) for i in range(info["l"])]
        bits = np.array([head_bits(C).sum() for C in counts])
        fvu, cum = prefix_curve(err, var, w, bits)
        for i in range(info["l"]):
            rows.append({"code": code, "prefix": i + 1, "fvu": float(fvu[i]),
                         "bits": float(cum[i]),
                         "nominal": (i + 1) * h * math.log2(k),
                         "layer_bits": float(bits[i]),
                         "dead": int((counts[i] == 0).sum())})
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True,
                   help="run directories, each at its latest step, or dir@step")
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_sent16.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/sentcache/mse_weights_no837.npy")
    p.add_argument("--rows", type=int, default=32765, help="cache tail rows")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--out", default="data/out/sonar/rate_penalty/ratecurve")
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float64)
    var = X.var(0)
    w = np.load(cfg.mse_weights).astype(np.float64)
    summary = {"cache": cfg.cache, "rows": cfg.rows, "mse_weights": cfg.mse_weights,
               "temperature": cfg.temperature, "runs": {}}
    table = []
    for spec in cfg.ckpt:
        path, _, s = spec.partition("@")
        codes, info = forward(path, int(s) if s else 0, cfg.temperature, X, cfg.batch)
        rows = score(codes, info, var, w)
        name = f"{Path(path).name}_{info['step']}"
        summary["runs"][name] = {**info, "prefixes": rows}
        for r in rows:
            table.append({"run": name, **r})
            print(name, r["code"], r["prefix"], f"bits {r['bits']:.1f}",
                  f"fvu {r['fvu']:.4f}", f"dead {r['dead']}", flush=True)
    with open(out / "curve.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(table[0]))
        wr.writeheader()
        wr.writerows(table)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {out / 'curve.csv'}, {out / 'summary.json'}")


if __name__ == "__main__":
    main()

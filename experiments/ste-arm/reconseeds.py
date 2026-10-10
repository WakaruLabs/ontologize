"""Do two runs compute the same function, or only reach the same error?

Two seeds can match each other's held-out error to four digits and still
reconstruct each row differently. This compares the reconstructions
themselves, on the cache tail in the objective's whitened frame:

  fvu_a, fvu_b  each run's whitened error, sum_w (x - xhat)^2 over the
                tail's whitened variance sum_w (x - mean)^2
  cross         sum_w (xhat_A - xhat_B)^2 on the same scale
  indep         fvu_a + fvu_b: what cross would be if the two residuals
                were uncorrelated
  resid_corr    whitened cosine between the residuals x - xhat_A and
                x - xhat_B, pooled over rows

Since xhat_A - xhat_B is the difference of the two residuals,
cross = indep - 2 resid_corr sqrt(fvu_a fvu_b). Two runs computing one
function have cross near 0 and resid_corr near 1; two that reach equal
error by different functions have cross near indep and resid_corr near 0.

An Ontologizer is scored per prefix (layers 0..i applied), from the
per-prefix decodes its own forward returns under deepsup
(`Ontologizer.withStats`), or on its final reconstruction without it. An
sae.py run (pass its params.npz) has one reconstruction, encoded by the
rule its meta.json records. Two models with the same number of prefixes
are compared prefix by prefix, any other two on their final
reconstructions. `--ctrl-step` adds the positive control for an
Ontologizer A: A against its own checkpoint at that step.

For two sae.py runs the supports are compared as well, over latents that
fire on the tail, with `--near` the decoder-direction cosines at which
two latents count as copies:
  best      per live latent of A, its best cosine to a live latent of B:
            median, mean, mean weighted by A's firing rate, and the share
            above each threshold
  assigned  the one-to-one linear assignment between the live latents:
            mean cosine and the share above each threshold
  rows      per row, the share of A's active latents with a copy among
            B's active latents on that row, averaged over rows

  uv run python experiments/ste-arm/reconseeds.py \\
      --a data/out/sonar/multilingual/ste_h76_init01 --step-a 369500 \\
      --b data/out/sonar/multilingual/ste_h76_i01_s43 --step-b 369500 \\
      --ctrl-step 350000

  uv run python experiments/ste-arm/reconseeds.py \\
      --a data/out/sonar/sae_conv/m5120_k32/params.npz \\
      --b data/out/sonar/sae_conv/m5120_k32_s43/params.npz

Writes <out>/recon.csv (pair, prefix, fvu_a, fvu_b, cross, indep,
resid_corr) and <out>/summary.json (inputs, and the support statistics
for two SAEs); default out directory
data/out/sonar/reconseeds/<a>_vs_<b>.
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float
from scipy.optimize import linear_sum_assignment

import sae
from ontologize.ontologizer import Ontologizer
from pareto import load_onto, parse_sae_name

# rows -> (per-prefix reconstructions, SAE activations or None)
Recon = Callable[[Float[np.ndarray, "b d"]],
                 Tuple[Float[np.ndarray, "p b d"],
                       Optional[Float[np.ndarray, "b m"]]]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True,
                   help="checkpoint dir, or an sae.py run's params.npz")
    p.add_argument("--b", required=True,
                   help="checkpoint dir, or an sae.py run's params.npz")
    p.add_argument("--step-a", type=int, default=0)
    p.add_argument("--step-b", type=int, default=0)
    p.add_argument("--ctrl-step", type=int, default=0,
                   help="score A against its own checkpoint at this step "
                        "(Ontologizer A; 0 = no control)")
    p.add_argument("--temperature", type=float, default=0.00015,
                   help="Ontologizer classification temperature; irrelevant "
                        "under select='ste'")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy",
                   help="per-dimension weights defining the whitened frame")
    p.add_argument("--rows", type=int, default=32768,
                   help="cache tail rows; the held-out split")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--near", type=float, nargs="+", default=[0.9, 0.7],
                   help="decoder cosines at which two SAE latents count as "
                        "copies")
    p.add_argument("--out", default=None)
    return p.parse_args()


def is_sae(path: str) -> bool:
    return path.endswith(".npz")


def onto_recon(ckpt: str, step: int, T: float) -> Tuple[Recon, int, str]:
    """An Ontologizer's per-prefix reconstructions, the step loaded and a
    name for it."""
    model, raw, used = load_onto(ckpt, step)
    params = {"params": raw}
    f = jax.jit(lambda p, x: model.apply(
        p, x, 0.0, None, temperature=T, method=Ontologizer.withStats)[0])

    def recon(X: Float[np.ndarray, "b d"]):
        Y = np.asarray(f(params, jnp.asarray(X)))
        return (Y[None] if Y.ndim == 2 else Y[:model.l]), None

    return recon, used, f"{Path(ckpt).name}_{used}"


def sae_recon(path: str) -> Tuple[Recon, Float[np.ndarray, "m d"], str]:
    """An sae.py run's reconstruction and activations, its decoder
    directions (rows of W_dec) and a name for it."""
    meta_path = Path(path).parent / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        topk, groups, group_fn = meta["topk"], meta["groups"], meta["group_fn"]
    else:  # pre-meta.json run: recover the encode rule from the name
        _, topk = parse_sae_name(path)
        groups, group_fn = 0, "top1"
    params = {k: jnp.asarray(v) for k, v in np.load(path).items()}

    @jax.jit
    def f(X):
        z = sae.encode(params, X, topk, groups, group_fn)
        return sae.decode(params, z), z

    def recon(X: Float[np.ndarray, "b d"]):
        Y, z = f(jnp.asarray(X))
        return np.asarray(Y)[None], np.asarray(z)

    return recon, np.asarray(params["W_dec"], np.float64), Path(path).parent.name


def batch_sums(X: Float[np.ndarray, "b d"], Y_a: Float[np.ndarray, "p b d"],
               Y_b: Float[np.ndarray, "p b d"], w: Float[np.ndarray, "d"]
               ) -> Float[np.ndarray, "p 4"]:
    """Per prefix, this batch's whitened sums: |e_a|^2, |e_b|^2,
    |xhat_a - xhat_b|^2 and <e_a, e_b>."""
    X = X.astype(np.float64)
    e_a = X[None] - Y_a.astype(np.float64)
    e_b = X[None] - Y_b.astype(np.float64)
    return np.stack([(w * e_a ** 2).sum((1, 2)), (w * e_b ** 2).sum((1, 2)),
                     (w * (e_a - e_b) ** 2).sum((1, 2)),
                     (w * e_a * e_b).sum((1, 2))], -1)


def agreement(sums: Float[np.ndarray, "p 4"], var: float) -> List[dict]:
    """The per-prefix rows from summed `batch_sums`, against the whitened
    tail variance `var` = sum_w (x - mean)^2."""
    rows = []
    for i, (s_a, s_b, cross, dot) in enumerate(sums):
        rows.append({"prefix": i + 1, "fvu_a": s_a / var, "fvu_b": s_b / var,
                     "cross": cross / var, "indep": (s_a + s_b) / var,
                     "resid_corr": dot / np.sqrt(s_a * s_b)})
    return rows


def decoder_cos(D_a: Float[np.ndarray, "ma d"], D_b: Float[np.ndarray, "mb d"]
                ) -> Float[np.ndarray, "ma mb"]:
    """Cosine between every decoder direction of A and every one of B."""
    u = lambda D: D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True),
                                 1e-30)
    return u(D_a) @ u(D_b).T


def best_match(C: Float[np.ndarray, "ma mb"], rate: Float[np.ndarray, "ma"],
               near: List[float]) -> Dict[str, float]:
    """Each latent of A's best cosine to a latent of B, summarized."""
    best = C.max(1)
    out = {"median": float(np.median(best)), "mean": float(best.mean()),
           "rate_weighted_mean": float((best * rate).sum() / rate.sum())}
    out.update({f"share_gt_{t}": float((best > t).mean()) for t in near})
    return out


def assigned(C: Float[np.ndarray, "ma mb"], near: List[float]
             ) -> Dict[str, float]:
    """The one-to-one linear assignment maximizing total cosine."""
    r, c = linear_sum_assignment(-C)
    v = C[r, c]
    out = {"mean": float(v.mean())}
    out.update({f"share_gt_{t}": float((v > t).mean()) for t in near})
    return out


def row_copies(act_a: Bool[np.ndarray, "n ma"], act_b: Bool[np.ndarray, "n mb"],
               copy: Bool[np.ndarray, "ma mb"]) -> float:
    """Per row, the share of A's active latents that have a copy among B's
    active latents on the same row, averaged over rows where A is
    active."""
    shares = []
    for a, b in zip(act_a, act_b):
        ia, ib = np.flatnonzero(a), np.flatnonzero(b)
        if len(ia):
            shares.append(copy[np.ix_(ia, ib)].any(1).mean())
    return float(np.mean(shares))


def supports(Z_a: Float[np.ndarray, "n ma"], Z_b: Float[np.ndarray, "n mb"],
             D_a: Float[np.ndarray, "ma d"], D_b: Float[np.ndarray, "mb d"],
             near: List[float]) -> dict:
    """The support statistics for two SAEs, over latents firing on the
    rows."""
    act_a, act_b = Z_a > 0, Z_b > 0
    live_a, live_b = act_a.any(0), act_b.any(0)
    C = decoder_cos(D_a[live_a], D_b[live_b])
    rows = {}
    for t in near:
        copy = np.zeros((len(live_a), len(live_b)), dtype=bool)
        copy[np.ix_(live_a, live_b)] = C > t
        rows[f"share_gt_{t}"] = row_copies(act_a, act_b, copy)
    return {"live_a": int(live_a.sum()), "live_b": int(live_b.sum()),
            "l0_a": float(act_a.sum(1).mean()),
            "l0_b": float(act_b.sum(1).mean()),
            "best": best_match(C, act_a.mean(0)[live_a], near),
            "assigned": assigned(C, near), "rows": rows}


def load(path: str, step: int, T: float
         ) -> Tuple[Recon, Optional[Float[np.ndarray, "m d"]], str]:
    """(reconstruct, decoder directions for an SAE, name)."""
    if is_sae(path):
        return sae_recon(path)
    recon, _, name = onto_recon(path, step, T)
    return recon, None, name


def main() -> None:
    cfg = parse_args()
    if cfg.ctrl_step and is_sae(cfg.a):
        raise SystemExit("--ctrl-step needs an Ontologizer checkpoint as --a")
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float32)
    w = np.load(cfg.mse_weights).astype(np.float64)
    Xd = X.astype(np.float64)
    var = float((w * (Xd - Xd.mean(0)) ** 2).sum())

    rec_a, D_a, name_a = load(cfg.a, cfg.step_a, cfg.temperature)
    rec_b, D_b, name_b = load(cfg.b, cfg.step_b, cfg.temperature)
    models = {"a": rec_a, "b": rec_b}
    pairs = {"seeds": ("a", "b")}
    if cfg.ctrl_step:
        models["c"], _, name_c = onto_recon(cfg.a, cfg.ctrl_step,
                                            cfg.temperature)
        pairs["control"] = ("c", "a")
    print(f"A {name_a}, B {name_b}"
          + (f", control {name_c}" if cfg.ctrl_step else "")
          + f": {len(X)} rows")

    sums: Dict[str, np.ndarray] = {}
    Z_a, Z_b = [], []
    for i in range(0, len(X), cfg.batch):
        xb = X[i:i + cfg.batch]
        outs = {key: f(xb) for key, f in models.items()}
        for pair, (p, q) in pairs.items():
            Yp, Yq = outs[p][0], outs[q][0]
            if Yp.shape[0] != Yq.shape[0]:   # unequal depths: finals only
                Yp, Yq = Yp[-1:], Yq[-1:]
            s = batch_sums(xb, Yp, Yq, w)
            sums[pair] = s if pair not in sums else sums[pair] + s
        if outs["a"][1] is not None and outs["b"][1] is not None:
            Z_a.append(outs["a"][1])
            Z_b.append(outs["b"][1])

    rows = [{"pair": pair, **r} for pair, s in sums.items()
            for r in agreement(s, var)]
    print(f"\n  {'pair':<8} {'prefix':>6} {'fvu_a':>7} {'fvu_b':>7} "
          f"{'cross':>7} {'indep':>7} {'resid_corr':>10}")
    for r in rows:
        print(f"  {r['pair']:<8} {r['prefix']:6d} {r['fvu_a']:7.4f} "
              f"{r['fvu_b']:7.4f} {r['cross']:7.4f} {r['indep']:7.4f} "
              f"{r['resid_corr']:10.4f}")

    summary = {"a": cfg.a, "b": cfg.b, "name_a": name_a, "name_b": name_b,
               "ctrl_step": cfg.ctrl_step, "rows": len(X), "cache": cfg.cache,
               "mse_weights": cfg.mse_weights, "temperature": cfg.temperature}
    if Z_a:
        summary["supports"] = supports(np.concatenate(Z_a),
                                       np.concatenate(Z_b), D_a, D_b,
                                       cfg.near)
        print("\nsupports:\n" + json.dumps(summary["supports"], indent=2))

    out = Path(cfg.out or f"data/out/sonar/reconseeds/{name_a}_vs_{name_b}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "recon.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

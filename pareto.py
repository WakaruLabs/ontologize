"""Capacity-Pareto comparison: Ontologizer deviation codes vs top-k SAEs.

Puts both models on one reconstruction-vs-code-capacity curve, scored on
identical held-out rows (the cache tail that sae.py excludes from training)
under the same whitened FVU. Capacity is counted as continuous coefficients
transmitted per sample, with the index side-channel reported separately:

  Ontologizer codes (--code, both share the origin below):
    dev (default): per head, keep the m entries of the classification p
      with the largest |p - origin|, pin the rest to the origin
      distribution, renormalize. Coefficients = l*h*m, index bits =
      l*h*m*log2(k).
    head: per layer, keep the full soft distribution of the m
      lowest-entropy (most confident) heads and mean-ablate the rest --
      pin them to the origin. Selection is per sample and causal: each
      layer ranks its own heads given the already-ablated prefix, so the
      forward stays a single pass. Coefficients = l*m*k, index bits =
      l*m*log2(h). dev m=1 and head m=1 both cost 160 coefficients, so
      the two curves are directly comparable.
    both: emit both curves in one table/plot.
  The origin is the corpus-mean classification E[p] (a fixed constant of
  the model+corpus, measured here over --origin-rows training rows; the
  resid_nc probes showed it decodes to the corpus mean while the uniform
  code lands off-manifold). m=0 is the input-independent origin point;
  full m is the exact soft model; "hard" is end-to-end argmax (0
  coefficients, l*h*log2(k) bits) -- the discrete-ontology reading.

  SAE: coefficients = measured eval L0 (= topk for top-k runs), index bits
  = L0*log2(m latents). Runs are auto-discovered from data/out/sonar/sae/
  (or pass --sae paths to params.npz); m and topk are parsed from the
  m{m}_k{topk} / m{m}_l1{l1} run-directory names sae.py writes.

Caveats for the writeup: the top-m points are post-hoc truncations of a
model never trained to truncate, while each SAE was trained at exactly its
k (favors the SAE end); the Ontologizer trained on the full cache, so the
eval tail is held out only for the SAEs (favors the Ontologizer end).

  uv run python pareto.py                       # table + pareto.csv
  uv run python pareto.py --plot                # + pareto.png (log-log)
  uv run python pareto.py --origin uniform      # ablate the E[p] origin

Writes <out>/pareto.csv (label, coeffs, index_bits, fvu_w); default out
directory data/out/sonar/pareto.
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import functools
import json
import math
import re
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jaxtyping import Float
from pathlib import Path

import sae
from ontologize.ontologizer import Ontologizer
from ontologize.training.serialize import restore_spec


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="data/out/sonar/multilingual/resid_nc",
                   help="Ontologizer checkpoint directory")
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--ms", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 16],
                   help="deviations per head (dev) / heads per layer (head); "
                        "origin, soft, and hard points are always added")
    p.add_argument("--code", choices=["dev", "head", "both"], default="dev",
                   help="dev = top-m deviations per head; head = mean-ablate "
                        "all but the m lowest-entropy heads per layer")
    p.add_argument("--origin", choices=["meanp", "uniform"], default="meanp")
    p.add_argument("--origin-rows", type=int, default=65536,
                   help="training rows used to measure the E[p] origin")
    p.add_argument("--sae", nargs="*", default=None,
                   help="params.npz paths (default: data/out/sonar/sae/*/)")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="cache tail rows; must match the sae.py eval split")
    p.add_argument("--b", type=int, default=4096)
    p.add_argument("--out", default="data/out/sonar/pareto")
    p.add_argument("--plot", action="store_true",
                   help="write pareto.png (log-log FVU vs coefficients)")
    return p.parse_args()


def load_onto(ckpt, step):
    manager = ocp.CheckpointManager(
        Path(ckpt).resolve(),
        checkpointers={"state": ocp.PyTreeCheckpointer(),
                       "spec": ocp.PyTreeCheckpointer()})
    step = step or manager.latest_step()
    model = Ontologizer(**restore_spec(manager, step))
    state = manager.restore(step, items={"state": None})["state"]
    params = state["params"] if "opt_state" in state else state
    while "params" in params:
        params = params["params"]
    return model, params, step


def topdev(P, origin, m, k):
    """Keep the m largest |p - origin| entries per head, pin the rest to
    the origin distribution, renormalize. m=0 is the pure origin; m>=k is
    the identity."""
    if m == 0:
        Pm = jnp.broadcast_to(origin, P.shape)
    elif m >= k:
        return P
    else:
        D = jnp.abs(P - origin)
        v = jax.lax.top_k(D, m)[0][..., -1:]
        Pm = jnp.where(D >= v, P, jnp.broadcast_to(origin, P.shape))
    return Pm / (Pm.sum(-1, keepdims=True) + 1e-9)


def tophead(P, origin, m, h):
    """Mean-ablate all but the m lowest-entropy heads: per sample, the m
    heads with the most confident classification keep their full soft
    distribution, the rest are pinned to the origin. m=0 is the pure
    origin; m>=h is the identity. No renormalization needed -- both
    branches are already distributions."""
    if m == 0:
        return jnp.broadcast_to(origin, P.shape)
    if m >= h:
        return P
    H = -(P * jnp.log(P + 1e-9)).sum(-1)      # (b, h) sample entropy
    thr = -jax.lax.top_k(-H, m)[0][..., -1:]  # m-th lowest entropy
    return jnp.where((H <= thr)[..., None], P,
                     jnp.broadcast_to(origin, P.shape))


def parse_sae_name(path):
    """m and topk from sae.py's run-name convention (m{m}_k{k} / m{m}_l1{c})."""
    name = Path(path).parent.name
    match = re.fullmatch(r"m(\d+)_(?:k(\d+)|l1.+)", name)
    if not match:
        raise ValueError(
            f"can't parse m/topk from run dir {name!r}; expected the "
            "m{m}_k{topk} or m{m}_l1{l1} naming sae.py uses")
    return int(match.group(1)), int(match.group(2) or 0)


def whitened_spectrum(X: Float[np.ndarray, "n d"], w: Float[np.ndarray, "d"]
                      ) -> Float[np.ndarray, "d"]:
    """Eigenvalues of the covariance of x * sqrt(w), descending. Their sum
    is d times `main`'s FVU base, so FVU_w is a total distortion over it."""
    Y = (X - X.mean(0)) * np.sqrt(w)
    C = (Y.T @ Y).astype(np.float64) / len(Y)
    return np.clip(np.linalg.eigvalsh(C)[::-1], 0.0, None)


def gaussian_reference(lam: Float[np.ndarray, "d"],
                       bits: Float[np.ndarray, "r"]) -> Float[np.ndarray, "r"]:
    """The Gaussian reference FVU at each rate in `bits` (bits per sample):
    reverse water-filling over a Gaussian source with covariance
    eigenvalues `lam`, so FVU = sum min(theta, lam) / sum lam with the
    water level theta set by bits = sum max(0, log2(lam / theta) / 2). A
    reference, not a floor: a non-Gaussian source with the same covariance
    can be coded with less distortion."""
    lam = np.asarray(lam, np.float64)
    lam = lam[lam > 0]
    bits = np.atleast_1d(np.asarray(bits, np.float64))
    # the rate falls monotonically in theta: bisect log theta per target
    lo = np.full(bits.shape, np.log(lam.min()) - 60.0)
    hi = np.full(bits.shape, np.log(lam.max()))
    for _ in range(200):
        mid = (lo + hi) / 2
        rate = np.maximum(0.0, np.log2(lam[None] / np.exp(mid)[:, None])
                          / 2).sum(-1)
        above = rate > bits
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    theta = np.exp(hi)
    return np.minimum(theta[:, None], lam[None]).sum(-1) / lam.sum()


def onto_points(cfg, X_eval, w, base_w):
    model, params, step = load_onto(cfg.ckpt, cfg.step)
    l, h, k = model.l, model.h, model.k
    T = cfg.temperature
    print(f"onto: {cfg.ckpt} step {step} (l={l} h={h} k={k} T={T})")

    # Each layer's gain-shape split has to be reproduced here: under
    # `resid_gain` the DictEnc classifies the unit-norm SHAPE of its input
    # and scales its contribution by the measured GAIN. Classifying the raw
    # input and accumulating an unscaled contribution silently reconstructs
    # a different model. `gainshape_in` is the identity and `gained` a no-op
    # when the flag is off, so this is correct for either setting.
    def probe(module, X, m, hard, code, origin):
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        E_in = module.constinput(E)
        for i, dictenc in enumerate(module.dictencs):
            U, G = dictenc.gainshape_in(E_in)
            K = dictenc.classifier(U)
            if hard:
                P = jax.nn.one_hot(jnp.argmax(K, -1), k, dtype=K.dtype)
            elif code == "dev":
                P = topdev(dictenc.dict.cluster(K, T), origin[i], m, k)
            else:
                P = tophead(dictenc.dict.cluster(K, T), origin[i], m, h)
            R = R + dictenc.gained(
                    dictenc.dict.combine(dictenc.head_outputs(U, P)), G)
            if i < module.l - 1:
                E_in = module.nextinput(X, R, P.reshape(P.shape[0], -1))
        return module.decode(R)

    def meanp(module, X):
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        E_in = module.constinput(E)
        Ps = []
        for i, dictenc in enumerate(module.dictencs):
            U, G = dictenc.gainshape_in(E_in)
            P = dictenc.dict.cluster(dictenc.classifier(U), T)
            Ps.append(P.mean(0))
            R = R + dictenc.gained(
                    dictenc.dict.combine(dictenc.head_outputs(U, P)), G)
            if i < module.l - 1:
                E_in = module.nextinput(X, R, P.reshape(P.shape[0], -1))
        return jnp.stack(Ps)  # (l, h, k)

    if cfg.origin == "uniform":
        origin = jnp.full((l, h, k), 1.0 / k)
    else:
        # E[p] over training rows (the cache head; the eval tail stays
        # untouched so the origin isn't fit to the eval set)
        mm = np.load(cfg.cache, mmap_mode="r")
        run_mp = jax.jit(functools.partial(model.apply, {"params": params},
                                           method=meanp))
        acc = np.zeros((l, h, k))
        nb = 0
        for i in range(0, min(cfg.origin_rows, mm.shape[0]), cfg.b):
            X = jnp.asarray(np.asarray(mm[i:i + cfg.b], dtype=np.float32))
            acc += np.asarray(run_mp(X))
            nb += 1
        origin = jnp.asarray(acc / nb)

    run = jax.jit(functools.partial(model.apply, {"params": params},
                                    method=probe),
                  static_argnames=["m", "hard", "code"])

    def fvu(m, code="dev", hard=False):
        err = np.zeros(X_eval.shape[1])
        nb = 0
        for i in range(0, len(X_eval) - cfg.b + 1, cfg.b):
            X = jnp.asarray(X_eval[i:i + cfg.b])
            Y = run(X, m=m, hard=hard, code=code, origin=origin)
            err += np.asarray(((Y - X) ** 2).mean(0))
            nb += 1
        return ((err / nb) * w).mean() / base_w

    points = [(f"onto m=0 ({cfg.origin} origin)", 0, 0, fvu(0)),
              ("onto hard (argmax)", 0, round(l * h * math.log2(k)),
               fvu(0, hard=True))]
    if cfg.code in ("dev", "both"):
        for m in cfg.ms:
            if not 0 < m < k:
                continue
            points.append((f"onto dev m={m}", l * h * m,
                           round(l * h * m * math.log2(k)), fvu(m)))
    if cfg.code in ("head", "both"):
        for m in cfg.ms:
            if not 0 < m < h:
                continue
            points.append((f"onto heads={m}/{h}", l * m * k,
                           round(l * m * math.log2(h)), fvu(m, code="head")))
    points.append((f"onto m={k} (soft)", l * h * k, 0, fvu(k)))
    return points


def sae_points(cfg, X_eval, w_sqrt, base_w):
    paths = cfg.sae
    if paths is None:
        paths = sorted(Path("data/out/sonar/sae").glob("*/params.npz"))
    points = []
    for path in paths:
        meta_path = Path(path).parent / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            m, topk = meta["m"], meta["topk"]
            groups, group_fn = meta["groups"], meta["group_fn"]
        else:  # pre-meta.json run: recover the encode rule from the name
            m, topk = parse_sae_name(path)
            groups, group_fn = 0, "top1"
        params = {k_: jnp.asarray(v) for k_, v in np.load(path).items()}
        fvu, l0, *_ = sae.evaluate(sae.make_eval(topk, groups, group_fn),
                                   params, X_eval, w_sqrt, base_w, cfg.b)
        # index bits only apply to codes sparse enough to need addressing
        bits = 0 if round(l0) >= m else round(l0 * math.log2(m))
        points.append((f"sae {Path(path).parent.name}", round(l0), bits, fvu))
        print(f"sae: {path} (m={m} topk={topk} groups={groups})")
    return points


def main():
    cfg = parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    mm = np.load(cfg.cache, mmap_mode="r")
    X_eval = np.asarray(mm[-cfg.eval_rows:], dtype=np.float32)
    w = np.load(cfg.mse_weights) if cfg.mse_weights else np.ones(mm.shape[1])
    base_w = (X_eval.var(0) * w).mean()

    points = sae_points(cfg, X_eval, jnp.asarray(np.sqrt(w), jnp.float32),
                        base_w)
    points += onto_points(cfg, X_eval, w, base_w)
    points.sort(key=lambda r: (r[1], r[2]))

    print(f"\n{'point':<28} {'coeffs':>7} {'idx bits':>9} {'FVU_w':>8}")
    for label, coeffs, bits, fvu in points:
        print(f"{label:<28} {coeffs:>7} {bits:>9} {fvu:>8.4f}")

    with open(out / "pareto.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "coeffs", "index_bits", "fvu_w"])
        writer.writerows(points)
    print(f"\n-> {out / 'pareto.csv'}")

    if cfg.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, coeffs, _, fvu in points:
            if coeffs == 0:
                continue  # origin/hard points have no log-x position
            c, mk = (("tab:orange", "s") if label.startswith("sae") else
                     ("tab:green", "^") if label.startswith("onto heads") else
                     ("tab:blue", "o"))
            ax.scatter(coeffs, fvu, c=c, marker=mk)
            ax.annotate(label, (coeffs, fvu), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("continuous coefficients / sample")
        ax.set_ylabel("FVU (whitened)")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "pareto.png", dpi=150)
        print(f"-> {out / 'pareto.png'}")


if __name__ == "__main__":
    main()

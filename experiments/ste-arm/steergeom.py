"""Does a steering direction avoid the OTHER heads' decision boundaries?

`steerembed.py` finds that collateral is usually set by step size alone --
a decode step and a random step of equal length move the same share of
other heads -- with one exception, `g160top1`, whose decode step disturbs a
tenth of what random does. This measures why, in the linear regime, by
asking how a unit step along a direction moves each head's logits.

For a target (layer li, head hi, entry ki), on rows that do not already
select it, and for each direction D:

  off      mean |d logit| per unit step over the OTHER heads of layer li,
           as a ratio to a random direction's: below 1 means D avoids
           the other heads' detectors
  flip     share of other (row, head) pairs whose top-1 / top-2 margin a
           step of --strength along D would close, to first order: the
           linearized collateral, direct path only
  own      d(target logit - incumbent logit) per unit step, over the
           gap to close: the steps needed to realize the target, to
           first order

Only the target's own layer is scored, with every upstream code held
fixed: under residual forwarding an input step D moves that layer's
residual input by exactly D, and the cascade -- downstream heads
reclassifying a changed residual -- is excluded by construction. It is
the direct term `steerembed.py`'s collateral cannot separate from the
cascade.

Directions: `decode` (where forcing the entry moves the output), `grad`
(the classifier's own ascent direction on target minus incumbent, the
analogue of an SAE's encoder column), `random`. An `sae.py` params.npz is
scored the same way, a group standing in for a head (top1 and softmax
groups; ungrouped SAEs have no heads and are refused).

  uv run python experiments/ste-arm/steergeom.py \\
      --model data/out/sonar/multilingual/ste_h76_init01 --temperature 0.00015
  uv run python experiments/ste-arm/steergeom.py \\
      --model data/out/sonar/sae_conv/m5120_g160top1/params.npz
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

import sae
from pareto import load_onto


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Ontologizer checkpoint dir, or a grouped sae.py "
                        "params.npz")
    p.add_argument("--temperature", type=float, default=0.00015,
                   help="Ontologizer only")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--n-features", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--strength", type=float, default=0.25,
                   help="step for the linearized flip share, as a fraction "
                        "of each row's norm")
    p.add_argument("--ref-rows", type=int, default=8192)
    p.add_argument("--ref-b", type=int, default=512)
    p.add_argument("--min-rate", type=float, default=0.005)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


def unit(V: Float[Array, "b d"]) -> Float[Array, "b d"]:
    return V / (jnp.linalg.norm(V, axis=-1, keepdims=True) + 1e-9)


def score(L: Float[Array, "b H K"], dL: Float[Array, "b H K"], hi: int,
          ki: int, s: float, n: Float[np.ndarray, "b"]) -> Dict[str, float]:
    """The three measures for one direction, from the layer's logits and
    their derivative along it. `dL` is per unit step; `n` is each row's
    input norm, so `flip`'s step and `own`'s unit are fractions of |x|."""
    H = L.shape[1]
    other = np.ones(H, bool)
    other[hi] = False
    L, dL = np.asarray(L), np.asarray(dL) * n[:, None, None]
    off = float(np.abs(dL[:, other]).mean())
    order = np.argsort(-L, -1)
    w1, w2 = order[..., 0], order[..., 1]                       # (b, H)
    t = lambda A, i: np.take_along_axis(A, i[..., None], -1)[..., 0]
    margin = t(L, w1) - t(L, w2)
    dmargin = t(dL, w1) - t(dL, w2)
    flip = float((margin + s * dmargin < 0)[:, other].mean())
    inc = w1[:, hi]
    gap = t(L[:, hi], inc) - L[:, hi, ki]                      # > 0
    gain = dL[:, hi, ki] - t(dL[:, hi], inc)
    own = float(np.median(gain / np.maximum(gap, 1e-12)))
    return dict(off=off, flip=flip, own=own,
                margin=float(np.median(margin[:, other])))


def onto_setup(cfg: argparse.Namespace):
    model, raw, step = load_onto(cfg.model, cfg.step)
    params = {"params": raw}
    T = cfg.temperature
    if model.forward not in ("resid",) or model.encoded:
        raise SystemExit(
            f"{cfg.model}: needs forward='resid' and no encoder, so an input "
            f"step moves each layer's residual input one for one")

    def layer_in(module, X: Float[Array, "b d"], li: int) -> Array:
        """Layer li's classifier input with every upstream code fixed."""
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        for i in range(li):
            de = module.dictencs[i]
            U, G = de.gainshape_in(Ein)
            P = de.dict.cluster(de.classifier(U), T)
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
            Ein = module.nextinput(X, R, P.reshape(X.shape[0], -1))
        return Ein

    def logits(module, Ein: Array, li: int) -> Float[Array, "b h k"]:
        de = module.dictencs[li]
        U, _ = de.gainshape_in(Ein)
        return de.classifier(U)

    def forced(module, X: Float[Array, "b d"], li: int, hi: int,
               ki: int) -> Float[Array, "b d_out"]:
        args = [{} for _ in range(module.l)]
        args[li] = dict(h_set=jnp.array([hi]), k_set=jnp.array([ki]))
        return module.withArgs(X, args, temperature=T)[0]

    def base(module, X: Float[Array, "b d"]) -> Float[Array, "b d_out"]:
        args = [{} for _ in range(module.l)]
        return module.withArgs(X, args, temperature=T)[0]

    # every jitted function takes the parameters as an argument: closing
    # over them bakes the whole model into each executable as a constant,
    # and `forced`/`grad` compile once per target, which exhausts the GPU
    j_in = jax.jit(lambda p, X, li: model.apply(p, X, li, method=layer_in),
                   static_argnums=2)
    j_log = jax.jit(lambda p, E, li: model.apply(p, E, li, method=logits),
                    static_argnums=2)
    j_jvp = jax.jit(lambda p, E, V, li: jax.jvp(
        lambda e: model.apply(p, e, li, method=logits), (E,), (V,)),
        static_argnums=3)
    j_forced = jax.jit(lambda p, X, li, hi, ki: model.apply(
        p, X, li, hi, ki, method=forced), static_argnums=(2, 3, 4))
    j_base = jax.jit(lambda p, X: model.apply(p, X, method=base))

    def grad(p, E: Array, li: int, hi: int, inc: Int[Array, "b"],
             ki: Int[Array, ""]) -> Array:
        """Ascent on target minus incumbent at the classifier input."""
        def m(e):
            L = model.apply(p, e, li, method=logits)[:, hi]
            return (L[:, ki] - jnp.take_along_axis(
                L, inc[:, None], -1)[:, 0]).sum()
        return jax.grad(m)(E)

    j_grad = jax.jit(grad, static_argnums=(2, 3))
    f_in = lambda X, li: j_in(params, X, li)
    f_log = lambda E, li: j_log(params, E, li)
    f_jvp = lambda E, V, li: j_jvp(params, E, V, li)
    f_forced = lambda X, li, hi, ki: j_forced(params, X, li, hi, ki)
    f_base = lambda X: j_base(params, X)
    f_grad = lambda E, li, hi, ki, inc: j_grad(params, E, li, hi, inc,
                                               jnp.asarray(ki))
    shape = (model.l, model.h, model.k)
    d_out = model.d_out

    def argmaxes(X: Float[Array, "b d"]) -> Int[np.ndarray, "l b h"]:
        return np.stack([np.asarray(jnp.argmax(f_log(f_in(X, li), li), -1))
                         for li in range(model.l)])

    def target(X, feat):
        li, hi, ki = feat
        E = f_in(X, li)
        L = f_log(E, li)
        inc = jnp.argmax(L[:, hi], -1)
        pad = lambda D: jnp.pad(D, ((0, 0), (0, E.shape[-1] - D.shape[-1])))
        dirs = {
            "decode": pad(unit(f_forced(X, li, hi, ki) - f_base(X))),
            "grad": unit(f_grad(E, li, hi, ki, inc)[..., :d_out]),
            "random": pad(unit(jax.random.normal(
                jax.random.PRNGKey(int(ki) + 1000 * li + 31 * hi),
                (X.shape[0], d_out)))),
        }
        dirs["grad"] = pad(dirs["grad"])
        return L, {k: f_jvp(E, D, li)[1] for k, D in dirs.items()}, (hi, ki)

    name = f"{Path(cfg.model).name}_{step}"
    return shape, argmaxes, target, name


def sae_setup(cfg: argparse.Namespace):
    path = Path(cfg.model)
    meta = json.loads((path.parent / "meta.json").read_text())
    G = meta["groups"]
    if not G or "W_enc" not in np.load(path):
        raise SystemExit(f"{path}: needs a grouped SAE with a linear encoder")
    m = meta["m"]
    gs = m // G
    params = {k: jnp.asarray(v) for k, v in np.load(path).items()}
    W_enc, W_dec = params["W_enc"], params["W_dec"]

    f_pre = jax.jit(lambda X: sae.preacts(params, X).reshape(
        X.shape[0], G, gs))

    def argmaxes(X: Float[Array, "b d"]) -> Int[np.ndarray, "l b h"]:
        return np.asarray(jnp.argmax(f_pre(X), -1))[None]

    def target(X, feat):
        _, g, e = feat
        j = g * gs + e
        L = f_pre(X)
        inc = np.asarray(jnp.argmax(L[:, g], -1))
        dlog = lambda D: (D @ W_enc).reshape(X.shape[0], G, gs)
        grad = W_enc[:, j][None] - W_enc[:, g * gs + inc].T
        dirs = {
            "decode": unit(jnp.broadcast_to(W_dec[j], X.shape)),
            "grad": unit(grad),
            "random": unit(jax.random.normal(jax.random.PRNGKey(j), X.shape)),
        }
        return L, {k: dlog(D) for k, D in dirs.items()}, (g, e)

    return (1, G, gs), argmaxes, target, f"sae_{path.parent.name}"


def main() -> None:
    cfg = parse_args()
    setup = sae_setup if cfg.model.endswith(".npz") else onto_setup
    (l, h, k), argmaxes, target, name = setup(cfg)
    out = Path(cfg.out or f"data/out/sonar/steergeom/{name}")
    out.mkdir(parents=True, exist_ok=True)

    mm = np.load(cfg.cache, mmap_mode="r")
    Xref = jnp.asarray(np.asarray(mm[-cfg.ref_rows:], np.float32))
    A_ref = np.concatenate([argmaxes(Xref[i:i + cfg.ref_b])
                            for i in range(0, Xref.shape[0], cfg.ref_b)],
                           axis=1)                               # (l, b, h)

    rng = np.random.default_rng(cfg.seed)
    feats: List[Tuple[int, int, int]] = []
    per = max(1, cfg.n_features // l)
    for li in range(l):
        pool = [(li, hi, ki) for hi in range(h) for ki in range(k)
                if cfg.min_rate <= (A_ref[li, :, hi] == ki).mean()
                and (A_ref[li, :, hi] != ki).sum() >= cfg.n_samples]
        feats += [pool[i] for i in rng.choice(
            len(pool), min(per, len(pool)), replace=False)]
    print(f"{name}: {l}x{h} heads, k={k}; {len(feats)} targets, "
          f"{cfg.n_samples} rows each, flip at strength {cfg.strength}\n")

    rows = []
    for feat in feats:
        li = feat[0]
        idx = np.where(A_ref[li, :, feat[1]] != feat[2])[0]
        idx = rng.choice(idx, cfg.n_samples, replace=False)
        X = Xref[jnp.asarray(idx)]
        L, dLs, (hi, ki) = target(X, feat)
        n = np.linalg.norm(np.asarray(X), axis=-1)
        for kind, dL in dLs.items():
            rows.append(dict(kind=kind, layer=li, **score(
                L, dL, hi, ki, cfg.strength, n)))

    summary: Dict[str, Dict[str, float]] = {}
    print(f"{'layer':>6} {'kind':>7} {'off/rand':>9} {'flip':>7} "
          f"{'own':>9} {'margin':>9}")
    for li in list(range(l)) + ["all"]:
        R0 = [r for r in rows if li == "all" or r["layer"] == li]
        rand = np.mean([r["off"] for r in R0 if r["kind"] == "random"])
        for kind in ("decode", "grad", "random"):
            R = [r for r in R0 if r["kind"] == kind]
            s = {"off_over_random": float(np.mean([r["off"] for r in R]) / rand),
                 "flip": float(np.mean([r["flip"] for r in R])),
                 "own": float(np.median([r["own"] for r in R])),
                 "margin": float(np.median([r["margin"] for r in R]))}
            summary[f"{kind}@L{li}"] = s
            print(f"{str(li):>6} {kind:>7} {s['off_over_random']:9.3f} "
                  f"{s['flip']:7.3f} {s['own']:9.2f} {s['margin']:9.2e}")
    (out / "summary.json").write_text(json.dumps(
        {"model": cfg.model, "n_features": len(feats),
         "n_samples": cfg.n_samples, "strength": cfg.strength,
         "results": summary}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

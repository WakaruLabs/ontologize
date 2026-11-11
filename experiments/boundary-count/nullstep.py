"""Can an input step flip one head and leave the others' margins alone?

`steerembed.py` finds that every direction it tries -- decode, gradient,
margin, adjoint -- flips other heads at a random step's rate. None of
those directions is chosen with the other heads in view. This script
builds the step that is: the shortest step that moves the target head's
margin across its boundary while holding other heads' margins fixed to
first order, and asks whether it exists and what it does.

Whether it exists is a count. A head with k entries has k - 1 margins
(winner minus each other entry) and flips when any of them crosses zero.
Holding c margins fixed is c linear constraints on a d-dimensional step,
so it can be done exactly only while c < d:

  one layer of h heads     h (k - 1) - (k - 1) constraints
  the whole stack          l h (k - 1) - (k - 1)

The margins are taken with every head's code frozen at the clean pass, so
the Jacobian is that of the model between flips: the classifier's own
gradient, the normalization of the layer input, and the gains of the
layers below (a layer's input is the residual, which moves with the gains
even when no code changes). The module stop-gradients the residual, so
`frozen_logits` rebuilds the hard forward from the module's own parts
without the stop, and `main` checks it against the module's logits.

For each sampled row and one target head per layer, the target is the
head's runner-up entry and the step moves the winner-minus-runner-up
margin m to -(--overshoot - 1) m. Steps:

  grad         the shortest step that does that, with no constraint
  null_layer   holding every margin of the layer's other heads
  null_ru      holding only their runner-up margins (the nearest boundary
               of each; the same set as null_layer when k = 2)
  null_upto    holding the runner-up margins of every other head in the
               target's layer and the layers below it, whose flips would
               change the target layer's input
  null_stack   holding the runner-up margins of every other head in the
               stack
  soft_layer   when the constraints outnumber the dimensions: the step
  soft_stack   that minimizes the sum of squared RELATIVE margin changes,
               (dm / m)^2, over the layer's or the stack's other margins

Each is scored beside a random step of its own length (and, with
--strengths, both rescaled to a fixed share of |x|, as `steerembed.py`
steps): `hit`, the target head now selects the runner-up; `col_layer`,
the share of the layer's other heads that changed; `col_stack`, of every
other head; and `lin_layer`, the share the first-order model predicts for
the layer.

`do` is the intervention itself: the target head set to its runner-up
with the input untouched. The layers above reclassify what it writes, so
its `col_stack` is the collateral any step that realizes the flip must
cause, and a step's excess over it is what the step adds.

  uv run python experiments/boundary-count/nullstep.py \\
      --model data/out/sonar/multilingual/ste_k2_h380
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from pareto import load_onto


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint dir")
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--temperature", type=float, default=0.00015,
                   help="irrelevant under select='ste', whose cluster is "
                        "the argmax")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--tail", type=int, default=32768,
                   help="rows are drawn from the cache's last TAIL rows")
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--overshoot", type=float, default=1.5,
                   help="the target margin m is moved by -OVERSHOOT * m")
    p.add_argument("--strengths", type=float, nargs="*", default=[0.25],
                   help="also score every direction at these shares of |x|")
    p.add_argument("--ridge", type=float, default=1e-6,
                   help="soft steps: lambda |d|^2, as a share of the mean "
                        "diagonal of the weighted Gram")
    p.add_argument("--jac-batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


# ---------- linear algebra (pure) ----------

def margins(K: Float[np.ndarray, "h k"]
            ) -> Tuple[Int[np.ndarray, "h"], Int[np.ndarray, "h"],
                       Float[np.ndarray, "h k"]]:
    """Each head's winner, its runner-up, and its margins: the winner's
    logit minus every entry's (zero at the winner, positive elsewhere)."""
    order = np.argsort(-K, axis=1)
    win, ru = order[:, 0], order[:, 1]
    return win, ru, K[np.arange(len(K)), win][:, None] - K


def margin_jacobian(dK: Float[np.ndarray, "h k d"], win: Int[np.ndarray, "h"]
                    ) -> Float[np.ndarray, "h k d"]:
    """Jacobian of `margins`' third output with the winners held."""
    return dK[np.arange(len(dK)), win][:, None, :] - dK


def constrained_step(g: Float[np.ndarray, "d"], t: float,
                     C: Float[np.ndarray, "c d"]
                     ) -> Tuple[Optional[Float[np.ndarray, "d"]], float]:
    """The shortest step with `g . step = t` and `C step = 0`, and the
    share of |g|^2 outside the span of `C`'s rows.

    The step is `t P g / (g . P g)` for `P` the projector off that span.
    When the share is numerically zero the constraints leave no direction
    that moves `g`, and the step is None."""
    if len(C) == 0:
        return t * g / (g @ g), 1.0
    Q, _ = np.linalg.qr(C.T)
    Pg = g - Q @ (Q.T @ g)
    # a second pass removes what cancellation left of the first
    Pg = Pg - Q @ (Q.T @ Pg)
    share = float(Pg @ Pg) / float(g @ g)
    if share < 1e-9:
        return None, share
    return t * Pg / (g @ Pg), share


def weighted_step(g: Float[np.ndarray, "d"], t: float,
                  C: Float[np.ndarray, "c d"], w: Float[np.ndarray, "c"],
                  ridge: float) -> Float[np.ndarray, "d"]:
    """The step with `g . step = t` that minimizes
    `sum_c w_c (C_c . step)^2 + lam |step|^2`, with `lam` equal to `ridge`
    times the mean diagonal of `C' diag(w) C`.

    Stationarity gives `step = t A^{-1} g / (g' A^{-1} g)` for
    `A = C' diag(w) C + lam I`."""
    A = (C * w[:, None]).T @ C
    A[np.diag_indices_from(A)] += ridge * np.trace(A) / len(A)
    s = np.linalg.solve(A, g)
    return t * s / (g @ s)


def crossings(m: Float[np.ndarray, "h k"], J: Float[np.ndarray, "h k d"],
              step: Float[np.ndarray, "d"]) -> Bool[np.ndarray, "h"]:
    """Heads the first-order model flips: some margin, moved by
    `J . step`, falls below zero. The winner's own margin stays zero."""
    return ((m + J @ step) < 0).any(1)


def collateral(A: Int[np.ndarray, "l h"], A0: Int[np.ndarray, "l h"],
               li: int, hi: int) -> Tuple[float, float]:
    """Share of the target layer's other heads, and of every other head,
    whose selection changed."""
    ch = A != A0
    layer = (ch[li].sum() - ch[li, hi]) / (ch.shape[1] - 1)
    stack = (ch.sum() - ch[li, hi]) / (ch.size - 1)
    return float(layer), float(stack)


# ---------- model ----------

def clean_logits(module, X: Float[Array, "b d_in"], T: float
                 ) -> Float[Array, "l b h k"]:
    """Every layer's classifier logits on a clean hard forward, by the
    module's own path."""
    E, _ = module.encode(X, 0.0, None)
    R = module.resid(E)
    Ein = module.constinput(E)
    Ks = []
    for i, de in enumerate(module.dictencs):
        U, G = de.gainshape_in(Ein)
        K = de.classifier(U)
        P = de.dict.cluster(K, T)
        Ks.append(K)
        R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
        if i < module.l - 1:
            Ein = module.nextinput(X, R, P.reshape(X.shape[0], -1))
    return jnp.stack(Ks)


def frozen_logits(module, x: Float[Array, "d_in"],
                  P: Float[Array, "l h k"]) -> Float[Array, "l h k"]:
    """Every layer's logits at `x` with the codes held at `P`: the hard
    forward between flips, as a differentiable function of `x`.

    The module's forward stops the gradient at each layer's residual input
    and at its gain; here both follow `x`, since a step changes them
    whether or not a code flips. Only the selections are constants."""
    X = x[None]
    E, _ = module.encode(X, 0.0, None)
    R = module.resid(E)
    D = E
    eps = jnp.finfo(module.dtype).eps
    Ks = []
    for i, de in enumerate(module.dictencs):
        n = jnp.linalg.norm(D, axis=-1, keepdims=True)
        U = D / (n + eps)
        if de.n_const:
            U = jnp.concatenate(
                [U, jnp.ones(U.shape[:-1] + (de.n_const,), U.dtype)], -1)
        Ks.append(de.classifier(U)[0])
        Y = de.dict.combine(de.head_outputs(jax.lax.stop_gradient(U),
                                            P[i][None]))
        R = R + de.place(Y * n)
        D = X.astype(module.dtype) - module.decode(R)
    return jnp.stack(Ks)


def forced_codes(module, X: Float[Array, "b d_in"], T: float,
                 li: Int[Array, "b"], hi: Int[Array, "b"],
                 ki: Int[Array, "b"]) -> Int[Array, "l b h"]:
    """Every head's selection on a hard forward in which row `n`'s head
    `hi[n]` of layer `li[n]` is set to entry `ki[n]`: the intervention an
    input step is trying to realize, with nothing else touched. What it
    changes downstream is what any step that realizes it must change."""
    E, _ = module.encode(X, 0.0, None)
    R = module.resid(E)
    Ein = module.constinput(E)
    A = []
    for i, de in enumerate(module.dictencs):
        U, G = de.gainshape_in(Ein)
        P = de.dict.cluster(de.classifier(U), T)
        here = ((li == i)[:, None]
                & (jnp.arange(P.shape[1])[None] == hi[:, None]))[..., None]
        P = jnp.where(here, jax.nn.one_hot(ki, P.shape[-1],
                                           dtype=P.dtype)[:, None], P)
        A.append(jnp.argmax(P, -1))
        R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
        if i < module.l - 1:
            Ein = module.nextinput(X, R, P.reshape(X.shape[0], -1))
    return jnp.stack(A)


def check_model(model) -> None:
    """`frozen_logits` rebuilds one forward; refuse the others."""
    bad = []
    if model.forward != "resid" or not model.resid_gain:
        bad.append("forward='resid' with resid_gain")
    if not (model.resid_const and model.const0):
        bad.append("resid_const with const0")
    if model.scaled or model.fiber_rank:
        bad.append("no router and no fibers")
    if model.select not in ("ste", "argmax"):
        bad.append("hard selection")
    if getattr(model, "encoded", False):
        bad.append("no encoder")
    if bad:
        raise ValueError("nullstep.py needs " + "; ".join(bad))


# ---------- the experiment ----------

def steps_for(li: int, hi: int, m: Float[np.ndarray, "l h k"],
              J: Float[np.ndarray, "l h k d"], win: Int[np.ndarray, "l h"],
              ru: Int[np.ndarray, "l h"], overshoot: float, ridge: float
              ) -> Tuple[Dict[str, Float[np.ndarray, "d"]],
                         Dict[str, float]]:
    """Every step for one target head, by name, and the share of the
    target gradient each exact constraint set leaves free."""
    l, h, k, d = J.shape
    g = J[li, hi, ru[li, hi]]
    t = -overshoot * m[li, hi, ru[li, hi]]
    other = np.ones((l, h), bool)
    other[li, hi] = False
    nonwin = np.ones((l, h, k), bool)
    nonwin[np.arange(l)[:, None], np.arange(h)[None], win] = False
    lay = np.zeros((l, h), bool)
    lay[li] = True
    upto = np.zeros((l, h), bool)
    upto[:li + 1] = True

    def rows(heads: Bool[np.ndarray, "l h"], runner_only: bool
             ) -> Tuple[Float[np.ndarray, "c d"], Float[np.ndarray, "c"]]:
        if runner_only:
            a, b = np.nonzero(heads)
            return J[a, b, ru[a, b]], m[a, b, ru[a, b]]
        sel = heads[:, :, None] & nonwin
        return J[sel], m[sel]

    out = {"grad": t * g / (g @ g)}
    share = {}
    sets = {"null_layer": (lay & other, False),
            "null_ru": (lay & other, True),
            "null_upto": (upto & other, True),
            "null_stack": (other, True)}
    for name, (heads, runner_only) in sets.items():
        if name == "null_ru" and k == 2:
            continue                        # the same rows as null_layer
        if name == "null_upto" and li in (0, l - 1):
            continue                        # null_ru's rows, or null_stack's
        C, _ = rows(heads, runner_only)
        if len(C) >= d:
            share[name] = 0.0               # more constraints than dimensions
            continue
        step, share[name] = constrained_step(g, t, C)
        if step is not None:
            out[name] = step
    for name, heads in (("soft_layer", lay & other), ("soft_stack", other)):
        C, mc = rows(heads, False)
        out[name] = weighted_step(g, t, C, 1.0 / np.maximum(mc, 1e-12) ** 2,
                                  ridge)
    return out, share


def main() -> None:
    cfg = parse_args()
    model, raw, step = load_onto(cfg.model, cfg.step)
    check_model(model)
    params = {"params": raw}
    l, h, k, d = model.l, model.h, model.k, model.d_in
    T = cfg.temperature
    out = Path(cfg.out or f"data/out/sonar/boundary/{Path(cfg.model).name}")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    mm = np.load(cfg.cache, mmap_mode="r")
    idx = np.sort(rng.choice(cfg.tail, cfg.rows, replace=False))
    X = np.asarray(mm[len(mm) - cfg.tail + idx], np.float32)

    f_clean = jax.jit(lambda p, x: model.apply(p, x, T, method=clean_logits))
    f_frozen = jax.jit(jax.vmap(
        lambda p, x, P: model.apply(p, x, P, method=frozen_logits),
        in_axes=(None, 0, 0)))
    f_jac = jax.jit(jax.vmap(
        jax.jacfwd(lambda x, p, P: model.apply(p, x, P, method=frozen_logits)),
        in_axes=(0, None, 0)))
    f_do = jax.jit(lambda p, x, li, hi, ki: model.apply(
        p, x, T, li, hi, ki, method=forced_codes))
    f_arg = jax.jit(lambda p, x: jnp.argmax(
        model.apply(p, x, T, method=clean_logits), -1))

    K0 = np.asarray(f_clean(params, jnp.asarray(X))).transpose(1, 0, 2, 3)
    A0 = K0.argmax(-1)                                         # (n, l, h)
    P0 = np.eye(k, dtype=np.float32)[A0]                       # (n, l, h, k)
    Kf = np.asarray(f_frozen(params, jnp.asarray(X), jnp.asarray(P0)))
    scale = np.abs(K0).max()
    err = np.abs(Kf - K0).max() / scale
    print(f"{Path(cfg.model).name} step {step}: {l}x{h} heads, k={k}, d={d}")
    print(f"frozen forward against the module's logits: max relative "
          f"error {err:.1e}")
    if err > 1e-3:
        raise RuntimeError("frozen_logits does not reproduce the module")
    print(f"margins to hold: one layer {(h - 1) * (k - 1)}"
          f" ({h - 1} runner-up), the stack {(l * h - 1) * (k - 1)}"
          f" ({l * h - 1} runner-up); dimensions {d}\n")

    recs: List[Dict[str, Any]] = []
    shares: List[Dict[str, Any]] = []
    for s0 in range(0, cfg.rows, cfg.jac_batch):
        sl = slice(s0, s0 + cfg.jac_batch)
        dK = np.asarray(f_jac(jnp.asarray(X[sl]), params,
                              jnp.asarray(P0[sl])), np.float64)
        for r in range(dK.shape[0]):
            n = s0 + r
            x = X[n].astype(np.float64)
            xn = np.linalg.norm(x)
            Kn = K0[n].astype(np.float64)
            win = np.empty((l, h), int)
            ru = np.empty((l, h), int)
            m = np.empty((l, h, k))
            J = np.empty((l, h, k, d))
            for i in range(l):
                win[i], ru[i], m[i] = margins(Kn[i])
                J[i] = margin_jacobian(dK[r, i], win[i])
            names, batch, meta = [], [], []
            for li in range(l):
                hi = int(rng.integers(h))
                steps, share = steps_for(li, hi, m, J, win, ru,
                                         cfg.overshoot, cfg.ridge)
                shares.append(dict(row=n, layer=li, head=hi, **share))
                for name, st in steps.items():
                    z = rng.standard_normal(d)
                    z *= np.linalg.norm(st) / np.linalg.norm(z)
                    for kind, v in ((name, st), (name + "/random", z)):
                        for s in [None] + list(cfg.strengths):
                            vv = v if s is None else v * (
                                s * xn / np.linalg.norm(v))
                            batch.append(x + vv)
                            meta.append(dict(
                                row=n, layer=li, head=hi, kind=kind,
                                strength="native" if s is None else s,
                                norm=float(np.linalg.norm(vv) / xn),
                                lin_layer=float(
                                    np.delete(crossings(m[li], J[li], vv),
                                              hi).mean())))
                    names.append(name)
            Xb = np.asarray(batch, np.float32)
            Xb = Xb * (xn / np.linalg.norm(Xb, axis=1, keepdims=True))
            # the intervention itself, one row per target
            tg = [(mt["layer"], mt["head"]) for mt in meta
                  if mt["kind"] == "grad" and mt["strength"] == "native"]
            Ad = np.asarray(f_do(
                params, jnp.asarray(np.repeat(X[n][None], len(tg), 0)),
                jnp.asarray([a for a, _ in tg]),
                jnp.asarray([b for _, b in tg]),
                jnp.asarray([ru[a, b] for a, b in tg]))).transpose(1, 0, 2)
            for (li, hi), a in zip(tg, Ad):
                cl, cs = collateral(a, A0[n], li, hi)
                recs.append(dict(row=n, layer=li, head=hi, kind="do",
                                 strength="native", norm=0.0, lin_layer=0.0,
                                 hit=1.0, col_layer=cl, col_stack=cs))
                recs.append(dict(recs[-1], kind="do/random", hit=0.0,
                                 col_layer=0.0, col_stack=0.0))
            A = np.asarray(f_arg(params, jnp.asarray(Xb))).transpose(1, 0, 2)
            for a, mt in zip(A, meta):
                li, hi = mt["layer"], mt["head"]
                cl, cs = collateral(a, A0[n], li, hi)
                recs.append(dict(mt, hit=float(a[li, hi] == ru[li, hi]),
                                 col_layer=cl, col_stack=cs))
        print(f"  rows {min(s0 + cfg.jac_batch, cfg.rows)}/{cfg.rows}",
              end="\r", flush=True)

    with open(out / "steps.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(recs[0]))
        wr.writeheader()
        wr.writerows(recs)

    summary = summarize(recs, shares, l)
    print_summary(summary)
    (out / "summary.json").write_text(json.dumps(
        {"model": str(cfg.model), "step": int(step), "l": l, "h": h, "k": k,
         "d": d, "rows": cfg.rows, "overshoot": cfg.overshoot,
         "frozen_forward_error": float(err), **summary}, indent=2))
    print(f"\n-> {out}")


def summarize(recs: List[Dict[str, Any]], shares: List[Dict[str, Any]],
              l: int) -> Dict[str, Any]:
    """Means by (strength, kind), over all layers and by target layer, and
    each exact constraint set's free share of the target gradient."""
    def mean(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        if not rows:
            return {"n": 0}
        return {f: float(np.mean([r[f] for r in rows]))
                for f in ("norm", "hit", "col_layer", "col_stack",
                          "lin_layer")} | {"n": len(rows)}

    table: Dict[str, Any] = {}
    for r in recs:
        table.setdefault(str(r["strength"]), {}).setdefault(
            r["kind"], []).append(r)
    res = {s: {kind: {"all": mean(rows),
                      **{f"layer{i}": mean([r for r in rows
                                            if r["layer"] == i])
                         for i in range(l)}}
               for kind, rows in kinds.items()}
           for s, kinds in table.items()}
    free = {}
    for name in ("null_layer", "null_ru", "null_upto", "null_stack"):
        for i in range(l):
            v = [s[name] for s in shares if name in s and s["layer"] == i]
            if v:
                free.setdefault(name, {})[f"layer{i}"] = {
                    "median_free_share": float(np.median(v)),
                    "feasible": float(np.mean(np.array(v) >= 1e-9))}
    return {"results": res, "free": free}


def print_summary(summary: Dict[str, Any]) -> None:
    print("share of the target gradient each exact constraint set leaves "
          "free (median), and how often the step exists, by target layer")
    for name, by in summary["free"].items():
        print(f"  {name:<11}" + "  ".join(
            f"{lay[-1]}: {v['median_free_share']:.3f} ({v['feasible']:.0%})"
            for lay, v in by.items()))
    for s, kinds in summary["results"].items():
        print(f"\nstrength {s}: step / a random step of its length")
        print(f"  {'step':<11}{'target':>7}{'|step|/|x|':>11}{'hit':>12}"
              f"{'layer':>14}{'stack':>14}{'lin':>14}")
        for kind, v in kinds.items():
            if kind.endswith("/random"):
                continue
            for where, a in v.items():
                if not a["n"]:
                    continue
                b = kinds[kind + "/random"][where]
                print(f"  {kind:<11}{where:>7}{a['norm']:>11.4f}"
                      + "".join(f"{a[f]:>8.3f}/{b[f]:.3f}" for f in
                                ("hit", "col_layer", "col_stack",
                                 "lin_layer")))


if __name__ == "__main__":
    main()

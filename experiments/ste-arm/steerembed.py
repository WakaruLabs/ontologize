"""Steering scored in embedding space, without the text round trip.

`steerfid.py` reads an intervention's effect off a re-encoding of the
generated text. That instrument is weak here: on an UNSTEERED round trip
only ~12% of head argmaxes survive (4.6% by layer 4, against 3.1%
chance), and the cycled embedding sits at cosine 0.38 to its origin
where two unrelated corpus embeddings sit at 0.31. A perfect
intervention could therefore be observed at most ~12% of the time, which
is the same order as the hit rates that metric reports -- so it cannot
separate a failed intervention from an unobservable one.

Dropping generation removes that ceiling. The question a steering claim
actually rests on is causal and lives in embedding space: move the
input, re-run the model, and ask whether the intended entry is now
selected and whether anything else moved.

  realization  P(head h selects entry k) after steering, against the
               rate before steering and against a random direction of
               the same magnitude. This is steerfid's "hit" without the
               cycle.
  collateral   the share of OTHER heads whose argmax changed. The
               embedding-space analogue of 1 - chrF, and a sharper one:
               a steering claim is that ONE variable moved, and this
               counts the ones that moved by accident.

Two directions, since they are not the same object and steerfid uses
only the first:

  decode    x + m * unit(Y_forced - Y_base): where the tag DECODES to.
  grad      the first-order direction that raises the target entry's
            logit and lowers the incumbent's. An unbiased bilinear logit
            is x'Bx, whose derivative is 2Bx, so this is the classifier's
            own Jacobian at the sample: (V.u)W + (W.u)V differenced
            between the two entries. Needs no eigendecomposition.
  adjoint   `Ontologizer.intervene`'s path, via `BilinearBlock.rev`'s
            top-1 eigenvector. Note that selects on |eigenvalue| and
            drops its sign, while ascent direction depends on it; and an
            eigenvector is the dominant QUADRATIC direction, not the
            first-order one, so it is only the right answer near x = 0.
  random    a random unit direction, the control curve.

  uv run python experiments/ste-arm/steerembed.py \\
      --model data/out/sonar/multilingual/ste_h76 --temperature 0.00015
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ontologize.layers.nlinear import orient
from ontologize.ontologizer import Ontologizer
from pareto import load_onto


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--n-features", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=32,
                   help="held-out rows steered per feature")
    p.add_argument("--strengths", type=float, nargs="+",
                   default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--ref-rows", type=int, default=8192,
                   help="rows for the baseline firing rates")
    p.add_argument("--ref-b", type=int, default=512,
                   help="rows per forward in the baseline pass")
    p.add_argument("--min-rate", type=float, default=0.005,
                   help="skip entries the model almost never selects")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    model, raw, step = load_onto(cfg.model, cfg.step)
    params = {"params": raw}
    l, h, k = model.l, model.h, model.k
    T = cfg.temperature
    out = Path(cfg.out or f"data/out/sonar/steerembed/"
                          f"{Path(cfg.model).name}_{step}")
    out.mkdir(parents=True, exist_ok=True)

    mm = np.load(cfg.cache, mmap_mode="r")
    Xref = jnp.asarray(np.asarray(mm[-cfg.ref_rows:], np.float32))

    def codes(module, X: Float[Array, "b d_in"]
              ) -> Tuple[Float[Array, "l b h k"], Float[Array, "b d_out"]]:
        """Per-layer classifications of a clean forward, (l, b, h, k)."""
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        Ps = []
        for i, de in enumerate(module.dictencs):
            U, G = de.gainshape_in(Ein)
            P = de.dict.cluster(de.classifier(U), T)
            Ps.append(P)
            R = R + de.gained(de.dict.combine(de.dict.hfwd(P)), G)
            if i < module.l - 1:
                Ein = module.nextinput(X, R, P.reshape(X.shape[0], -1))
        return jnp.stack(Ps), module.decode(R)

    f_codes = jax.jit(lambda p, X: model.apply(p, X, method=codes))
    f_arg = jax.jit(lambda p, X: jnp.argmax(
        model.apply(p, X, method=codes)[0], -1))
    # the baseline pass only keeps the argmax, but `hfwd` on the way there
    # is (rows, h, e_dec): whole-reference-set at a few hundred heads is
    # tens of gigabytes, so take it a chunk of rows at a time
    A_ref = np.concatenate(                                       # (l, b, h)
        [np.asarray(f_arg(params, Xref[i:i + cfg.ref_b]))
         for i in range(0, Xref.shape[0], cfg.ref_b)], axis=1)

    # entries the model actually uses, so "make it select this" is possible
    rng = np.random.default_rng(cfg.seed)
    live = [(li, hi, ki) for li in range(l) for hi in range(h) for ki in range(k)
            if (A_ref[li, :, hi] == ki).mean() >= cfg.min_rate]
    # stratify by layer: uniform sampling lands few features in layer 0
    # (its usage is the most concentrated, so fewest entries clear
    # --min-rate), and the layers differ sharply in how a steering
    # direction reaches them
    per = max(1, cfg.n_features // l)
    feats = []
    for li in range(l):
        pool = [f for f in live if f[0] == li]
        if pool:
            feats += [pool[i] for i in rng.choice(
                len(pool), min(per, len(pool)), replace=False)]
    print(f"{Path(cfg.model).name} step {step}: {l}x{h} heads, k={k}, T={T:g}")
    print(f"{len(live)} of {l*h*k} entries fire on >= {cfg.min_rate:.1%} of "
          f"rows; steering {len(feats)} of them, {cfg.n_samples} rows each\n")

    def forced_decode(module, X: Float[Array, "b d_in"], li: int, hi: int,
                      ki: int) -> Float[Array, "b d_out"]:
        """Y with layer li's head hi forced to entry ki, and Y itself."""
        args = [{} for _ in range(module.l)]
        args[li] = dict(h_set=jnp.array([hi]), k_set=jnp.array([ki]))
        Yi, _, _, _ = module.withArgs(X, args, temperature=T)
        return Yi

    f_forced = jax.jit(
        lambda p, X, li, hi, ki: model.apply(
            p, X, li, hi, ki, method=forced_decode),
        static_argnums=(2, 3, 4))

    def layer_input(module, X: Float[Array, "b d_in"], li: int
                    ) -> Tuple[Any, Float[Array, "b d"],
                               Float[Array, "b h k"]]:
        """The DictEnc, its shaped classifier input, and its P."""
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        for i in range(li):
            de = module.dictencs[i]
            U, G = de.gainshape_in(Ein)
            P = de.dict.cluster(de.classifier(U), T)
            R = R + de.gained(de.dict.combine(de.dict.hfwd(P)), G)
            Ein = module.nextinput(X, R, P.reshape(X.shape[0], -1))
        de = module.dictencs[li]
        U, _ = de.gainshape_in(Ein)
        return de, U, de.dict.cluster(de.classifier(U), T)

    def grad_dir(module, X: Float[Array, "b d_in"], li: int, hi: int,
                 ki: int) -> Float[Array, "b d_out"]:
        """First-order ascent on the target entry's logit, descent on the
        incumbent's: the classifier Jacobian for head hi alone."""
        de, U, P = layer_input(module, X, li)
        W, V = de.classifier.weight_ubind()          # (h, k, d_in) each
        Wh, Vh = W[hi], V[hi]
        uW = jnp.einsum("bd,kd->bk", U, Wh)
        uV = jnp.einsum("bd,kd->bk", U, Vh)
        J = (uV[..., None] * Wh + uW[..., None] * Vh)   # (b, k, d_in)
        cur = jnp.argmax(P[:, hi, :], -1)
        D = J[:, ki, :] - jnp.take_along_axis(
            J, cur[:, None, None], axis=1).squeeze(1)
        return D[..., :module.d_out]

    def margin_dir(module, X: Float[Array, "b d_in"], li: int, hi: int,
                   ki: int) -> Float[Array, "b d_out"]:
        """Best direction in a ball for flipping the argmax: the top
        eigenvector of the DIFFERENCE form B_ki - B_cur.

        A hard code changes only when the target entry's logit overtakes
        the incumbent's, so the object to maximize is the margin, not
        either logit. `grad` is that margin's first-order term and decays
        once the step leaves the linear regime; this is its exact
        maximizer over a unit ball. Each B_k is rank 2, so the
        difference is rank <= 4 and lives in span(W_ki, V_ki, W_cur,
        V_cur) -- a 4x4 eigenproblem per sample, not d x d."""
        de, U, P = layer_input(module, X, li)
        W, V = de.classifier.weight_ubind()
        Wh, Vh = W[hi], V[hi]
        cur = jnp.argmax(P[:, hi, :], -1)
        B = jnp.stack([jnp.broadcast_to(Wh[ki], U.shape),
                       jnp.broadcast_to(Vh[ki], U.shape),
                       Wh[cur], Vh[cur]], axis=1)          # (b, 4, d)
        Q, _ = jnp.linalg.qr(jnp.swapaxes(B, 1, 2))        # (b, d, 4)

        def form(w, v, sgn):
            qw = jnp.einsum("bda,bd->ba", Q, w)
            qv = jnp.einsum("bda,bd->ba", Q, v)
            return sgn * 0.5 * (qw[:, :, None] * qv[:, None, :]
                                + qv[:, :, None] * qw[:, None, :])

        M = (form(jnp.broadcast_to(Wh[ki], U.shape),
                  jnp.broadcast_to(Vh[ki], U.shape), 1.0)
             + form(Wh[cur], Vh[cur], -1.0))               # (b, 4, 4)
        vals, vecs = jnp.linalg.eigh(M)
        top = vecs[:, :, -1]                               # largest eigenvalue
        D = jnp.einsum("bda,ba->bd", Q, top)
        # sign the ball direction by the first-order margin change
        def lin(w, v):
            return (jnp.einsum("bd,bd->b", D, v) * jnp.einsum("bd,bd->b", U, w)
                    + jnp.einsum("bd,bd->b", D, w) * jnp.einsum("bd,bd->b", U, v))
        first = (lin(jnp.broadcast_to(Wh[ki], U.shape),
                     jnp.broadcast_to(Vh[ki], U.shape))
                 - lin(Wh[cur], Vh[cur]))
        D = D * jnp.where(first < 0, -1.0, 1.0)[:, None]
        return D[..., :module.d_out]

    def adjoint_dir(module, X: Float[Array, "b d_in"], li: int, hi: int,
                    ki: int) -> Float[Array, "b d_out"]:
        """Input-space direction that would cause head hi to select ki.

        `Ontologizer.intervene` reverses the whole classification delta
        through `BilinearBlock.rev`, which eigendecomposes every head:
        (h, k, d, d) is 10GB at this size and will not run. The delta is
        nonzero only at the target head, so only that head's bilinear
        form is needed, which is (k, d, d)."""
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        for i in range(li):
            de = module.dictencs[i]
            U, G = de.gainshape_in(Ein)
            P = de.dict.cluster(de.classifier(U), T)
            R = R + de.gained(de.dict.combine(de.dict.hfwd(P)), G)
            Ein = module.nextinput(X, R, P.reshape(X.shape[0], -1))
        de = module.dictencs[li]
        U, _ = de.gainshape_in(Ein)
        P = de.dict.cluster(de.classifier(U), T)
        dP = (P.at[:, hi, :].set(0.0).at[:, hi, ki].set(1.0) - P)[:, hi, :]
        W, V = de.classifier.weight_ubind()
        B = jnp.einsum("ke,kf->kef", W[hi], V[hi])
        vals, vecs = jnp.linalg.eigh(0.5 * (B + jnp.swapaxes(B, -1, -2)))
        top = jnp.argmax(jnp.abs(vals), -1)
        vtop = jnp.take_along_axis(vecs, top[:, None, None], -1).squeeze(-1)
        # the package's convention, so this path reflects what
        # `BilinearBlock.rev` now returns (it cannot run at this h: it
        # decomposes every head, a 10GB tensor)
        vtop = orient(vtop, jnp.take_along_axis(vals, top[:, None], -1))
        D = dP @ vtop                                   # (b, d_in)
        return D[..., :module.d_out]                    # drop const/code tail

    def adjoint_oriented_dir(module, X: Float[Array, "b d_in"], li: int,
                             hi: int, ki: int) -> Float[Array, "b d_out"]:
        """The same eigendirection, but signed per sample so the
        first-order change in the target logit is positive. `rev` cannot
        do this -- the sign depends on x.v, a property of the sample --
        so this is the eigendirection's best case."""
        de, U, P = layer_input(module, X, li)
        D = adjoint_dir(module, X, li, hi, ki)
        W, V = de.classifier.weight_ubind()
        Wh, Vh = W[hi], V[hi]
        d = jnp.pad(D, ((0, 0), (0, U.shape[-1] - D.shape[-1])))
        # d/de of (W_ki.u)(V_ki.u) along d, at u
        first = (jnp.einsum("bd,d->b", d, Vh[ki]) * jnp.einsum("bd,d->b", U, Wh[ki])
                 + jnp.einsum("bd,d->b", d, Wh[ki]) * jnp.einsum("bd,d->b", U, Vh[ki]))
        return D * jnp.where(first < 0, -1.0, 1.0)[:, None]

    f_adj = jax.jit(
        lambda p, X, li, hi, ki: model.apply(
            p, X, li, hi, ki, method=adjoint_dir),
        static_argnums=(2, 3, 4))
    f_grad = jax.jit(
        lambda p, X, li, hi, ki: model.apply(
            p, X, li, hi, ki, method=grad_dir),
        static_argnums=(2, 3, 4))
    f_margin = jax.jit(
        lambda p, X, li, hi, ki: model.apply(
            p, X, li, hi, ki, method=margin_dir),
        static_argnums=(2, 3, 4))
    f_adjor = jax.jit(
        lambda p, X, li, hi, ki: model.apply(
            p, X, li, hi, ki, method=adjoint_oriented_dir),
        static_argnums=(2, 3, 4))

    def unit(V: Float[Array, "b d"]) -> Float[Array, "b d"]:
        return V / (jnp.linalg.norm(V, axis=-1, keepdims=True) + 1e-9)

    rows = []
    Xs_all = Xref
    for (li, hi, ki) in feats:
        # steer rows that do NOT already select the entry
        idx = np.where(A_ref[li, :, hi] != ki)[0]
        idx = rng.choice(idx, min(cfg.n_samples, len(idx)), replace=False)
        X = Xs_all[jnp.asarray(idx)]
        A0 = np.asarray(jnp.argmax(f_codes(params, X)[0], -1))   # (l, b, h)
        n = jnp.linalg.norm(X, axis=-1, keepdims=True)

        Yb = f_codes(params, X)[1]
        dirs = {
            "decode": unit(f_forced(params, X, li, hi, ki) - Yb),
            "grad": unit(f_grad(params, X, li, hi, ki)),
            "margin": unit(f_margin(params, X, li, hi, ki)),
            "adjoint": unit(f_adj(params, X, li, hi, ki)),
            "adjoint_or": unit(f_adjor(params, X, li, hi, ki)),
            "random": unit(jax.random.normal(
                jax.random.PRNGKey(int(idx[0])), X.shape)),
        }
        for s in cfg.strengths:
            for kind, D in dirs.items():
                Xi = X + s * D
                Xi = Xi * n / (jnp.linalg.norm(Xi, axis=-1, keepdims=True) + 1e-9)
                A = np.asarray(jnp.argmax(f_codes(params, Xi)[0], -1))
                rows.append(dict(
                    kind=kind, layer=li, head=hi, entry=ki, strength=s,
                    hit=float((A[li, :, hi] == ki).mean()),
                    collateral=float(_collateral(A, A0, li, hi))))

    with open(out / "steer.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'kind':>8} {'strength':>9} {'realized':>9} {'collateral':>11}")
    summary = {}
    for kind in ("decode", "grad", "margin", "adjoint", "adjoint_or", "random"):
        for s in cfg.strengths:
            R = [r for r in rows if r["kind"] == kind and r["strength"] == s]
            hit = float(np.mean([r["hit"] for r in R]))
            col = float(np.mean([r["collateral"] for r in R]))
            summary[f"{kind}@{s}"] = {"hit": hit, "collateral": col}
            print(f"{kind:>8} {s:9.2f} {hit:9.3f} {col:11.3f}")
    print(f"\nbaseline: the entry was selected on 0 of the steered rows by "
          f"construction; chance for a head is 1/k = {1/k:.3f}")
    (out / "summary.json").write_text(json.dumps(
        {"model": str(cfg.model), "step": int(step), "T": T,
         "n_features": len(feats), "n_samples": cfg.n_samples,
         "chance": 1 / k, "results": summary}, indent=2))
    print(f"-> {out}")


def _collateral(A: Int[np.ndarray, "l b h"], A0: Int[np.ndarray, "l b h"],
                li: int, hi: int) -> float:
    """Share of other (layer, head) argmaxes that moved."""
    ch = A != A0
    ch[li, :, hi] = False
    return ch.sum() / (ch.size - ch.shape[1])


if __name__ == "__main__":
    main()

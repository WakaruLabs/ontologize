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

  decode    x + m * unit(Y_forced - Y_base): where the entry DECODES to.
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
  random    an isotropic random unit direction, the control curve.

A unit step in raw coordinates is long or short in the data's own metric
depending on how it lies against the data's covariance, so the isotropic
control is one shape among several. Three more controls take the same raw
length, the first two reshaping its draw z ~ N(0, I), with Sigma the
covariance of the last --cov-rows cache rows:

  cov       Sigma^{1/2} z: shaped like the data
  icov      Sigma^{-1/2} z: weighted toward the directions the data barely
            vary along (eigenvalues floored at 1e-6 of the largest)
  diff      another reference row minus this one: a step toward a real row

Each is scored as a direction is, so `hit` is the rate at which the target
entry is selected by accident and `collateral` the share of the other
l*h - 1 heads that moved (the share of all l*h heads differs from it by at
most 1/(l*h)). `decode` is also split by where it lies in the data's
spread:

  on<p>     its projection on the top principal subspace of those rows,
            the fewest components holding p% of their variance (--var)
  off<p>    the remainder

geometry.csv places every direction against the data, per target, as the
median over the steered rows of each unit direction v's

  var_share    v' Sigma v over Sigma's mean eigenvalue: 1 for an isotropic
               direction on average
  whitened     its length in the objective's metric (--mse-weights)
  mahalanobis  its length in the data's own metric, |Sigma^{-1/2} v|
  top<p>       the share of |v|^2 inside the top-p% subspace

both lengths relative to the median distance between two tail rows in
that metric and scaled to raw units, so a direction shaped like the data
reads about 1 in either, and an isotropic one reads more in the
Mahalanobis metric the more anisotropic the data are.

  uv run python experiments/ste-arm/steerembed.py \\
      --model data/out/sonar/multilingual/ste_h76 --temperature 0.00015

An `sae.py` checkpoint (a `params.npz`) is scored the same way, with a
latent in place of a (layer, head, entry). What "selected" and
"collateral" mean depends on the code's form:

  heads     (--groups in sae.py) the latent is selected when it wins its
            head, and collateral is the share of OTHER heads whose
            winner changed. A top1 head whose winner is not positive
            abstains, which counts as its own outcome.
  no heads  (top-k, L1) the latent is selected when it is active, and
            collateral is the share of the row's other active latents
            that dropped out -- each of the k slots standing in for a
            head's choice.

and its directions are:

  decode    W_dec[j]: where the latent decodes to, the standard SAE
            steering vector, and the Ontologizer's `decode` with a
            latent for an entry.
  grad      the gradient of latent j's pre-activation: W_enc[:, j]
            (W_gate for a gated SAE), per sample for a bilinear one.
  eig       a bilinear latent's top eigenfeature (`sae.eigenfeatures`).
  random    as above.

Both modes also score the two standard SUPERVISED steering vectors for
the same target (`supervised_dirs`), fitted on the reference rows with
the steered rows held out and the code's own selection as the label:

  dm        difference of means, selected minus not: activation addition
  probe     a class-balanced logistic-regression probe's weight vector

so supervised, SAE and Ontologizer steering are scored on one target, one
success measure and one budget.

Every direction above is a unit vector swept over fixed strengths and
rescaled to |x|, which compares directions at equal budget but never at
the magnitude a user's intervention would actually apply. `native` adds
that operating point: x + (Y_forced - Y_base) for the Ontologizer, and
x + a_j W_dec[j] for an SAE, a_j being latent j's mean nonzero
activation on the reference rows -- the decoded change of clamping the
feature on. No rescale, and `native_random` is the control: a random
direction of the same per-row length. Its implied strength, |delta|/|x|,
is reported alongside. `native_dm` does the same for the
difference-of-means vector at its own length -- the step activation
addition takes -- with `native_dm_random` as its control.

  uv run python experiments/ste-arm/steerembed.py \\
      --model data/out/sonar/sae_conv/m11264_k32/params.npz
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float, Int

import sae
from ontologize.layers.nlinear import orient
from ontologize.ontologizer import Ontologizer
from pareto import load_onto, parse_sae_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Ontologizer checkpoint dir, or an sae.py params.npz")
    p.add_argument("--temperature", type=float, default=0.00015,
                   help="Ontologizer only")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--n-features", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=32,
                   help="held-out rows steered per feature")
    # a strength is a step of that fraction of the row's own norm, so it
    # means the same on unit-norm SONAR embeddings and on GPT-2 activations
    # of norm ~120
    p.add_argument("--strengths", type=float, nargs="+",
                   default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--ref-rows", type=int, default=8192,
                   help="rows for the baseline firing rates")
    p.add_argument("--ref-b", type=int, default=512,
                   help="rows per forward in the baseline pass")
    p.add_argument("--min-rate", type=float, default=0.005,
                   help="skip entries the model almost never selects")
    p.add_argument("--cov-rows", type=int, default=32768,
                   help="cache-tail rows whose covariance shapes the cov, "
                        "icov and diff controls, the on/off split and the "
                        "geometry")
    p.add_argument("--var", type=float, nargs="+", default=[0.5, 0.8],
                   help="variance fractions whose principal subspaces split "
                        "the decode direction into on<p> and off<p>")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy",
                   help="the objective's inverse-variance weights, for the "
                        "whitened length in geometry.csv")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    if cfg.model.endswith(".npz"):
        main_sae(cfg)
    else:
        main_onto(cfg)


def main_onto(cfg: argparse.Namespace) -> None:
    model, raw, step = load_onto(cfg.model, cfg.step)
    params = {"params": raw}
    l, h, k = model.l, model.h, model.k
    T = cfg.temperature
    out = Path(cfg.out or f"data/out/sonar/steerembed/"
                          f"{Path(cfg.model).name}_{step}")
    out.mkdir(parents=True, exist_ok=True)

    mm = np.load(cfg.cache, mmap_mode="r")
    Xref = jnp.asarray(np.asarray(mm[-cfg.ref_rows:], np.float32))
    metric, ranks = load_metric(cfg)
    Sh, Sih, Q = (jnp.asarray(a, jnp.float32)
                  for a in (metric.Sh, metric.Sih, metric.Q))
    pair_rng = np.random.default_rng([cfg.seed, 2])

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
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
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
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
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
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
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

    rows, geo = [], []
    Xs_all = Xref
    for (li, hi, ki) in feats:
        # steer rows that do NOT already select the entry
        idx = np.where(A_ref[li, :, hi] != ki)[0]
        idx = rng.choice(idx, min(cfg.n_samples, len(idx)), replace=False)
        X = Xs_all[jnp.asarray(idx)]
        A0 = np.asarray(jnp.argmax(f_codes(params, X)[0], -1))   # (l, b, h)
        n = jnp.linalg.norm(X, axis=-1, keepdims=True)

        Yb = f_codes(params, X)[1]
        delta = f_forced(params, X, li, hi, ki) - Yb
        z = jax.random.normal(jax.random.PRNGKey(int(idx[0])), X.shape)
        rnd = unit(z)
        rows += _native(delta, rnd, X,
                        lambda Xi: np.asarray(jnp.argmax(
                            f_codes(params, Xi)[0], -1)),
                        lambda A: (A[li, :, hi] == ki),
                        lambda A: _collateral(A, A0, li, hi),
                        dict(layer=li, head=hi, entry=ki))
        dm, probe = supervised_dirs(Xref, A_ref[li, :, hi] == ki, idx)
        rows += _native(jnp.broadcast_to(dm, X.shape), rnd, X,
                        lambda Xi: np.asarray(jnp.argmax(
                            f_codes(params, Xi)[0], -1)),
                        lambda A: (A[li, :, hi] == ki),
                        lambda A: _collateral(A, A0, li, hi),
                        dict(layer=li, head=hi, entry=ki), name="native_dm")
        dirs = {
            "decode": unit(delta),
            "grad": unit(f_grad(params, X, li, hi, ki)),
            "margin": unit(f_margin(params, X, li, hi, ki)),
            "adjoint": unit(f_adj(params, X, li, hi, ki)),
            "adjoint_or": unit(f_adjor(params, X, li, hi, ki)),
            "dm": unit(jnp.broadcast_to(dm, X.shape)),
            "probe": jnp.broadcast_to(probe, X.shape),
            **controls(z, X, Xs_all[jnp.asarray(
                partners(idx, Xs_all.shape[0], pair_rng))], Sh, Sih),
            **split_dirs(delta, Q, ranks),
        }
        geo += geometry_rows(dirs, metric, ranks,
                             dict(layer=li, head=hi, entry=ki))
        for s in cfg.strengths:
            for kind, D in dirs.items():
                Xi = X + s * n * D
                Xi = Xi * n / (jnp.linalg.norm(Xi, axis=-1, keepdims=True) + 1e-9)
                A = np.asarray(jnp.argmax(f_codes(params, Xi)[0], -1))
                rows.append(dict(
                    kind=kind, layer=li, head=hi, entry=ki, strength=s,
                    hit=float((A[li, :, hi] == ki).mean()),
                    collateral=float(_collateral(A, A0, li, hi))))

    _write_csv(out / "steer.csv", rows)
    _write_csv(out / "geometry.csv", geo)
    summary = _swept_summary(rows, cfg.strengths)
    summary.update(_native_summary(rows))
    geometry = _geometry_summary(geo)
    print(f"\nbaseline: the entry was selected on 0 of the steered rows by "
          f"construction; chance for a head is 1/k = {1/k:.3f}")
    (out / "summary.json").write_text(json.dumps(
        {"model": str(cfg.model), "step": int(step), "T": T,
         "n_features": len(feats), "n_samples": cfg.n_samples,
         "chance": 1 / k, "results": summary,
         **_metric_summary(cfg, metric, ranks), "geometry": geometry},
        indent=2))
    print(f"-> {out}")


def main_sae(cfg: argparse.Namespace) -> None:
    path = Path(cfg.model)
    meta_path = path.parent / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        m, topk = meta["m"], meta["topk"]
        groups, group_fn = meta["groups"], meta["group_fn"]
    else:  # pre-meta.json run: recover the encode rule from the name
        m, topk = parse_sae_name(path)
        groups, group_fn = 0, "top1"
    params = {k_: jnp.asarray(v) for k_, v in np.load(path).items()}
    gs = m // groups if groups else 0
    name = path.parent.name
    out = Path(cfg.out or f"data/out/sonar/steerembed/sae_{name}")
    out.mkdir(parents=True, exist_ok=True)

    mm = np.load(cfg.cache, mmap_mode="r")
    Xref = jnp.asarray(np.asarray(mm[-cfg.ref_rows:], np.float32))
    metric, ranks = load_metric(cfg)
    Sh, Sih, Q = (jnp.asarray(a, jnp.float32)
                  for a in (metric.Sh, metric.Sih, metric.Q))
    pair_rng = np.random.default_rng([cfg.seed, 2])

    def select(X: Float[Array, "b d"]) -> Int[Array, "b c"]:
        """The code's discrete outcome: per-head winner (-1 when a top1
        head abstains), or each latent's active flag."""
        z = sae.encode(params, X, topk, groups, group_fn)
        if not groups:
            return (z > 0).astype(jnp.int32)
        g = z.reshape(X.shape[0], groups, gs)
        win = jnp.argmax(g, -1)
        if group_fn != "softmax":
            win = jnp.where(g.max(-1) > 0, win, -1)
        return win

    def pre(x: Float[Array, "d"]) -> Float[Array, "m"]:
        """The logits that decide firing: the gate for a gated SAE."""
        if "W_gate" in params:
            return sae.gated_pre(params, x[None])[0][0]
        return sae.preacts(params, x[None])[0]

    def act_sums(X: Float[Array, "b d"]
                 ) -> Tuple[Float[Array, "m"], Float[Array, "m"]]:
        """Per-latent sum and count of nonzero activations."""
        z = sae.encode(params, X, topk, groups, group_fn)
        return z.sum(0), (z > 0).sum(0).astype(jnp.float32)

    def grad_dir(X: Float[Array, "b d"], j: Int[Array, ""],
                 cur: Int[Array, "b"]) -> Float[Array, "b d"]:
        """Ascent on latent j's logit, and for a code with heads descent on
        the incumbent's, as the Ontologizer's `grad` does.
        `cur` < 0 means no incumbent."""
        def one(x, c):
            f = lambda x_: pre(x_)[j] - jnp.where(
                c >= 0, pre(x_)[jnp.maximum(c, 0)], 0.0)
            return jax.grad(f)(x)
        return jax.vmap(one)(X, cur)

    f_select = jax.jit(select)
    f_grad = jax.jit(grad_dir)
    S_ref = np.concatenate([np.asarray(f_select(Xref[i:i + cfg.ref_b]))
                            for i in range(0, Xref.shape[0], cfg.ref_b)])
    f_sums = jax.jit(act_sums)
    tot, cnt = np.zeros(m), np.zeros(m)
    for i in range(0, Xref.shape[0], cfg.ref_b):
        s_, c_ = f_sums(Xref[i:i + cfg.ref_b])
        tot += np.asarray(s_)
        cnt += np.asarray(c_)
    # the latent's typical value when it fires: what clamping it "on"
    # writes, so `native` steers by a_j * W_dec[j]
    a_on = tot / np.maximum(cnt, 1)

    def hits(S: Int[np.ndarray, "b c"], j: int) -> Bool[np.ndarray, "b"]:
        return S[:, j // gs] == j % gs if groups else S[:, j] == 1

    rate = np.array([hits(S_ref, j).mean() for j in range(m)])
    n_ref = S_ref.shape[0]
    live = np.where((rate >= cfg.min_rate)
                    & ((1 - rate) * n_ref >= cfg.n_samples))[0]
    rng = np.random.default_rng(cfg.seed)
    if not len(live):
        raise SystemExit(
            f"sae {name}: no latent is selected on between {cfg.min_rate:.1%} "
            f"of rows and all but {cfg.n_samples} of them, so there is "
            f"nothing to steer (rates: {(rate == 0).sum()} never, "
            f"{(rate == 1).sum()} always)")
    feats = rng.choice(live, min(cfg.n_features, len(live)), replace=False)
    eig = sae.eigenfeatures(params) if "W_enc1" in params else None
    W_dec = params["W_dec"]
    form = (f"{groups} heads of {gs}, {group_fn}" if groups
            else f"top-{topk}" if topk else "unconstrained")
    print(f"sae {name}: m={m}, {form}")
    print(f"{len(live)} of {m} latents are selected on >= {cfg.min_rate:.1%} "
          f"of rows; steering {len(feats)} of them, {cfg.n_samples} rows "
          f"each\n")

    rows, geo = [], []
    for j in feats:
        j = int(j)
        idx = np.where(~hits(S_ref, j))[0]
        idx = rng.choice(idx, min(cfg.n_samples, len(idx)), replace=False)
        X = Xref[jnp.asarray(idx)]
        S0 = np.asarray(f_select(X))
        n = jnp.linalg.norm(X, axis=-1, keepdims=True)
        if groups:
            win = S0[:, j // gs]
            cur = np.where(win >= 0, (j // gs) * gs + win, -1)
        else:
            cur = np.full(X.shape[0], -1)
        z = jax.random.normal(jax.random.PRNGKey(int(idx[0])), X.shape)
        rnd = unit(z)
        rows += _native(jnp.broadcast_to(a_on[j] * W_dec[j], X.shape), rnd, X,
                        lambda Xi: np.asarray(f_select(Xi)),
                        lambda S: hits(S, j),
                        lambda S: _collateral_sae(S, S0, j, gs),
                        dict(latent=j, base_rate=float(rate[j])))
        dm, probe = supervised_dirs(Xref, hits(S_ref, j), idx)
        rows += _native(jnp.broadcast_to(dm, X.shape), rnd, X,
                        lambda Xi: np.asarray(f_select(Xi)),
                        lambda S: hits(S, j),
                        lambda S: _collateral_sae(S, S0, j, gs),
                        dict(latent=j, base_rate=float(rate[j])),
                        name="native_dm")
        delta = jnp.broadcast_to(W_dec[j], X.shape)
        dirs = {
            "decode": unit(delta),
            "grad": unit(f_grad(X, jnp.asarray(j), jnp.asarray(cur))),
        }
        if eig is not None:
            dirs["eig"] = unit(jnp.broadcast_to(eig[j], X.shape))
        dirs.update({
            "dm": unit(jnp.broadcast_to(dm, X.shape)),
            "probe": jnp.broadcast_to(probe, X.shape),
            **controls(z, X, Xref[jnp.asarray(
                partners(idx, Xref.shape[0], pair_rng))], Sh, Sih),
            **split_dirs(delta, Q, ranks),
        })
        geo += geometry_rows(dirs, metric, ranks,
                             dict(latent=j, base_rate=float(rate[j])))
        for s in cfg.strengths:
            for kind, D in dirs.items():
                Xi = X + s * n * D
                Xi = Xi * n / (jnp.linalg.norm(Xi, axis=-1, keepdims=True) + 1e-9)
                S = np.asarray(f_select(Xi))
                rows.append(dict(
                    kind=kind, latent=j, base_rate=float(rate[j]), strength=s,
                    hit=float(hits(S, j).mean()),
                    collateral=float(_collateral_sae(S, S0, j, gs))))

    _write_csv(out / "steer.csv", rows)
    _write_csv(out / "geometry.csv", geo)
    summary = _swept_summary(rows, cfg.strengths)
    summary.update(_native_summary(rows))
    geometry = _geometry_summary(geo)
    base = float(rate[feats].mean())
    print(f"\nbaseline: the latent was selected on 0 of the steered rows by "
          f"construction; its mean selection rate elsewhere is {base:.3f}")
    (out / "summary.json").write_text(json.dumps(
        {"model": str(cfg.model), "m": m, "topk": topk, "groups": groups,
         "group_fn": group_fn, "n_features": len(feats),
         "n_samples": cfg.n_samples, "chance": base, "results": summary,
         **_metric_summary(cfg, metric, ranks), "geometry": geometry},
        indent=2))
    print(f"-> {out}")


def supervised_dirs(Xref: Float[Array, "n d"], y: Bool[np.ndarray, "n"],
                    exclude: Int[np.ndarray, "e"], steps: int = 300,
                    l2: float = 1e-3
                    ) -> Tuple[Float[Array, "d"], Float[Array, "d"]]:
    """The two standard supervised steering vectors for "make the target
    selected", fitted on the reference rows with `exclude` (the rows about
    to be steered) held out, the model's own selection as the label:

      dm     difference of means, mean(selected) - mean(not), at its own
             length (the vector activation addition adds)
      probe  the weight vector of a class-balanced logistic-regression
             probe for "selected", unit length

    The labels are the model's, so this asks whether a supervised method
    given the model's partition steers into it better than the model's own
    directions do -- the same target, metric and budget for both."""
    keep = np.ones(len(y), bool)
    keep[exclude] = False
    X = Xref[jnp.asarray(np.where(keep)[0])]
    t = jnp.asarray(y[keep], jnp.float32)
    dm = X[t > 0].mean(0) - X[t == 0].mean(0)

    # centre and rescale so one learning rate serves SONAR (|x| = 1) and
    # GPT-2 (|x| ~ 120); neither changes the direction of w
    Xc = X - X.mean(0)
    Xc = Xc / jnp.linalg.norm(Xc, axis=-1).mean()
    pos = jnp.maximum(t.mean(), 1e-6)
    wt = jnp.where(t > 0, 0.5 / pos, 0.5 / (1 - pos))

    def loss(p):
        z = Xc @ p["w"] + p["b"]
        nll = wt * (jax.nn.softplus(z) - t * z)
        return nll.mean() + l2 * (p["w"] ** 2).sum()

    tx = optax.adam(5e-2)
    p = {"w": jnp.zeros(X.shape[1]), "b": jnp.zeros(())}
    st = tx.init(p)

    @jax.jit
    def step(p, st):
        g = jax.grad(loss)(p)
        u, st = tx.update(g, st, p)
        return optax.apply_updates(p, u), st

    for _ in range(steps):
        p, st = step(p, st)
    w = p["w"]
    return dm, w / (jnp.linalg.norm(w) + 1e-9)


def unit(V: Float[Array, "b d"]) -> Float[Array, "b d"]:
    return V / (jnp.linalg.norm(V, axis=-1, keepdims=True) + 1e-9)


class DataMetric(NamedTuple):
    """The cache tail's covariance in the forms the data-shaped controls and
    geometry.csv use, and the median distance between two of its rows in
    each metric the geometry reports."""
    lam: Float[np.ndarray, "d"]       # eigenvalues, descending
    Q: Float[np.ndarray, "d d"]       # their eigenvectors, as columns
    Sh: Float[np.ndarray, "d d"]      # Sigma^{1/2}
    Sih: Float[np.ndarray, "d d"]     # Sigma^{-1/2}, eigenvalues floored
    w: Float[np.ndarray, "d"]         # the objective's inverse-variance weights
    d_raw: float
    d_w: float
    d_m: float


def data_metric(Xt: Float[np.ndarray, "n d"], w: Float[np.ndarray, "d"],
                rng: np.random.Generator, floor: float = 1e-6,
                n_pairs: int = 4096) -> DataMetric:
    """Sigma = cov(Xt) and its square roots. `floor`, a fraction of the
    largest eigenvalue, bounds Sigma^{-1/2} along directions the rows do
    not vary along; the distances are medians over `n_pairs` random pairs
    of rows."""
    Xt = np.asarray(Xt, np.float64)
    w = np.asarray(w, np.float64)
    lam, Q = np.linalg.eigh(np.cov(Xt, rowvar=False))
    lam = np.maximum(lam[::-1], 0.0)
    Q = np.ascontiguousarray(Q[:, ::-1])
    Sh = (Q * np.sqrt(lam)) @ Q.T
    Sih = (Q / np.sqrt(np.maximum(lam, lam[0] * floor))) @ Q.T
    pair = rng.choice(len(Xt), (n_pairs, 2))
    D = Xt[pair[:, 0]] - Xt[pair[:, 1]]
    return DataMetric(
        lam=lam, Q=Q, Sh=Sh, Sih=Sih, w=w,
        d_raw=float(np.median(np.linalg.norm(D, axis=1))),
        d_w=float(np.median(np.sqrt((D ** 2 * w).sum(1)))),
        d_m=float(np.median(np.linalg.norm(D @ Sih, axis=1))))


def principal_ranks(lam: Float[np.ndarray, "d"], fracs: Sequence[float]
                    ) -> Dict[float, int]:
    """For each fraction p, the fewest leading components (lam descending)
    that hold p of the variance."""
    cum = np.cumsum(lam) / lam.sum()
    return {p: min(int(np.searchsorted(cum, p)) + 1, len(lam)) for p in fracs}


def pct(p: float) -> str:
    """A variance fraction as the percent in a kind's name: 0.5 -> "50"."""
    return f"{100 * p:g}"


def load_metric(cfg: argparse.Namespace
                ) -> Tuple[DataMetric, Dict[float, int]]:
    """The metric of the last --cov-rows cache rows, and the principal
    ranks --var asks for."""
    mm = np.load(cfg.cache, mmap_mode="r")
    w = np.load(cfg.mse_weights)
    if w.shape != mm.shape[1:]:
        raise SystemExit(f"--mse-weights {cfg.mse_weights} has shape "
                         f"{w.shape}, the cache's rows {mm.shape[1:]}")
    metric = data_metric(np.asarray(mm[-cfg.cov_rows:], np.float64), w,
                         np.random.default_rng([cfg.seed, 1]))
    return metric, principal_ranks(metric.lam, cfg.var)


def partners(idx: Int[np.ndarray, "b"], n: int, rng: np.random.Generator
             ) -> Int[np.ndarray, "b"]:
    """For each row in idx, a uniformly random other row of the n."""
    return (idx + 1 + rng.integers(0, n - 1, len(idx))) % n


def controls(z: Float[Array, "b d"], X: Float[Array, "b d"],
             X_other: Float[Array, "b d"], Sh: Float[Array, "d d"],
             Sih: Float[Array, "d d"]) -> Dict[str, Float[Array, "b d"]]:
    """The untargeted unit steps: one draw z ~ N(0, I) taken isotropic
    (`random`), shaped like the data (`cov`) and against it (`icov`), and
    the step toward another row (`diff`). Sh and Sih are symmetric, so
    z @ Sh is Sigma^{1/2} z row by row."""
    return {"random": unit(z), "cov": unit(z @ Sh), "icov": unit(z @ Sih),
            "diff": unit(X_other - X)}


def split_dirs(delta: Float[Array, "b d"], Q: Float[Array, "d d"],
               ranks: Dict[float, int]) -> Dict[str, Float[Array, "b d"]]:
    """delta's projection on each leading principal subspace (`on<p>`) and
    what is left of it (`off<p>`), each at unit length."""
    out = {}
    for p, r in ranks.items():
        on = (delta @ Q[:, :r]) @ Q[:, :r].T
        out[f"on{pct(p)}"] = unit(on)
        out[f"off{pct(p)}"] = unit(delta - on)
    return out


GEOMETRY = ("var_share", "whitened", "mahalanobis")


def step_geometry(V: Float[np.ndarray, "b d"], metric: DataMetric,
                  ranks: Dict[float, int]
                  ) -> Dict[str, Float[np.ndarray, "b"]]:
    """Each row of V as a unit direction v: v' Sigma v over Sigma's mean
    eigenvalue; v's whitened and Mahalanobis lengths, each relative to the
    median distance between two rows in that metric and scaled to raw
    units; and the share of |v|^2 in each leading principal subspace."""
    V = np.asarray(V, np.float64)
    V = V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-12)
    C = V @ metric.Q
    out = {
        "var_share": (C ** 2 * metric.lam).sum(-1) / metric.lam.mean(),
        "whitened": (np.sqrt((V ** 2 * metric.w).sum(-1))
                     / metric.d_w * metric.d_raw),
        "mahalanobis": (np.linalg.norm(V @ metric.Sih, axis=-1)
                        / metric.d_m * metric.d_raw)}
    for p, r in ranks.items():
        out[f"top{pct(p)}"] = (C[:, :r] ** 2).sum(-1)
    return out


def geometry_rows(dirs: Dict[str, Float[Array, "b d"]], metric: DataMetric,
                  ranks: Dict[float, int], ident: Dict[str, Any]
                  ) -> List[Dict[str, Any]]:
    """One geometry.csv row per direction kind, each statistic's median
    over the steered rows."""
    return [dict(kind=kind, **ident,
                 **{s: float(np.median(v)) for s, v in step_geometry(
                     np.asarray(D), metric, ranks).items()})
            for kind, D in dirs.items()]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _swept_summary(rows: List[Dict[str, Any]], strengths: Sequence[float]
                   ) -> Dict[str, Dict[str, float]]:
    """Print and return realization and collateral per swept kind and
    strength, the kinds in the order they were scored."""
    kinds = list(dict.fromkeys(r["kind"] for r in rows
                               if not r["kind"].startswith("native")))
    print(f"{'kind':>10} {'strength':>9} {'realized':>9} {'collateral':>11}")
    out = {}
    for kind in kinds:
        for s in strengths:
            R = [r for r in rows if r["kind"] == kind and r["strength"] == s]
            hit = float(np.mean([r["hit"] for r in R]))
            col = float(np.mean([r["collateral"] for r in R]))
            out[f"{kind}@{s}"] = {"hit": hit, "collateral": col}
            print(f"{kind:>10} {s:9.2f} {hit:9.3f} {col:11.3f}")
    return out


def _geometry_summary(geo: List[Dict[str, Any]]
                      ) -> Dict[str, Dict[str, float]]:
    """Print and return each kind's geometry, the median over targets."""
    stats = [s for s in geo[0] if s in GEOMETRY or s.startswith("top")]
    print(f"\n{'kind':>10} " + " ".join(f"{s:>11}" for s in stats)
          + "   (median over targets of the median over rows)")
    out = {}
    for kind in dict.fromkeys(r["kind"] for r in geo):
        R = [r for r in geo if r["kind"] == kind]
        out[kind] = {s: float(np.median([r[s] for r in R])) for s in stats}
        print(f"{kind:>10} "
              + " ".join(f"{out[kind][s]:11.3f}" for s in stats))
    return out


def _metric_summary(cfg: argparse.Namespace, metric: DataMetric,
                    ranks: Dict[float, int]) -> Dict[str, Any]:
    """What the controls and the geometry were measured against."""
    return {"cov_rows": cfg.cov_rows, "mse_weights": cfg.mse_weights,
            "ranks": {pct(p): r for p, r in ranks.items()},
            "median_pair_distance": {"raw": metric.d_raw,
                                     "whitened": metric.d_w,
                                     "mahalanobis": metric.d_m}}


def _native(delta: Float[Array, "b d"], rnd: Float[Array, "b d"],
            X: Float[Array, "b d"], read: Callable[[Any], np.ndarray],
            hit: Callable[[np.ndarray], np.ndarray],
            collateral: Callable[[np.ndarray], float],
            ident: Dict[str, Any], name: str = "native"
            ) -> List[Dict[str, Any]]:
    """Steering at the method's own magnitude: x + delta with no rescale to
    |x| -- for the model, what the intervention changes in the output; for
    `name="native_dm"`, the difference-of-means vector at its own length.
    Its control (`<name>_random`) is a random direction of the same per-row
    length. `strength` is the mean implied |delta| / |x|, on the fixed
    sweep's scale."""
    mag = jnp.linalg.norm(delta, axis=-1, keepdims=True)
    s = float((mag / jnp.linalg.norm(X, axis=-1, keepdims=True)).mean())
    out = []
    for kind, Xi in ((name, X + delta), (f"{name}_random", X + mag * rnd)):
        A = read(Xi)
        out.append(dict(kind=kind, **ident, strength=s,
                        hit=float(hit(A).mean()),
                        collateral=float(collateral(A))))
    return out


def _native_summary(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Print and return the native rows: realization and collateral, and
    the spread of the implied strength across features."""
    out = {}
    for kind in ("native", "native_random", "native_dm", "native_dm_random"):
        R = [r for r in rows if r["kind"] == kind]
        s = np.array([r["strength"] for r in R])
        hit = float(np.mean([r["hit"] for r in R]))
        col = float(np.mean([r["collateral"] for r in R]))
        q = np.quantile(s, [0.1, 0.5, 0.9])
        out[kind] = {"hit": hit, "collateral": col,
                     "strength_p10_p50_p90": [float(v) for v in q]}
        print(f"{kind:>13}  strength p50 {q[1]:.3f} (p10 {q[0]:.3f}, "
              f"p90 {q[2]:.3f})  realized {hit:.3f}  collateral {col:.3f}")
    return out


def _collateral_sae(S: Int[np.ndarray, "b c"], S0: Int[np.ndarray, "b c"],
                    j: int, gs: int) -> float:
    """With heads (gs > 0, entries per head): share of other heads whose
    winner changed. Without: share of each row's other active latents that
    dropped out, averaged over rows that had any."""
    if gs:
        ch = S != S0
        ch[:, j // gs] = False
        return ch.sum() / (ch.size - ch.shape[0])
    was = S0.astype(bool)
    was[:, j] = False
    lost = (was & ~S.astype(bool)).sum(1)
    k0 = was.sum(1)
    ok = k0 > 0
    return float((lost[ok] / k0[ok]).mean()) if ok.any() else 0.0


def _collateral(A: Int[np.ndarray, "l b h"], A0: Int[np.ndarray, "l b h"],
                li: int, hi: int) -> float:
    """Share of other (layer, head) argmaxes that moved."""
    ch = A != A0
    ch[li, :, hi] = False
    return ch.sum() / (ch.size - ch.shape[1])


if __name__ == "__main__":
    main()

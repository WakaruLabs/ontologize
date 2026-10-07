"""Per-feature bilinear structure of the Ontologizer classifier.

With the live classifier settings (n=2, gate "none", activation_cl
"none", biased_cl off) every (layer, head, entry) logit is one bilinear
pair over the layer's classifier input u,

    K_hk(u) = (w_hk . u)(v_hk . u) = u' B_hk u,   B = (w v' + v w') / 2,

where w, v are `classifier.weight[0, h, k]` and `[1, h, k]`. B has rank
at most 2 with eigenvalues

    lambda_+- = (w.v +- |w||v|) / 2,

so every spectral statistic of a feature is a function of the signed
cosine c = cos(w, v):
  |c| -> 1   B ~ lambda (u.x)^2: the logit is even in the direction u,
             blind to the sign of x along it (its superlevel sets are
             pairs of antipodal half-spaces, or a slab when lambda < 0)
  c = 0      lambda_- = -lambda_+: a saddle, (w.x)(v.x) with orthogonal
             factors, which does see the sign of each projection
  top        max|lambda| / sum|lambda| = (1 + |c|) / 2
  neg        negative share of |lambda| mass = (1 - c) / 2; above 1/2
             the dominant eigenvalue is negative
  erank      participation ratio (sum|lambda|)^2 / sum lambda^2
             = 2 / (1 + c^2), in [1, 2]
These are computed on the SEMANTIC block of u: the classifier input
minus its pass-through tail. Layer 0 reads the encoder output (the
cache row itself unless `encoded`); layers >= 1 read the
reconstruction residual (forward="resid"), unit-normalized by
resid_norm or by gain-shape (resid_gain). With resid_const the input
ends in a constant 1 at index d_sem (layer 0 only when const0), and a
feature becomes (w.u + a)(v.u + b): the pair is still rank 1 over the
d_sem + 1 coordinates, but over the semantic block it adds the odd term
(b w + a v).u and a constant ab. The constant coordinate is what can
break sign-blindness, so it is scored separately:

  odd        the odd part's share of the logit's variance for u
             uniform on the unit sphere of the semantic block,
             E[(l.u)^2] / (E[(l.u)^2] + Var(u'Bu)), l = b w + a v.
             0 = even in u (sign-blind), 1 = linear in u.
             Isotropic u is a weights-only reference measure: real
             inputs are anisotropic, so this is not the share on data.
             An isotropic projection w.u is only ~|w|/sqrt(d), so in
             high d a small constant already dominates this share.
  rho_w      |a| / |w_s| (rho_v likewise): for unit u at cosine t with
             w_s the factor is |w_s| (t + a/|w_s|), so the constant
             dominates it for |t| < rho_w and the projection for
             |t| > rho_w. Data-free; which regime real inputs sit in is
             a question about their alignment with w_s.

Eigenvectors live in the classifier-input space (the residual basis for
layers >= 1); nothing here is projected back to text.

Classifiers with n != 2 have no quadratic form and are skipped with a
message. A gate (`gate` != "none") or `activation_cl` is applied around
the product; the statistics then describe the pre-activation form, and
the script says so.

  uv run python bilinspec.py --ckpt data/out/sonar/multilingual/resid_nc_hm
  uv run python bilinspec.py --ckpt data/out/gpt2_l8/ste_h76 \\
      --steps 10000 100000 371900

Steps: --steps, else --n-steps saved steps spaced geometrically between
the first and last (0 = every saved step). The last chosen step is the
"final" one drawn in the per-feature figures. Writes to <out>:
  bilin_ecdf.png    per layer, the ECDF of |cos(w,v)| over all features
                    (black) and per head (faint), and of the odd share
  bilin_steps.png   median and IQR per layer against step
  bilin_heads.png   layer x head median |cos| and median odd share
  bilinspec.npz     per step: signed cos (semantic and full input), odd
                    share and |w||v| for every (layer, head, entry)
  bilinspec.csv     per (step, layer) summary
--replot redraws from bilinspec.npz without the checkpoint. NOTE:
Ontologizer checkpoints restore on GPU JAX only.
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from jaxtyping import Float, Int

LAYER_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                 "#008300", "#4a3aa7", "#e34948")
SHARP = 0.9  # |cos| above which a feature is reported as ~ lambda (u.x)^2


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="Ontologizer checkpoint dir")
    p.add_argument("--steps", type=int, nargs="+", default=None,
                   help="checkpoint steps (default: --n-steps of them)")
    p.add_argument("--n-steps", type=int, default=8,
                   help="geometrically spaced saved steps; 0 = all")
    p.add_argument("--check", type=int, default=4,
                   help="features per layer whose closed-form spectrum is "
                   "checked against a dense eigvalsh at the final step")
    p.add_argument("--replot", action="store_true",
                   help="redraw from <out>/bilinspec.npz without the model")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="default <ckpt>/bilinspec")
    return p.parse_args()


# ---------- spectra ----------

def pair_cos(W: Float[np.ndarray, "... d"], V: Float[np.ndarray, "... d"]
             ) -> Float[np.ndarray, "..."]:
    """Signed cos(w, v) per feature (0 for a zero factor)."""
    nw, nv = np.linalg.norm(W, axis=-1), np.linalg.norm(V, axis=-1)
    den = nw * nv
    return np.where(den > 0, (W * V).sum(-1) / np.where(den > 0, den, 1.0),
                    0.0)


def pair_eigs(W: Float[np.ndarray, "... d"], V: Float[np.ndarray, "... d"]
              ) -> Float[np.ndarray, "... 2"]:
    """The two nonzero eigenvalues (lambda_+, lambda_-) of (w v' + v w')/2:
    (w.v +- |w||v|) / 2. The other d - 2 are zero."""
    dot = (W * V).sum(-1)
    nn = np.linalg.norm(W, axis=-1) * np.linalg.norm(V, axis=-1)
    return np.stack([(dot + nn) / 2, (dot - nn) / 2], -1)


def spectrum_stats(lam: Float[np.ndarray, "... m"]
                   ) -> Dict[str, Float[np.ndarray, "..."]]:
    """From eigenvalues (any count; zeros contribute nothing): top =
    max|lambda| / sum|lambda|, neg = negative share of sum|lambda|, erank =
    (sum|lambda|)^2 / sum lambda^2. NaN for an all-zero spectrum."""
    a = np.abs(lam)
    s = a.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"top": a.max(-1) / s,
                "neg": np.where(lam < 0, a, 0.0).sum(-1) / s,
                "erank": s ** 2 / (lam ** 2).sum(-1)}


def odd_share(W: Float[np.ndarray, "... d"], V: Float[np.ndarray, "... d"],
              a: Float[np.ndarray, "..."], b: Float[np.ndarray, "..."]
              ) -> Float[np.ndarray, "..."]:
    """Odd share of the variance of (w.u + a)(v.u + b) for u uniform on
    the unit sphere in R^d. With B = (w v' + v w')/2 and l = b w + a v,
    E[(l.u)^2] = |l|^2 / d and Var(u'Bu) = (tr(B)^2 + 2 tr(B^2)) /
    (d (d + 2)) - tr(B)^2 / d^2; the cross term integrates to zero. NaN
    when both parts vanish."""
    d = W.shape[-1]
    trB = (W * V).sum(-1)
    trB2 = (trB ** 2 + (W * W).sum(-1) * (V * V).sum(-1)) / 2
    var_even = (trB ** 2 + 2 * trB2) / (d * (d + 2)) - trB ** 2 / d ** 2
    L = b[..., None] * W + a[..., None] * V
    odd = (L * L).sum(-1) / d
    with np.errstate(invalid="ignore", divide="ignore"):
        return odd / (odd + var_even)


def layer_layout(spec: dict, i: int, width: int) -> Tuple[int, bool]:
    """(d_sem, has_const) for layer i's classifier input of `width`: the
    semantic block is the first d_sem coordinates, and with has_const
    the constant 1 sits at index d_sem. The tail is [const | code], as
    `Ontologizer.setup` lays it out."""
    fwd = spec.get("forward", "labels")
    const = bool(spec.get("resid_const")) and fwd != "labels"
    if i == 0:
        const = const and bool(spec.get("const0", True))
        tail = int(const)
    elif fwd == "resid_labels":
        tail = int(const) + spec["k"] * spec["h"]
    else:
        tail = int(const)
    return width - tail, const


def feature_stats(Wc: Float[np.ndarray, "2 h k D"], d_sem: int,
                  has_const: bool) -> Dict[str, Float[np.ndarray, "h k"]]:
    """Per (head, entry) of one layer's bilinear classifier weight."""
    W, V = Wc[0].astype(np.float64), Wc[1].astype(np.float64)
    Ws, Vs = W[..., :d_sem], V[..., :d_sem]
    if has_const:
        a, b = W[..., d_sem], V[..., d_sem]
    else:
        a = b = np.zeros(W.shape[:-1])
    nw, nv = np.linalg.norm(Ws, axis=-1), np.linalg.norm(Vs, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"cos": pair_cos(Ws, Vs),
                "cos_full": pair_cos(W, V),
                "odd": odd_share(Ws, Vs, a, b),
                "rho_w": np.abs(a) / nw, "rho_v": np.abs(b) / nv,
                "scale": nw * nv}


def pick_steps(steps: Sequence[int], n: int) -> List[int]:
    """n of the sorted saved steps, spaced geometrically in step between
    the first and last (each target snaps to the nearest saved step, so
    fewer than n may come back); all of them when n <= 0 or n >= len."""
    s = np.sort(np.asarray(steps))
    if n <= 0 or n >= len(s):
        return s.tolist()
    lo = max(int(s[0]), 1)
    tgt = np.geomspace(lo, max(int(s[-1]), lo), n)
    idx = np.abs(s[None, :] - tgt[:, None]).argmin(1)
    return sorted({int(s[j]) for j in idx} | {int(s[-1])})


def _nanmed(x: np.ndarray) -> float:
    return float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan")


def summarize(st: Dict[str, Float[np.ndarray, "h k"]]) -> Dict[str, float]:
    """One layer's summary row from its `feature_stats`."""
    c = np.abs(st["cos"]).ravel()
    q25, med, q75 = np.quantile(c, [0.25, 0.5, 0.75])
    return {"n": c.size, "abscos_med": med, "abscos_q25": q25,
            "abscos_q75": q75, "frac_sharp": float((c > SHARP).mean()),
            "frac_negdom": float((st["cos"] < 0).mean()),
            "odd_med": _nanmed(st["odd"]),
            "rho_w_med": _nanmed(st["rho_w"]),
            "rho_v_med": _nanmed(st["rho_v"]),
            "abscos_full_med": float(np.median(np.abs(st["cos_full"])))}


# ---------- checkpoint ----------

def load_classifiers(ckpt, steps: Optional[Sequence[int]], n_steps: int):
    """(spec, chosen steps, {step: [layer weight (n, h, k, D)]})."""
    import jax
    import orbax.checkpoint as ocp
    from ontologize.training.serialize import restore_spec

    manager = ocp.CheckpointManager(
        Path(ckpt).resolve(),
        checkpointers={"state": ocp.PyTreeCheckpointer(),
                       "spec": ocp.PyTreeCheckpointer()})
    chosen = list(steps) if steps else pick_steps(manager.all_steps(), n_steps)
    spec = restore_spec(manager, chosen[-1])
    out = {}
    for st in chosen:
        state = manager.restore(st, items={"state": None})["state"]
        params = state["params"] if "opt_state" in state else state
        while "params" in params:
            params = params["params"]
        out[st] = [np.asarray(jax.device_get(
            params[f"dictencs_{i}"]["classifier"]["weight"]), np.float32)
            for i in range(spec["l"])]
        del state, params
        print(f"step {st}: restored")
    return spec, chosen, out


def check_dense(Ws: Sequence[Float[np.ndarray, "2 h k D"]],
                layout: Sequence[Tuple[int, bool]], n: int, seed: int):
    """Closed-form spectrum vs a dense eigvalsh of the symmetrized pair,
    on n random features per layer; returns the largest relative error
    over eigenvalues and the top/neg statistics."""
    rng = np.random.default_rng(seed)
    err = 0.0
    for Wc, (d_sem, _) in zip(Ws, layout):
        _, h, k, _ = Wc.shape
        for f in rng.choice(h * k, min(n, h * k), replace=False):
            w = Wc[0].reshape(h * k, -1)[f, :d_sem].astype(np.float64)
            v = Wc[1].reshape(h * k, -1)[f, :d_sem].astype(np.float64)
            B = (np.outer(w, v) + np.outer(v, w)) / 2
            lam = np.linalg.eigvalsh(B)
            ref = np.sort(pair_eigs(w, v))
            ext = np.array([lam[0], lam[-1]])
            sc = np.abs(lam).max()
            err = max(err, np.abs(ext - ref).max() / sc,
                      np.abs(np.delete(lam, [0, len(lam) - 1])).max() / sc)
            a, b = spectrum_stats(lam), spectrum_stats(pair_eigs(w, v))
            err = max(err, abs(a["top"] - b["top"]), abs(a["neg"] - b["neg"]))
    return err


# ---------- figures ----------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def ecdf(x: Float[np.ndarray, "n"]):
    x = np.sort(x[np.isfinite(x)])
    return x, np.arange(1, len(x) + 1) / max(len(x), 1)


def draw_ecdf(cos: Float[np.ndarray, "l h k"], odd: Float[np.ndarray, "l h k"],
              has_const: Sequence[bool], title: str, path):
    """Row 1: |cos(w,v)| ECDF per layer, all features black, each head
    faint. Row 2: the odd share, likewise (blank where the layer has no
    constant coordinate, where it is 0 by construction)."""
    plt = _plt()
    l = cos.shape[0]
    fig, axes = plt.subplots(2, l, figsize=(2.6 * l + 0.6, 5.4),
                             sharex=True, sharey=True, squeeze=False)
    for L in range(l):
        for r, (X, name) in enumerate(((np.abs(cos[L]), "|cos(w, v)|"),
                                       (odd[L], "odd share"))):
            ax = axes[r, L]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25, lw=0.5)
            if r == 1 and not has_const[L]:
                ax.text(0.5, 0.5, "no constant\ncoordinate:\nodd share = 0",
                        ha="center", va="center", fontsize=8, color="0.4")
                ax.set_xlabel(name)
                continue
            for hd in range(X.shape[0]):
                ax.step(*ecdf(X[hd]), where="post", color=LAYER_COLOURS[
                    L % len(LAYER_COLOURS)], alpha=0.18, lw=0.6)
            ax.step(*ecdf(X.ravel()), where="post", color="black", lw=1.6)
            med = np.nanmedian(X)
            ax.axvline(med, color="black", lw=0.6, ls=":")
            if r == 0:
                neg = (cos[L] < 0).mean()
                sharp = (np.abs(cos[L]) > SHARP).mean()
                ax.set_title(f"layer {L}\nmed {med:.2f}, >{SHARP}: "
                             f"{sharp:.0%}, neg-dom {neg:.0%}", fontsize=8)
            else:
                ax.set_title(f"med {med:.2f}", fontsize=8)
            ax.set_xlabel(name)
        axes[0, 0].set_ylabel("ECDF over features")
        axes[1, 0].set_ylabel("ECDF over features")
    fig.suptitle(title + "   (black: all features; faint: one line per head; "
                 "neg-dom: dominant eigenvalue < 0)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_steps(steps: Sequence[int], cos: Float[np.ndarray, "s l h k"],
               odd: Float[np.ndarray, "s l h k"], has_const: Sequence[bool],
               title: str, path, d_sem: int = 0):
    """Median (line) and IQR (band) per layer against step, with the
    random-pair |cos| level for semantic width d_sem (0 = omit)."""
    plt = _plt()
    s, l = cos.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    x = np.asarray(steps)
    for ax, (X, name) in zip(axes, ((np.abs(cos), "|cos(w, v)|"),
                                    (odd, "odd share"))):
        for L in range(l):
            if name == "odd share" and not has_const[L]:
                continue
            q = np.nanquantile(X[:, L].reshape(s, -1), [0.25, 0.5, 0.75], 1)
            c = LAYER_COLOURS[L % len(LAYER_COLOURS)]
            ax.fill_between(x, q[0], q[2], color=c, alpha=0.18, lw=0)
            ax.plot(x, q[1], color=c, lw=2, marker="o", ms=4,
                    label=f"layer {L}")
        if name == "|cos(w, v)|" and d_sem:
            # E|cos| of two independent isotropic vectors in R^d
            ax.axhline(np.sqrt(2 / (np.pi * d_sem)), color="0.5", lw=0.8,
                       ls="--", label="random pair")
        ax.set_ylim(0, 1)
        if len(x) > 1 and x.min() > 0 and x.max() / x.min() > 20:
            ax.set_xscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel(f"{name}: median, IQR band")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_heads(cos: Float[np.ndarray, "l h k"], odd: Float[np.ndarray, "l h k"],
               has_const: Sequence[bool], title: str, path):
    """Layer x head medians, heads in index order, fixed [0, 1] scale.
    Odd-share rows of layers without a constant coordinate are grey."""
    plt = _plt()
    l, h, _ = cos.shape
    fig, axes = plt.subplots(2, 1, figsize=(min(3 + h / 6, 22), 2.2 + 0.7 * l),
                             squeeze=False)
    cmap = plt.get_cmap("Blues").with_extremes(bad="0.85")
    for r, (ax, (X, name)) in enumerate(zip(
            axes[:, 0], ((np.abs(cos), "median |cos(w, v)|"),
                         (odd, "median odd share")))):
        M = np.nanmedian(X, -1)
        if r == 1:
            M = np.where(np.asarray(has_const)[:, None], M, np.nan)
            for L in np.flatnonzero(~np.asarray(has_const)):
                ax.text(h / 2, L, "no constant coordinate", ha="center",
                        va="center", fontsize=7, color="0.3")
        im = ax.imshow(M, aspect="auto", interpolation="nearest",
                       cmap=cmap, vmin=0, vmax=1)
        ax.set_yticks(range(l), [f"L{L}" for L in range(l)], fontsize=8)
        step = max(1, h // 38)
        ax.set_xticks(range(0, h, step), [str(i) for i in range(0, h, step)],
                      fontsize=6)
        ax.set_xlabel("head")
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.set_label(name, fontsize=8)
    axes[0, 0].set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_all(z: dict, out: Path):
    steps = [int(s) for s in z["steps"]]
    has_const = [bool(c) for c in z["has_const"]]
    run = str(z["run"])
    cos, odd = z["cos"], z["odd"]
    draw_ecdf(cos[-1], odd[-1], has_const, f"{run} step {steps[-1]}",
              out / "bilin_ecdf.png")
    draw_heads(cos[-1], odd[-1], has_const, f"{run} step {steps[-1]}",
               out / "bilin_heads.png")
    if len(steps) > 1:
        draw_steps(steps, cos, odd, has_const, run, out / "bilin_steps.png",
                   int(np.max(z["d_sem"])))
    for name in ("bilin_ecdf.png", "bilin_heads.png", "bilin_steps.png"):
        if (out / name).exists():
            print(f"-> {out / name}")


def main():
    cfg = parse_args()
    if cfg.replot:
        assert cfg.out or cfg.ckpt, "--replot needs --out or --ckpt"
        out =Path(cfg.out) if cfg.out else Path(cfg.ckpt) / "bilinspec"
        draw_all(dict(np.load(out / "bilinspec.npz")), out)
        return
    assert cfg.ckpt, "--ckpt is required unless --replot with --out"
    out = Path(cfg.out) if cfg.out else Path(cfg.ckpt) / "bilinspec"

    spec, steps, Ws = load_classifiers(cfg.ckpt, cfg.steps, cfg.n_steps)
    n = spec.get("n", 2)
    if n != 2:
        print(f"classifier has n={n}: a degree-{n} form, not a quadratic "
              "one, so it has no symmetric interaction matrix; nothing to do")
        return
    if spec.get("gate", "none") != "none" or \
            spec.get("activation_cl", "none") != "none":
        print(f"NOTE: gate={spec.get('gate')!r}, activation_cl="
              f"{spec.get('activation_cl')!r}: the logit is not the bilinear "
              "form itself; these statistics describe the ungated product")
    layout = [layer_layout(spec, i, W.shape[-1])
              for i, W in enumerate(Ws[steps[-1]])]
    print(f"{spec['l']} layers x {spec['h']} heads x {spec['k']} entries; "
          "classifier input (d_sem, const): " +
          ", ".join(f"L{i} ({d}, {c})" for i, (d, c) in enumerate(layout)))
    if cfg.check:
        print(f"closed form vs dense eigvalsh, max rel err "
              f"{check_dense(Ws[steps[-1]], layout, cfg.check, cfg.seed):.2e}")

    per = [[feature_stats(W, *lay) for W, lay in zip(Ws[st], layout)]
           for st in steps]
    arr = {key: np.stack([np.stack([f[key] for f in fs]) for fs in per])
           for key in per[0][0]}  # each (s, l, h, k)
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"step": st, "layer": L, **summarize(per[j][L])}
            for j, st in enumerate(steps) for L in range(spec["l"])]
    with open(out / "bilinspec.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows[-spec["l"]:]:
        print(f"step {r['step']} layer {r['layer']}: |cos| median "
              f"{r['abscos_med']:.3f} [IQR {r['abscos_q25']:.3f}-"
              f"{r['abscos_q75']:.3f}], >{SHARP} {r['frac_sharp']:.0%}, "
              f"neg-dominant {r['frac_negdom']:.0%}, odd share median "
              f"{r['odd_med']:.3f}, rho_w/rho_v median {r['rho_w_med']:.3f}/"
              f"{r['rho_v_med']:.3f}, |cos| over full input "
              f"{r['abscos_full_med']:.3f}")
    z = dict(arr, steps=np.asarray(steps), run=Path(cfg.ckpt).name,
             has_const=np.asarray([c for _, c in layout]),
             d_sem=np.asarray([d for d, _ in layout]))
    np.savez(out / "bilinspec.npz", **z)
    print(f"-> {out / 'bilinspec.csv'}")
    draw_all(z, out)


if __name__ == "__main__":
    main()

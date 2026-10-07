"""DictEnc anatomy: one (layer, head) of a trained Ontologizer as a block
figure, every block computed by the code path that defines it.

For n held-out rows (the tail --eval-rows of the cache, sae.py's holdout;
a seeded draw without replacement, so no language or document is chosen),
the figure draws, with ONE row ordering shared by the two n-row blocks:

  P_h       (n, k)  `DictBlock.cluster(classifier(u), t)` for head h: the
                    layer's assignment probabilities on the plain forward
                    pass (as `autointerp.onto_probe` runs it). Rows sorted
                    by argmax entry, entries by mean probability (most used
                    first), ties within an entry by the winning
                    probability, descending. A bar strip above gives each
                    entry's mean p_bar, the usage `hmean_kl` and
                    `support_overlap` read.
  G_h       (n, n)  the per-head PWAK kernel `diffuse_kl` builds,
                        G_h = wak(P_h P_h^T * affinity(E, tau)),
                    with E = `DictEnc.pwak_in(u)` (the shaped classifier
                    input, constant coordinates dropped), affinity the heat
                    kernel exp((<e_i,e_j> - 1)/tau) on unit-normalized rows
                    with its diagonal zeroed, and wak the row
                    normalization (rows sum to 1; an all-zero row stays
                    zero). Row i is the weights sample i's PWAK target
                    averages its neighbors with at s=1. tau is the run's
                    `pwak_tau` (0.2 when the run predates the field). Its
                    scale is clipped at the 99th percentile of the nonzero
                    weights (values above saturate).
  cos_k     (k, k)  the matrix `DictBlock.rowcos` reduces: cosines between
                    head h's rows of `dicts()` (|W|, row-normalized under
                    `norm_rows`, signed under `signed`). Its off-diagonal
                    mean is head h's `cossim_k`, which the title prints.
                    Entries in the same usage order as P_h's columns.
  overlap   (h, h)  the matrix `DictBlock.support_overlap` reduces: cosines
                    between the heads' usage-weighted coordinate profiles
                    s_i = sum_j p_bar_ij |W_ij|, with p_bar the mean over
                    the n rows. Head h's row is outlined; the off-diagonal
                    mean, the `support` stat, is in the title. Under
                    `concat` (ConcatDictBlock) the heads use disjoint
                    slices and the statistic is 0 by construction, so this
                    panel says so instead.
  W_h       (k, c)  head h's `dicts()` rows restricted to the c coordinates
                    with the largest s_h (its own coordinate profile),
                    sorted by s_h. A column subset, not a projection.

Every block has its own sequential scale with white at 0 and its own
colorbar; the cosine blocks switch to a symmetric diverging scale when a
`signed` dictionary makes them negative. Nothing is clustered, so a head
that has collapsed reads as one dark column of P_h and one dense block of
G_h.

The PWAK kernel is computed whatever the run's `s_pwak`/`s_L2pwak`: for a
run trained without them it is the graph those terms would have built,
not one the training saw.

  uv run python anatomy.py --ckpt data/out/sonar/multilingual/resid_nc \\
      --layer 0 --head 0

The temperature defaults to the run's schedule at the restored step (its
log.jsonl env_config), as in assignmap.py. Writes <out>/<name>.pdf and a
PNG preview, plus <name>.npz with every block's inputs so the figure can be
redrawn without the model (--replot). NOTE: Ontologizer checkpoints
restore on GPU JAX only. With the default --out (writeup/figures) the PNG
goes to its preview/, as make_figs.py lays it out, and the npz (the
layer's whole dictionary, tens of MB) to <ckpt>/anatomy/, so --replot
then takes --ckpt to find it; any other --out holds all three.
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from jaxtyping import Float, Int

from assignmap import eff_entries, row_eff

# house palette (writeup/figures/make_figs.py, whose module body runs
# every figure on import, so it is restated rather than imported)
PINE = "#0B5132"
INK = "#232329"
PINE50 = "#86A791"
INKMUT = "#7B7B80"
FIGDIR = Path(__file__).resolve().parent / "writeup" / "figures"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="Ontologizer checkpoint dir")
    p.add_argument("--step", type=int, default=0, help="default: latest")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--head", type=int, default=None,
                   help="default 0, or with --replot the stored head")
    p.add_argument("--temperature", type=float, default=None,
                   help="default: the run's schedule at --step")
    p.add_argument("--tau", type=float, default=None,
                   help="affinity width; default the run's pwak_tau (0.2)")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="held-out tail rows of the cache (sae.py's default)")
    p.add_argument("--n", type=int, default=384, help="rows drawn")
    p.add_argument("--coords", type=int, default=48,
                   help="dictionary coordinates shown in the W_h block")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="fig-anatomy")
    p.add_argument("--out", default=str(FIGDIR))
    p.add_argument("--replot", action="store_true",
                   help="redraw from <out>/<name>.npz without the model")
    return p.parse_args()


# ---------- the code-path quantities ----------

def pwak_kernel(P: Float[np.ndarray, "n h k"], E: Float[np.ndarray, "n d"],
                tau: float, head: int) -> Float[np.ndarray, "n n"]:
    """Head `head`'s slice of the per-head graph `pwak.diffuse_kl` builds,
    `wak(P_h P_h^T * affinity(E, tau))`, through the same functions."""
    import jax.numpy as jnp
    from ontologize.fns.pwak import affinity, wak
    Ph = jnp.asarray(P[:, head], jnp.float32)
    D = affinity(jnp.asarray(E, jnp.float32), tau)
    return np.asarray(wak(jnp.einsum("ic,jc->ij", Ph, Ph) * D))


def unit_rows(V: Float[np.ndarray, "... m d"]) -> Float[np.ndarray, "... m d"]:
    """Rows scaled to unit L2 norm; a zero row stays zero (`recip_norm`)."""
    sq = (V * V).sum(-1, keepdims=True)
    return V * np.where(sq > 0, 1.0 / np.sqrt(np.where(sq > 0, sq, 1.0)), 0.0)


def rowcos_matrix(W: Float[np.ndarray, "h k d"]) -> Float[np.ndarray, "h k k"]:
    """Per head, the cosine matrix `DictBlock.rowcos` reduces (`W` is
    `dicts()`). Its off-diagonal mean is that head's `cossim_k`."""
    U = unit_rows(W.astype(np.float64))
    return np.einsum("hkd,hjd->hkj", U, U)


def offdiag_mean(C: Float[np.ndarray, "... m m"]) -> Float[np.ndarray, "..."]:
    """Mean over the off-diagonal entries of the trailing square axes."""
    m = C.shape[-1]
    return (C.sum((-2, -1)) - np.trace(C, axis1=-2, axis2=-1)) / (m * (m - 1))


def profiles(P: Float[np.ndarray, "n h k"],
             W: Float[np.ndarray, "h k d"]) -> Float[np.ndarray, "h d"]:
    """The heads' usage-weighted coordinate profiles s_i = sum_j p_bar_ij
    |W_ij|, with p_bar the mean over the rows (`support_overlap`)."""
    return np.einsum("hk,hkd->hd", P.mean(0), np.abs(W))


def support_matrix(P: Float[np.ndarray, "n h k"],
                   W: Float[np.ndarray, "h k d"]) -> Float[np.ndarray, "h h"]:
    """The cosine matrix `DictBlock.support_overlap` reduces; its
    off-diagonal mean is the `support` stat."""
    S = unit_rows(profiles(P, W).astype(np.float64))
    return S @ S.T


def anatomy_order(Ph: Float[np.ndarray, "n k"]
                  ) -> Tuple[Int[np.ndarray, "n"], Int[np.ndarray, "k"]]:
    """(row order, entry order). Entries by mean probability, most used
    first (stable); rows by their argmax entry's position in that order,
    then by the winning probability, descending."""
    eorder = np.argsort(-Ph.mean(0), kind="stable")
    rank = np.argsort(eorder)
    win = Ph.argmax(-1)
    rorder = np.lexsort((-Ph.max(-1), rank[win]))
    return rorder, eorder


def top_coords(s: Float[np.ndarray, "d"], c: int) -> Int[np.ndarray, "c"]:
    """The c coordinates with the largest profile value, largest first."""
    return np.argsort(-s, kind="stable")[:c]


def clip_top(A: np.ndarray, q: float = 99.0) -> float:
    """The q-th percentile of A's nonzero entries (1.0 when there are
    none), for a sequential scale that one heavy row cannot wash out."""
    nz = A[A > 0]
    return float(np.percentile(nz, q)) if nz.size else 1.0


# ---------- model ----------

def anatomy_probe(module, X, temperature, layer):
    """Layer `layer`'s assignments (b, h, k), its PWAK input
    `pwak_in(u)` and its `dicts()`, on the plain forward pass as
    `autointerp.onto_probe` runs it. Run under `model.apply`."""
    E, _ = module.encode(X, 0.0, None)
    R = module.resid(E)
    E_in = module.constinput(E)
    for i, de in enumerate(module.dictencs):
        U, G = de.gainshape_in(E_in)
        P = de.dict.cluster(de.classifier(U), temperature)
        if i == layer:
            return P, de.pwak_in(U), de.dict.dicts()
        R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
        E_in = module.nextinput(X, R, P.reshape(P.shape[0], -1))
    raise ValueError(f"layer {layer} >= l={module.l}")


def compute(cfg) -> dict:
    """Run the model on the drawn rows and return every block's inputs."""
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from assignmap import read_hyper, temperature_at
    from ontologize.ontologizer import Ontologizer
    from ontologize.training.serialize import restore_spec

    manager = ocp.CheckpointManager(
        Path(cfg.ckpt).resolve(),
        checkpointers={'state': ocp.PyTreeCheckpointer(),
                       'spec': ocp.PyTreeCheckpointer()})
    step = cfg.step or manager.latest_step()
    hyper = read_hyper(cfg.ckpt)
    T = cfg.temperature
    if T is None:
        assert hyper is not None, \
            "no env_config in log.jsonl; pass --temperature"
        T = temperature_at(hyper, step)
    tau = cfg.tau if cfg.tau is not None else \
        float((hyper or {}).get("pwak_tau", 0.2))
    model = Ontologizer(**restore_spec(manager, step))
    state = manager.restore(step, items={'state': None})['state']
    params = state['params'] if 'opt_state' in state else state
    while 'params' in params:
        params = params['params']
    params = {'params': params}

    mm = np.load(cfg.cache, mmap_mode="r")
    lo = mm.shape[0] - cfg.eval_rows
    rng = np.random.default_rng(cfg.seed)
    rows = lo + np.sort(rng.choice(cfg.eval_rows, cfg.n, replace=False))
    X = jnp.asarray(np.asarray(mm[rows], dtype=np.float32))

    fn = jax.jit(lambda p, x: model.apply(p, x, T, cfg.layer,
                                          method=anatomy_probe))
    P, E, W = (np.asarray(a, np.float32) for a in fn(params, X))
    print(f"step {step}: l={model.l} h={model.h} k={model.k}, "
          f"T={T:g}, tau={tau:g}, n={cfg.n}")
    return dict(P=P, E=E, W=W, rows=rows, step=step, temperature=T,
                tau=tau, concat=bool(getattr(model, "concat", False)),
                signed=bool(getattr(model, "signed", False)),
                run=Path(cfg.ckpt).name, layer=cfg.layer, head=cfg.head)


# ---------- figure ----------

def seq_cmap(color, name):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, ["#FFFFFF", color])


def div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("inkpine",
                                             [INK, "#FFFFFF", PINE])


def draw(z: dict, coords: int, pdf: Path, png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({
        "font.family": "serif", "font.size": 7.5, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.6,
        "figure.facecolor": "none", "savefig.facecolor": "none",
        "pdf.fonttype": 42})

    P, E, W = z["P"], z["E"], z["W"]
    hd, tau = int(z["head"]), float(z["tau"])
    n, h, k = P.shape
    Ph = P[:, hd]
    rorder, eorder = anatomy_order(Ph)
    G = pwak_kernel(P, E, tau, hd)[np.ix_(rorder, rorder)]
    Pp = Ph[np.ix_(rorder, eorder)]
    pbar = Pp.mean(0)
    Ck = rowcos_matrix(W)[hd][np.ix_(eorder, eorder)]
    cosk = float(offdiag_mean(rowcos_matrix(W))[hd])
    concat = bool(z["concat"])
    S = None if concat else support_matrix(P, W)
    cols = top_coords(profiles(P, W)[hd], coords)
    Wc = W[hd][np.ix_(eorder, cols)]

    use, reff = eff_entries(Ph[:, None])[0], row_eff(Ph[:, None])[0]

    fig = plt.figure(figsize=(5.4, 6.1))
    # top: [p_bar strip over P_h | cb | G_h | cb]
    # bottom: [cos_k | cb | overlap | cb | W_h | cb]
    top = fig.add_gridspec(2, 5, left=0.10, right=0.93, top=0.92,
                           bottom=0.42, width_ratios=[1.0, 0.05, 0.30,
                                                      2.3, 0.05],
                           height_ratios=[0.13, 1.0], hspace=0.05,
                           wspace=0.05)
    bot = fig.add_gridspec(1, 8, left=0.10, right=0.93, top=0.30,
                           bottom=0.07, width_ratios=[1, 0.06, 0.55, 1,
                                                      0.06, 0.6, 1.4,
                                                      0.06], wspace=0.08)

    def cbar(im, cax, label=None, extend="neither"):
        cb = fig.colorbar(im, cax=cax, extend=extend)
        cb.ax.tick_params(labelsize=5.5, length=2, width=0.4)
        cb.outline.set_linewidth(0.4)
        for lb in cb.ax.get_yticklabels():
            lb.set_fontfamily("monospace")
        if label:
            cb.set_label(label, fontsize=6, labelpad=2)
        return cb

    def mono(ax):
        for lb in ax.get_xticklabels() + ax.get_yticklabels():
            lb.set_fontfamily("monospace")

    lettered = []

    def letter(ax, ch, row):
        """Queue a bold panel letter so captions can name panels by letter;
        `place_letters` sets it once the layout is final."""
        lettered.append((ax, ch, row))

    def place_letters():
        """Each letter left of its axes, level with the top of the tallest
        title in its row: the square blocks shrink to their aspect, so the
        axes' own tops do not line up."""
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        tops, lefts = {}, {}
        for ax, _, row in lettered:
            t = inv.transform(ax.title.get_window_extent(r))
            pos = ax.get_position()
            tops[row] = max(tops.get(row, 0.0), t[1, 1], pos.y1)
            # a title wider than its axes starts left of them
            lefts[ax] = min(pos.x0, t[0, 0]) if ax.get_title() else pos.x0
        for ax, ch, row in lettered:
            fig.text(lefts[ax] - 0.01, tops[row], ch, fontsize=9,
                     fontweight="bold", color=INK, ha="right", va="top")

    # p_bar strip
    ax_u = fig.add_subplot(top[0, 0])
    ax_u.bar(np.arange(k), pbar, width=1.0, color=INKMUT)
    ax_u.set_xlim(-0.5, k - 0.5)
    ax_u.set_ylim(0, max(pbar.max() * 1.1, 1e-9))
    ax_u.set_yticks([])
    ax_u.set_xticks([])
    for s in ("top", "right", "left"):
        ax_u.spines[s].set_visible(False)
    ax_u.set_ylabel(r"$\bar p$", rotation=0, labelpad=8, va="center")
    ax_u.set_title(r"assignments $P_i$ at temperature $t$"
                   f"\nuse {use:.1f}, row {reff:.1f} of k={k}",
                   fontsize=7, pad=3)
    letter(ax_u, "a", 0)

    # P_h
    ax_p = fig.add_subplot(top[1, 0])
    im = ax_p.imshow(Pp, aspect="auto", interpolation="nearest",
                     cmap=seq_cmap(PINE, "pine"), vmin=0, vmax=1)
    ax_p.set_xlabel("entry (by usage)")
    ax_p.set_ylabel(f"held-out row (by argmax entry), n={n}")
    ax_p.set_xticks([0, k - 1])
    ax_p.set_yticks([0, n - 1])
    mono(ax_p)
    cbar(im, fig.add_subplot(top[1, 1]), "probability")
    # cell boundaries: where the argmax entry changes
    win = Pp.argmax(-1)
    for b in np.flatnonzero(np.diff(win)) + 0.5:
        ax_p.axhline(b, color=INKMUT, lw=0.2)

    # G_h
    ax_g = fig.add_subplot(top[1, 3])
    vg = clip_top(G)
    im = ax_g.imshow(G, aspect="auto", interpolation="nearest",
                     cmap=seq_cmap(INK, "ink"), vmin=0, vmax=vg)
    ax_g.set_yticks([])
    ax_g.set_xticks([0, n - 1])
    mono(ax_g)
    ax_g.set_xlabel("neighbor row (same order)")
    ax_g.set_title(r"PWAK neighbor graph: $P_iP_i^\top\odot$ heat kernel"
                   f" (bandwidth {tau:g})\nself-edges zeroed, rows sum to 1",
                   fontsize=7, pad=3)
    letter(ax_g, "b", 0)
    cbar(im, fig.add_subplot(top[1, 4]), "row weight (clipped at p99)",
         extend="max")

    # cos_k
    ax_c = fig.add_subplot(bot[0])
    if Ck.min() < 0:
        im = ax_c.imshow(Ck, cmap=div_cmap(), vmin=-1, vmax=1,
                         interpolation="nearest")
    else:
        im = ax_c.imshow(Ck, cmap=seq_cmap(PINE, "pine"), vmin=0, vmax=1,
                         interpolation="nearest")
    ax_c.set_xticks([0, k - 1])
    ax_c.set_yticks([0, k - 1])
    mono(ax_c)
    ax_c.set_xlabel("entry (by usage)")
    ax_c.set_title("atom cosines\n" + rf"$\kappa_i$ = {cosk:.3f}",
                   fontsize=7, pad=3)
    letter(ax_c, "c", 1)
    cbar(im, fig.add_subplot(bot[1]))

    # overlap
    ax_s = fig.add_subplot(bot[3])
    cax_s = fig.add_subplot(bot[4])
    if S is None:
        ax_s.text(0.5, 0.5, r"$\omega$ = 0 by construction" "\n"
                  "(concat: disjoint slices)", ha="center", va="center",
                  transform=ax_s.transAxes, fontsize=6.5)
        ax_s.set_axis_off()
        cax_s.set_axis_off()
    else:
        # omega sits near 1 in trained models, so a 0-based scale renders
        # the block uniformly dark; start it at the off-diagonal p5, below
        # which a few outlier heads are clipped (the colorbar's min arrow)
        off = S[~np.eye(h, dtype=bool)]
        vs = min(np.floor(np.percentile(off, 5) * 100) / 100, 0.99)
        im = ax_s.imshow(S, cmap=seq_cmap(INK, "ink"), vmin=vs, vmax=1,
                         interpolation="nearest")
        ax_s.add_patch(Rectangle((-0.5, hd - 0.5), h, 1, fill=False,
                                 ec="#FFFFFF" if S[hd].mean() > 0.5
                                 else PINE, lw=0.9))
        ax_s.set_xticks([0, h - 1])
        ax_s.set_yticks([hd])
        ax_s.set_yticklabels([f"i={hd}"])
        mono(ax_s)
        ax_s.set_xlabel(r"head $i'$")
        ax_s.set_title(r"profile cosines ($\boldsymbol{\pi}$)" "\n"
                       rf"$\omega$ = {offdiag_mean(S):.3f}",
                       fontsize=7, pad=3)
        cbar(im, cax_s, extend="min" if off.min() < vs else "neither")
    letter(ax_s, "d", 1)

    # W_h column subset
    ax_w = fig.add_subplot(bot[6])
    if Wc.min() < 0:
        vw = float(np.abs(Wc).max())
        im = ax_w.imshow(Wc, aspect="auto", cmap=div_cmap(), vmin=-vw,
                         vmax=vw, interpolation="nearest")
    else:
        im = ax_w.imshow(Wc, aspect="auto", cmap=seq_cmap(PINE, "pine"),
                         vmin=0, vmax=float(Wc.max()),
                         interpolation="nearest")
    ax_w.set_xticks([0, len(cols) - 1])
    ax_w.set_yticks([0, k - 1])
    mono(ax_w)
    ax_w.set_xlabel(f"top {len(cols)} of {W.shape[-1]} coords by "
                    + r"$\boldsymbol{\pi}_i$")
    ax_w.set_title(r"atoms $\mathbf{a}_{ij}$, column subset", fontsize=7,
                   pad=3)
    letter(ax_w, "e", 1)
    cbar(im, fig.add_subplot(bot[7]))

    fig.text(0.10, 0.985, f"{z['run']}  step {int(z['step'])}  layer "
             f"{int(z['layer'])}  head {hd}  t={float(z['temperature']):g}",
             fontsize=6.5, fontfamily="monospace", color=INKMUT, va="top")
    place_letters()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {pdf}\n-> {png}")


def main():
    cfg = parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    infig = out.resolve() == FIGDIR
    if infig:
        # the layer's whole dictionary is tens of MB: keep it with the
        # (untracked) run, not in the tracked figure data
        assert cfg.ckpt, "--ckpt is required to locate the npz"
        npz = Path(cfg.ckpt) / "anatomy" / f"{cfg.name}.npz"
        npz.parent.mkdir(exist_ok=True)
    else:
        npz = out / f"{cfg.name}.npz"
    if cfg.replot:
        # the npz holds the whole layer, so any head can be redrawn
        z = dict(np.load(npz))
        if cfg.head is not None:
            z["head"] = cfg.head
    else:
        assert cfg.ckpt, "--ckpt is required unless --replot"
        cfg.head = cfg.head or 0
        z = compute(cfg)
        np.savez(npz, **z)
    prev = out / "preview" if infig else out
    prev.mkdir(exist_ok=True)
    draw(z, cfg.coords, out / f"{cfg.name}.pdf", prev / f"{cfg.name}.png")


if __name__ == "__main__":
    main()

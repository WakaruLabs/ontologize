"""House-style result figures for the ontologize paper.

Palette and rules follow the flock design system: pine / ink / paper and
their mixes, flat, no gridlines, left+bottom spines, serif text with mono
tick labels. The one exception is LEAF, a lighter green for plotted data:
pine and ink are both near-black, so pine series beside ink ones blur
together. Pine is kept for eyebrows. Outputs PDF (for the paper) and PNG
(preview).
"""

import csv
import json
import os

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import (FixedLocator, FuncFormatter, LogLocator,
                               NullFormatter)

PINE = "#0B5132"
LEAF = "#2A8C5A"     # data green: separable from ink by lightness
INK = "#232329"
PINE50 = "#86A791"   # pine 50% on paper
PINE25 = "#C2D3C7"   # pine 25% on paper
INKMUT = "#7B7B80"   # ink 55%

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIGDIR = os.path.dirname(os.path.abspath(__file__))
PREVIEW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview")
os.makedirs(PREVIEW, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "pdf.fonttype": 42,
})

MONO = {"family": "monospace", "size": 7.0}


def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name + ".pdf"))
    fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=170)
    plt.close(fig)
    print("wrote", name)


def eyebrow(ax, text, y=1.06):
    """The house '//' label, top-left above the axes. In a lettered figure
    it starts after the panel letter (see `letters`)."""
    ax.annotate("// " + text, xy=(0.0, y), xycoords="axes fraction",
                xytext=(0, 0), textcoords="offset points",
                fontfamily="monospace", fontsize=7.0, color=PINE,
                va="bottom")


def letters(axes):
    """Bold panel letters a, b, ... at the start of each panel's eyebrow
    line, pushing the eyebrow right, so captions can name panels by letter.
    Call after the eyebrows are drawn."""
    for ch, ax in zip("abcdefgh", axes):
        ax.annotate(ch, xy=(0.0, 1.06), xycoords="axes fraction",
                    fontsize=9.0, fontweight="bold", color=INK,
                    ha="left", va="bottom")
        for t in ax.texts:
            if t.get_text().startswith("// ") and t.xy == (0.0, 1.06):
                t.xyann = (10, 0)


def logx(ax):
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("monospace")
        lbl.set_fontsize(7.0)


def monoticks(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("monospace")
        lbl.set_fontsize(7.0)


# ---- devinterp: geometry consolidates, code softens ------------------------

rows = list(csv.DictReader(open(os.path.join(RESULTS, "devinterp_steps.csv"))))
steps = [int(r["step"]) for r in rows]
meanz = [float(r["mean_z_cos"]) for r in rows]
medz = [float(r["median_z_cos"]) for r in rows]
hsamp = [float(r["mean_H_sample"]) for r in rows]

fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 2.1))
fig.subplots_adjust(wspace=0.32, bottom=0.21, top=0.86, left=0.09, right=0.99)

a.plot(steps, meanz, color=LEAF, lw=1.3, label="mean")
a.plot(steps, medz, color=PINE50, lw=1.1, label="median")
a.axhline(2, color=INKMUT, lw=0.6, ls=(0, (2, 2)))
a.text(steps[0], 2, " z=2", color=INKMUT, fontsize=6.5, va="bottom",
       family="monospace")
a.set_xlabel("training step")
a.set_ylabel(r"within-head $z_{\cos}$ (refit directions)")
a.legend(loc="upper left", handlelength=1.4)
logx(a)
eyebrow(a, "geometry consolidates")

b.plot(steps, hsamp, color=INK, lw=1.3)
b.axhline(5.0, color=INKMUT, lw=0.6, ls=(0, (2, 2)))
b.text(steps[0], 5.0, " 5-bit max", color=INKMUT, fontsize=6.5, va="bottom",
       family="monospace")
b.set_xlabel("training step")
b.set_ylabel("per-sample entropy (bits)")
b.set_ylim(3.2, 5.25)
logx(b)
eyebrow(b, "code softens")
letters((a, b))
save(fig, "fig-devinterp")

# ---- freeze: transient in one run, persistent in the other -----------------

def load_freeze(name):
    rows = list(csv.DictReader(open(os.path.join(RESULTS, name))))
    bylayer = {}
    for r in rows:
        bylayer.setdefault(int(r["layer"]), []).append(
            (int(r["step"]), int(r["n_frozen"])))
    return {k: sorted(v) for k, v in bylayer.items()}


hm = load_freeze("diag_hm_summary.csv")
nc = load_freeze("diag_nc_summary.csv")
shades = {0: PINE25, 1: LEAF, 2: PINE50, 3: INKMUT, 4: INK}

fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 2.1), sharey=True)
fig.subplots_adjust(wspace=0.10, bottom=0.21, top=0.86, left=0.10, right=0.98)

for layer in sorted(hm):
    pts = hm[layer]
    a.plot([p[0] for p in pts], [p[1] for p in pts], color=shades[layer],
           lw=1.4 if layer == 1 else 1.0, label=f"layer {layer}")
peak = max(hm[1], key=lambda p: p[1])
a.annotate(f"{peak[1]}/32 at {peak[0]//1000}k", xy=peak,
           xytext=(peak[0] * 4, peak[1] - 1),
           fontsize=6.8, family="monospace", color=LEAF,
           arrowprops=dict(arrowstyle="-", color=LEAF, lw=0.6))
a.set_xlabel("training step")
a.set_ylabel("frozen heads (of 32)")
a.set_ylim(-0.8, 31)
a.legend(loc="center right", ncols=1, handlelength=1.2, fontsize=6.6)
logx(a)
eyebrow(a, "resid_nc_hm: transient")

for layer in sorted(nc):
    pts = nc[layer]
    b.plot([p[0] for p in pts], [p[1] for p in pts], color=shades[layer],
           lw=1.4 if layer == 4 else 1.0)
last = nc[4][-1]
b.annotate(f"layer 4: {last[1]}/32 at\nlast checkpoint", xy=last,
           xytext=(1.4e4, 23.0), fontsize=6.8, family="monospace", color=INK,
           arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
b.set_xlabel("training step")
logx(b)
eyebrow(b, "resid_nc: late, persistent")
letters((a, b))
save(fig, "fig-freeze")

# ---- naturalness: arms indistinguishable -----------------------------------

d = json.load(open(os.path.join(RESULTS, "naturalness-summary.json")))
series = {}
for r in d["by_kind_mag"]:
    series.setdefault(r["kind"], []).append((r["mag"], r["chrf"]))
style = {
    "dm": (INK, "-", "difference of means"),
    "onto": (LEAF, "-", "classification forcing"),
    "probe": (PINE50, "-", "linear probe"),
    "random": (INKMUT, (0, (2, 2)), "random direction"),
}
fig, ax = plt.subplots(figsize=(5.4, 2.0))
fig.subplots_adjust(bottom=0.23, top=0.85, left=0.08, right=0.99)
for kind, (c, ls, label) in style.items():
    pts = sorted(series.get(kind, []))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls,
            lw=1.2, marker="o", ms=2.6, label=label)
ax.set_xlabel("intervention magnitude")
ax.set_ylabel("chrF vs unsteered decode")
ax.set_xticks([0.25, 0.5, 1.0])
ax.set_ylim(0, 0.42)
monoticks(ax)
ax.legend(loc="lower left", handlelength=1.6)
eyebrow(ax, "no arm distinguishes itself")
save(fig, "fig-naturalness")

# ---- seed stability --------------------------------------------------------

taus = [0.3, 0.5, 0.7, 0.9]
selfmatch = [0.652, 0.446, 0.296, 0.201]
sae = [0.31, 0.24, 0.15, 0.03]
onto = [0.000, 0.000, 0.000, 0.000]

fig, ax = plt.subplots(figsize=(5.4, 2.0))
fig.subplots_adjust(bottom=0.23, top=0.85, left=0.08, right=0.99)
ax.plot(taus, selfmatch, color=PINE50, ls=(0, (2, 2)), lw=1.2, marker="o",
        ms=2.6, label="same run, 2.6k steps apart (ceiling)")
ax.plot(taus, sae, color=INK, lw=1.2, marker="o", ms=2.6,
        label="SAE k32, seed pair")
ax.plot(taus, onto, color=LEAF, lw=1.4, marker="o", ms=2.8,
        label="Ontologizer, seed pair")
ax.set_xlabel(r"match threshold $\tau$ (activation frame)")
ax.set_ylabel("matched-entry fraction")
ax.set_xticks(taus)
ax.set_ylim(-0.03, 0.72)
monoticks(ax)
ax.legend(loc="upper right", handlelength=1.6)
eyebrow(ax, "partitions do not reproduce across seeds")
save(fig, "fig-seed")

# ---- layer-4 retrain variants: which knob moves the transient --------------

VARIANTS = [
    ("diag_baseline_summary.csv", "control", INK, "-"),
    ("diag_slow_anneal_summary.csv", "anneal 50k→150k", LEAF, "-"),
    ("diag_drop_ramp_summary.csv", "dropout 0.1→0.25", PINE50, (0, (4, 2))),
    ("diag_sdk_floor_summary.csv", "noise floor held", INKMUT, (0, (1, 1.5))),
]
fig, ax = plt.subplots(figsize=(5.4, 2.0))
fig.subplots_adjust(bottom=0.23, top=0.85, left=0.08, right=0.99)
for fname, label, c, ls in VARIANTS:
    rows = list(csv.DictReader(open(os.path.join(RESULTS, fname))))
    pts = sorted((int(r["step"]), int(r["n_frozen"]))
                 for r in rows if r["layer"] == "1")
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls,
            lw=1.2, label=label)
ax.set_xlabel("training step")
ax.set_ylabel("layer-1 frozen heads (of 32)")
ax.set_xlim(9000, 2e5)
ax.set_ylim(-0.8, 25)
ax.legend(loc="upper right", handlelength=1.8)
logx(ax)
eyebrow(ax, "only annealing speed moves the transient")
save(fig, "fig-variants")

# ---- hsic: reconstruction bought by freezing the model ---------------------

HSIC_ARMS = [
    ("diag_baseline_summary.csv", "auxiliary losses (control)", INK, "-"),
    ("diag_hsic_summary.csv", "HSIC replaces them", LEAF, "-"),
    ("diag_both_summary.csv", "HSIC + auxiliary", PINE50, (0, (4, 2))),
]
fig, ax = plt.subplots(figsize=(5.4, 2.0))
fig.subplots_adjust(bottom=0.23, top=0.85, left=0.10, right=0.99)
for fname, label, c, ls in HSIC_ARMS:
    rows = list(csv.DictReader(open(os.path.join(RESULTS, fname))))
    tot = {}
    for r in rows:
        s = int(r["step"])
        tot[s] = tot.get(s, 0) + int(r["n_frozen"])
    pts = sorted(tot.items())
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls,
            lw=1.2, label=label)
ax.set_xlabel("training step")
ax.set_ylabel("frozen heads (of 160)")
ax.set_ylim(-2, 100)
ax.legend(loc="upper left", handlelength=1.8)
logx(ax)
eyebrow(ax, "both hsic arms freeze late and stay frozen")
save(fig, "fig-hsic")

# ---- scaling sweep: coherence vs each parameter axis -----------------------

rows = list(csv.DictReader(open(os.path.join(RESULTS, "scaling_results.csv"))))
byname = {r["name"]: r for r in rows}
AXES = [
    ("heads $h$", ["h16_k32_l5", "h32_k32_l5", "h64_k32_l5"], [16, 32, 64]),
    ("entries $k$", ["h32_k16_l5", "h32_k32_l5", "h32_k64_l5"], [16, 32, 64]),
    ("layers $l$", ["h32_k32_l3", "h32_k32_l5", "h32_k32_l7"], [3, 5, 7]),
]
UNCONVERGED = {"h64_k32_l5"}
fig, axs = plt.subplots(1, 3, figsize=(5.4, 1.9), sharey=True)
fig.subplots_adjust(wspace=0.12, bottom=0.25, top=0.78, left=0.09, right=0.99)
for ax, (label, names, xs) in zip(axs, AXES):
    ys = [float(byname[n]["mean_z_cos"]) for n in names]
    ax.plot(xs, ys, color=LEAF, lw=1.2, zorder=1)
    for n, x, y in zip(names, xs, ys):
        filled = n not in UNCONVERGED
        ax.plot([x], [y], marker="o", ms=4.5, color=LEAF,
                mfc=LEAF if filled else PAPER if False else "#F4F7F2",
                mec=LEAF, zorder=2)
    ax.set_xlabel(label)
    ax.set_xticks(xs)
    monoticks(ax)
axs[0].set_ylabel(r"mean within-head $z_{\cos}$")
axs[0].set_ylim(-1.5, 23)
eyebrow(axs[0], "coherence does not rise with parameter count", y=1.17)
axs[0].annotate("unconverged", xy=(64, float(byname["h64_k32_l5"]["mean_z_cos"])),
                xytext=(34, 12), fontsize=6.6, family="monospace", color=INKMUT,
                arrowprops=dict(arrowstyle="-", color=INKMUT, lw=0.6))
letters(axs)
save(fig, "fig-scaling")

# ---- rl: a classification is a trainable reward ----------------------------

rows = list(csv.DictReader(open(os.path.join(RESULTS, "rl_log.csv"))))
steps = [int(r["step"]) for r in rows]
eff = [float(r["eff"]) for r in rows]
coll = [float(r["coll"]) for r in rows]
fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 1.9))
fig.subplots_adjust(wspace=0.28, bottom=0.25, top=0.82, left=0.09, right=0.99)
a.plot(steps, eff, color=LEAF, lw=1.1)
a.set_xlabel("REINFORCE step")
a.set_ylabel("effect (quantile units)")
monoticks(a)
eyebrow(a, "the reward is learnable")
b.plot(steps, coll, color=INK, lw=1.1)
b.set_xlabel("REINFORCE step")
b.set_ylabel("collateral")
b.set_ylim(0, 0.2)
monoticks(b)
eyebrow(b, "at flat collateral")
letters((a, b))
save(fig, "fig-rl")

# ---- teaser: the proposition, measured -------------------------------------

fig, (a, b, c) = plt.subplots(1, 3, figsize=(5.4, 1.9))
fig.subplots_adjust(wspace=0.42, bottom=0.30, top=0.80, left=0.10, right=0.99)

# (a) unsupervised steering on the shared frame (steer_overlay medians)
arms = ["probe", "onto", "dm"]
effs = [0.072, 0.302, 0.474]
cols = [PINE50, PINE, INK]
a.bar(range(3), effs, color=cols, width=0.62)
a.set_xticks(range(3))
a.set_xticklabels(["probe", "onto", "dm\n(superv.)"], fontsize=6.8,
                  family="monospace")
a.set_ylabel("steering effect (dm frame)")
a.text(1, 0.315, "4× probe", ha="center", fontsize=6.6,
       family="monospace", color=PINE)
monoticks(a)
eyebrow(a, "steers unsupervised")

# (b) composition additivity by intervention layer (working notes)
layers = [0, 1, 2, 3, 4]
coss = [0.959, 0.947, 0.967, 0.996, 1.000]
b.plot(layers, coss, color=PINE, lw=1.2, marker="o", ms=3.2)
b.axhline(1.0, color=INKMUT, lw=0.6, ls=(0, (2, 2)))
b.set_xticks(layers)
b.set_xlabel("intervention layer")
b.set_ylabel("joint vs summed cos")
b.set_ylim(0.93, 1.006)
monoticks(b)
eyebrow(b, "interventions compose")

# (c) code round-trip text fidelity (working notes, forced-English decodes)
names = ["k32", "k160", "onto\nsoft", "k5120\n(dense)"]
chrfs = [0.228, 0.282, 0.552, 0.841]
ccols = [INKMUT, INKMUT, PINE, PINE25]
c.bar(range(4), chrfs, color=ccols, width=0.62)
c.axhline(0.176, color=INKMUT, lw=0.6, ls=(0, (2, 2)))
c.text(3.45, 0.160, "corpus\nfloor", fontsize=6.0, family="monospace",
       color=INKMUT, ha="right", va="top")
c.set_xticks(range(4))
c.set_xticklabels(names, fontsize=6.8, family="monospace")
c.set_ylabel("round-trip chrF")
monoticks(c)
eyebrow(c, "codes keep text")
letters((a, b, c))
save(fig, "fig-teaser")

# ---- training curves: held-out FVU and row cosine per checkpoint ------------
# Scored from checkpoints by extract_curves.py, not read from loss.csv: the
# logged MSE averages every deep-supervision prefix, so it is not the FVU
# the tables report.

def load_curves(family):
    rows = list(csv.DictReader(open(os.path.join(RESULTS,
                                                 f"curves_{family}.csv"))))
    arms = {}
    for r in rows:
        arms.setdefault(r["arm"], []).append(
            (int(r["step"]), float(r["fvu"]), float(r["c"])))
    return {k: sorted(v) for k, v in arms.items()}


def curve(ax, pts, col, **kw):
    ax.plot([p[0] for p in pts], [p[col] for p in pts], lw=1.2, **kw)


DASH = (0, (4, 2))
DOT = (0, (1, 1.5))

# non-negativity x width on GPT-2 layer 8 (tab:orthant)
cv = load_curves("orthant")
ORTHANT = [
    ("ste_h76_sgn", "signed, e=1536", LEAF, "-"),
    ("ste_h76_sgn768", "signed, e=768", LEAF, DASH),
    ("ste_h76", "abs, e=1536", INK, "-"),
    ("ste_h76_e768", "abs, e=768", INK, DASH),
    ("ste_h76_cat32", r"concat, $d_{\mathrm{head}}$=32", INKMUT, DOT),
]
fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 2.6))
fig.subplots_adjust(wspace=0.30, bottom=0.32, top=0.89, left=0.09, right=0.99)
for arm, label, col, ls in ORTHANT:
    curve(a, cv[arm], 1, color=col, ls=ls, label=label)
    curve(b, cv[arm], 2, color=col, ls=ls)
for ax in (a, b):
    ax.axvline(93000, color=INKMUT, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlabel("training step")
    logx(ax)
a.set_ylabel(r"held-out FVU$_w$")
eyebrow(a, "reconstruction")
b.set_ylabel(r"mean row cosine $c$")
# neither panel has a free corner, so the legend goes underneath
fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncols=3,
           handlelength=1.8, fontsize=6.6, bbox_to_anchor=(0.5, 0.0))
eyebrow(b, "dictionary geometry")
letters((a, b))
save(fig, "fig-orthant-curves")

# the s_Hm selection sweep: error flattens, hard arms keep decorrelating
cv = load_curves("sweep")
SWEEP = [
    ("sweep_top2_shm", "top2", INK, "-"),
    ("sweep_top4_shm", "top4", INK, DASH),
    ("sweep_top8_shm", "top8", INKMUT, "-"),
    ("sweep_top16_shm", "top16", INKMUT, DASH),
    ("sweep_softmax_shm", "softmax", LEAF, "-"),
]
fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 2.2))
fig.subplots_adjust(wspace=0.30, bottom=0.20, top=0.87, left=0.09, right=0.99)
for arm, label, col, ls in SWEEP:
    curve(a, cv[arm], 1, color=col, ls=ls, label=label)
    curve(b, cv[arm], 2, color=col, ls=ls)
for ax in (a, b):
    ax.set_xlabel("training step")
    logx(ax)
a.set_ylabel(r"FVU$_w$ (in-sample tail)")
a.legend(loc="upper right", handlelength=1.8, fontsize=6.6)
eyebrow(a, "reconstruction")
b.set_ylabel(r"mean row cosine $c$")
eyebrow(b, "dictionary geometry")
letters((a, b))
save(fig, "fig-sweep-curves")

# depth vs. flat at the fixed initialization, with seed replicas
cv = load_curves("depth")
DEPTH = [
    ("ste_h76_init01", r"5$\times$76", LEAF, "-"),
    ("ste_h76_i01_s43", r"5$\times$76, seed 43", PINE50, DASH),
    ("ste_l1_h380_i01_hm1e4", r"1$\times$380, $s_{H_m}$=1e-4", INK, "-"),
    ("ste_l1_h380_i01", r"1$\times$380", INKMUT, "-"),
    ("ste_l1_h380_i01_s43", r"1$\times$380, seed 43", INKMUT, DASH),
]
fig, ax = plt.subplots(figsize=(5.4, 2.1))
fig.subplots_adjust(bottom=0.21, top=0.86, left=0.12, right=0.99)
for arm, label, col, ls in DEPTH:
    curve(ax, cv[arm], 1, color=col, ls=ls, label=label)
ax.set_xlabel("training step")
ax.set_ylabel(r"held-out FVU$_w$")
# the band between the flat arms and the stack is empty
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.52), ncols=2,
          handlelength=1.8, fontsize=6.4)
logx(ax)
eyebrow(ax, "depth leads at every checkpoint")
save(fig, "fig-depth-curves")

# ---- logged training statistics (loss.csv, not scored from checkpoints) -----

def load_logged(family, *cols):
    rows = list(csv.DictReader(open(os.path.join(RESULTS,
                                                 f"curves_{family}.csv"))))
    arms = {}
    for r in rows:
        arms.setdefault(r["arm"], []).append(
            (int(r["step"]),) + tuple(float(r[c]) for c in cols))
    return {k: sorted(v) for k, v in arms.items()}


# realized code bits, floor-corrected, as a share of nominal
cv = load_logged("bits", "bits", "nominal")
BITS = [
    ("ste_h76", "5×76, shipped init", INK, "-"),
    ("ste_h76_init01", "5×76, fixed init", LEAF, "-"),
    ("ste_h76_i01_s43", "5×76, fixed init, seed 43", PINE50, DASH),
    ("ste_l1_h380_i01_hm1e4", "1×380, fixed init", INKMUT, DOT),
]
fig, ax = plt.subplots(figsize=(5.4, 2.1))
fig.subplots_adjust(bottom=0.21, top=0.86, left=0.10, right=0.99)
for arm, label, col, ls in BITS:
    pts = cv[arm]
    ax.plot([p[0] for p in pts], [100 * p[1] / p[2] for p in pts],
            color=col, ls=ls, lw=1.2, label=label)
ax.set_xlabel("training step")
ax.set_ylabel("realized bits (% of 1900)")
ax.set_ylim(40, 101)
ax.legend(loc="lower right", handlelength=1.8, fontsize=6.6)
logx(ax)
eyebrow(ax, "the fixed initialization fills the code at once")
save(fig, "fig-realized-bits")

# the row-collinearity setpoint: the watched max, and the applied multiplier
cv = load_logged("kcos", "cossim_k_max", "s_kcossim")
KCOS = [("ste_h76", "uncontrolled", INK, "-"),
        ("ste_h76_kcos", "setpoint 0.5", LEAF, "-")]
fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 2.2))
fig.subplots_adjust(wspace=0.34, bottom=0.20, top=0.87, left=0.09, right=0.99)
for arm, label, col, ls in KCOS:
    curve(a, cv[arm], 1, color=col, ls=ls, label=label)
a.axhline(0.5, color=INKMUT, lw=0.6, ls=(0, (2, 2)))
a.set_xlabel("training step")
a.set_ylabel(r"max row cosine (heads, layers)")
a.legend(loc="lower left", handlelength=1.8, fontsize=6.6)
logx(a)
eyebrow(a, "pulled to 0.5, it stays there")
pts = cv["ste_h76_kcos"]
b.plot([p[0] for p in pts], [1e3 * p[2] for p in pts], color=LEAF, lw=1.0)
b.set_xlabel("training step")
b.set_ylabel(r"applied $s_{\mathrm{kcossim}}$ ($\times 10^{-3}$)")
logx(b)
eyebrow(b, "one burst, then idle")
letters((a, b))
save(fig, "fig-kcos")

# ---- headline results, from the eval scripts' own outputs -------------------
# extract_results.py copies each figure's numbers out of data/out, so none is
# restated from a table; the GPT-2 seed pair is the one exception, below.
# Families keep one color across these figures: Ontologizers LEAF, SAEs ink,
# the single-layer variants muted, references dashed.

def load_rows(name):
    with open(os.path.join(RESULTS, name)) as f:
        return list(csv.DictReader(f))


def mono_axes(ax):
    """Mono tick labels that survive a later rescale, which `monoticks`
    (styling only the labels that exist when called) does not."""
    ax.tick_params(labelfontfamily="monospace", labelsize=7.0)


def percent_axis(axis, ticks):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}%"))
    axis.set_minor_formatter(NullFormatter())


RING = dict(mec="white", mew=0.6)  # keeps overlapping markers apart
NULL = dict(color=INKMUT, lw=0.6, ls=(0, (2, 2)))

# reconstruction against code size: the Pareto table, the capacity table
# and the discrete-versus-linear frontier in one plane, by index bits (a)
# and by continuous coefficients (b)
P = load_rows("pareto_sonar.csv")
G = load_rows("ratefloor_sonar.csv")


def frontier(family, axis, kinds=None):
    """(x, fvu) of a family's points along `axis`, sorted; a point at zero
    has no place on a log axis and is drawn separately if at all."""
    pts = [(float(r[axis]), float(r["fvu"])) for r in P
           if r["family"] == family and (kinds is None or r["kind"] in kinds)]
    return sorted(p for p in pts if p[0] > 0)


TOPK_FRONT = ["m11264_k32", "m11264_k160", "m11264_k5120"]
TOPK_REST = ["m5120_k32", "m5120_k32_s43", "m5120_k32_p5"]
ste = next(r for r in P if r["family"] == "ste" and r["kind"] == "hard")
argmax = next(r for r in P if r["family"] == "softmax" and r["kind"] == "hard")
fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 3.0), sharey=True)
fig.subplots_adjust(wspace=0.08, bottom=0.34, top=0.90, left=0.10, right=0.99)
a.plot([float(r["bits"]) for r in G], [float(r["fvu"]) for r in G],
       color=INKMUT, lw=0.9, ls=DASH, label="Gaussian reference", zorder=2)
for ax, axis in ((a, "bits"), (b, "coeffs")):
    ax.plot(*zip(*frontier("sae_topk", axis, TOPK_FRONT)), color=INK, lw=1.0,
            marker="s", ms=3.6, **RING, label=r"$\mathtt{TopK}$ SAE", zorder=3)
    ax.plot(*zip(*frontier("sae_topk", axis, TOPK_REST)), ls="none",
            color=INK, marker="s", ms=3.6, **RING, zorder=3)
    ax.plot(*zip(*frontier("sae_l1", axis)), color=INK, lw=0.8, ls=DOT,
            marker="o", ms=3.4, mfc="white", label="L1 SAE", zorder=3)
    ax.plot(*zip(*frontier("sae_bilinear", axis)), ls="none", color=INK,
            marker="^", ms=3.8, mfc="white", label="bilinear SAE", zorder=3)
    ax.plot(*zip(*frontier("softmax", axis, ["dev", "soft"])), color=LEAF,
            lw=1.2, marker="o", ms=3.6, mfc="white",
            label=r"softmax, top $\nu$ per head", zorder=4)
    ax.plot(*zip(*frontier("variant", axis)), ls="none", color=INKMUT,
            marker="v", ms=4.4, **RING, label="single-layer variants",
            zorder=4)
a.plot([float(argmax["bits"])], [float(argmax["fvu"])], ls="none",
       color=LEAF, marker="x", ms=4.5, mew=1.2, zorder=4)
a.annotate("softmax argmax", xy=(float(argmax["bits"]), float(argmax["fvu"])),
           xytext=(6, -2), textcoords="offset points", fontsize=6.6,
           color=INK, va="center")
a.plot([float(ste["bits"])], [float(ste["fvu"])], ls="none", color=LEAF,
       marker="D", ms=5.5, **RING, zorder=5,
       label=r"straight-through, 5$\times$76")
# the hard code transmits no coefficients, so in (b) it is a level, not a point
b.axhline(float(ste["fvu"]), color=LEAF, lw=0.8, ls=DASH, zorder=2)
b.annotate(r"5$\times$76, no coefficients",
           xy=(3.0, float(ste["fvu"])), xytext=(0, -3),
           textcoords="offset points", fontsize=6.6, color=INK, ha="left",
           va="top")
for ax in (a, b):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    mono_axes(ax)
a.set_xlim(30, 4e4)
b.set_xlim(2.5, 1.6e4)
a.set_ylim(5e-5, 60)
a.set_xlabel("index bits per sample")
b.set_xlabel("continuous coefficients per sample")
a.set_ylabel(r"FVU$_w$, evaluation tail")
eyebrow(a, "by bits")
eyebrow(b, "by coefficients")
fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncols=4,
           handlelength=1.8, fontsize=6.6, bbox_to_anchor=(0.5, 0.0))
letters((a, b))
save(fig, "fig-pareto")

# steering trade-off: realization against collateral as the step grows from
# 0.25 to 4 times the input norm, one draw of targets per model
S = load_rows("steer_sweep.csv")


def sweep(model, kind):
    pts = sorted((float(r["strength"]), float(r["collateral"]),
                  float(r["hit"])) for r in S
                 if r["model"] == model and r["kind"] == kind)
    return [p[1] for p in pts], [p[2] for p in pts]


DIRS = [
    ("decode", "own decode direction", LEAF, "-",
     dict(marker="o", ms=3.4, **RING)),
    ("dm", "difference of means (supervised)", INK, "-",
     dict(marker="s", ms=3.2, **RING)),
    ("grad", "logit gradient", INK, DASH, dict(marker="^", ms=3.4,
                                               mfc="white")),
    ("random", "random direction", INKMUT, DOT, dict(marker="o", ms=3.0,
                                                     mfc="white")),
]
PANELS = [
    ("softmax stack", "softmax stack"),
    ("hard-code stack", "hard-code stack, 5×76"),
    ("hard-code flat", "hard-code flat, 1×380"),
    ("g160top1", "single-layer hard, g160top1"),
]
fig, axs = plt.subplots(2, 2, figsize=(5.4, 4.4), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.08, hspace=0.30, bottom=0.18, top=0.94,
                    left=0.10, right=0.98)
for ax, (model, title) in zip(axs.flat, PANELS):
    for kind, label, col, ls, mk in DIRS:
        ax.plot(*sweep(model, kind), color=col, ls=ls, lw=1.1, label=label,
                **mk)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.04, 1.06)
    mono_axes(ax)
    eyebrow(ax, title)
# the step sizes, once, where the random curve has room above it
x, y = sweep("g160top1", "random")
for i, s in ((0, "0.25"), (-1, "4")):
    axs[1, 1].annotate(s, xy=(x[i], y[i]), xytext=(0, 7),
                       textcoords="offset points", fontsize=6.2,
                       family="monospace", color=INKMUT, ha="center")
fig.supxlabel("collateral (share of the other heads changed)", y=0.095,
              fontsize=8.5)
for ax in axs[:, 0]:
    ax.set_ylabel("realized")
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center",
           ncols=2, handlelength=2.2, fontsize=6.6, bbox_to_anchor=(0.5, 0.0))
letters(axs.flat)
save(fig, "fig-steer-curves")

# within-head decode-direction coherence: every head's z against its
# size-matched null, by grouping
H = load_rows("headcoh_z.csv")
GROUPS = [
    ("g160top1", "g160top1\n(trained, hard)", INKMUT),
    ("k32 discovered", "$\\mathtt{TopK}$ SAE groups\n(discovered)", INK),
    ("onto heads", "Ontologizer heads\n(trained, soft)", LEAF),
    ("g160softmax", "g160softmax\n(trained, soft)", INKMUT),
]
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(5.4, 2.3))
fig.subplots_adjust(bottom=0.17, top=0.88, left=0.11, right=0.99)
for i, (g, _, col) in enumerate(GROUPS):
    z = np.array([float(r["z_cos"]) for r in H if r["grouping"] == g])
    ax.plot(i + rng.uniform(-0.22, 0.22, len(z)), z, ls="none", marker="o",
            ms=2.4, color=col, mec="none", alpha=0.6, zorder=3)
    ax.plot([i - 0.3, i + 0.3], [z.mean()] * 2, color=INK, lw=1.4,
            solid_capstyle="butt", zorder=4)
    ax.text(i, 900, f"{(z > 2).mean():.0%} above +2\n"
                    f"{(z < -2).mean():.0%} below −2",
            ha="center", va="center", fontsize=6.2, family="monospace",
            color=INKMUT)
for v in (-2, 2):
    ax.axhline(v, **NULL, zorder=1)
ax.set_yscale("symlog", linthresh=2, linscale=0.6)
ticks = [-30, -10, -2, 0, 2, 10, 100]
ax.yaxis.set_major_locator(FixedLocator(ticks))
ax.yaxis.set_major_formatter(FuncFormatter(
    lambda v, _: f"{v:+g}".replace("-", "−") if v else "0"))
ax.yaxis.set_minor_locator(FixedLocator([]))
ax.set_ylim(-45, 2500)
ax.set_xlim(-0.55, 3.55)
ax.set_xticks(range(len(GROUPS)))
ax.set_xticklabels([g[1] for g in GROUPS], fontsize=7.0)
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", labelfontfamily="monospace", labelsize=7.0)
ax.set_ylabel(r"within-head $\zeta_{\cos}$")
eyebrow(ax, "hard heads cohere, soft heads repel")
save(fig, "fig-headcoh")

# describability against code density: (a) activation descriptions by
# nominal density, (b) contrastive descriptions by firing rate, fitted on the
# sae.py runs with the Ontologizer held out
A = load_rows("autointerp_density.csv")
FAM = {"onto": (LEAF, "o"), "k32": (INK, "s"), "k32_bl": (INK, "s"),
       "k5120": (INK, "s"), "g160top1": (INKMUT, "v"),
       "g160softmax": (INKMUT, "v")}
NAME = {"onto": "resid_nc"}
# label offsets (points) that keep the six names apart
OFF_A = {"onto": (0, 8, "center"), "k32": (6, 1, "left"),
         "k32_bl": (6, -1, "left"), "k5120": (-6, -1, "right"),
         "g160top1": (6, 2, "left"), "g160softmax": (0, -8, "center")}
OFF_B = {"onto": (6, -4, "left"), "k32": (6, 2, "left"),
         "k32_bl": (6, -3, "left"), "k5120": (-6, -5, "right"),
         "g160top1": (6, 2, "left")}
fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 2.7), sharey=True)
fig.subplots_adjust(wspace=0.08, bottom=0.31, top=0.89, left=0.10, right=0.99)
for ax, xkey, ykey, off in ((a, "density", "acts", OFF_A),
                            (b, "freq", "cacts", OFF_B)):
    for r in A:
        x, y = float(r[xkey]), float(r[ykey])
        if x <= 0:
            continue  # g160softmax never crosses the firing threshold
        col, mk = FAM[r["run"]]
        ax.plot([x], [y], ls="none", color=col, marker=mk, ms=4.6, **RING,
                zorder=4)
        dx, dy, ha = off[r["run"]]
        ax.annotate(NAME.get(r["run"], r["run"]), xy=(x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.0,
                    family="monospace", color=INK, ha=ha, va="center")
    # a description that matches every item scores F1 = 2/3 on balanced sets
    ax.axhline(2 / 3, **NULL, zorder=1)
    ax.set_xscale("log")
    mono_axes(ax)
b.text(0.28, 2 / 3, "always-match null", fontsize=6.2, color=INKMUT,
       ha="right", va="bottom", family="monospace")
x = np.log([float(r["density"]) for r in A])
y = [float(r["acts"]) for r in A]
a.text(0.97, 0.05, f"correlation {np.corrcoef(x, y)[0, 1]:+.2f}",
       transform=a.transAxes, ha="right", fontsize=6.4, color=INK,
       family="monospace")
fit = [r for r in A if r["sae_py"] == "1" and float(r["freq"]) > 0]
slope, icept = np.polyfit(np.log([float(r["freq"]) for r in fit]),
                          [float(r["cacts"]) for r in fit], 1)
xs = np.geomspace(0.004, 0.3, 50)
b.plot(xs, icept + slope * np.log(xs), color=INK, lw=0.8, zorder=2,
       label="fit to the sae.py runs")
# the one run well off the fit, measured to it
top1 = next(r for r in A if r["run"] == "g160top1")
fx, fy = float(top1["freq"]), float(top1["cacts"])
on_fit = icept + slope * np.log(fx)
b.plot([fx, fx], [on_fit, fy], color=INKMUT, lw=0.6, ls=DOT, zorder=2)
b.annotate(f"{fy - on_fit:+.2f}", xy=(fx, (fy + on_fit) / 2), xytext=(4, 0),
           textcoords="offset points", fontsize=6.2, family="monospace",
           color=INKMUT, va="center")
percent_axis(a.xaxis, [0.01, 0.1, 1.0])
percent_axis(b.xaxis, [0.01, 0.1])
a.set_xlim(0.0035, 2.2)
b.set_xlim(0.004, 0.3)
a.set_ylim(0.3, 0.72)
a.set_xlabel("nominal code density")
b.set_xlabel(r"firing rate at the $2/k$ threshold")
a.set_ylabel("detection F1")
eyebrow(a, "activation descriptions")
eyebrow(b, "contrastive descriptions")
handles = [Line2D([], [], ls="none", color=c, marker=m, ms=4.6, **RING)
           for c, m in ((LEAF, "o"), (INK, "s"), (INKMUT, "v"))]
handles += [Line2D([], [], color=INK, lw=0.8)]
fig.legend(handles, ["Ontologizer", "SAEs", "single-layer variants",
                     "fit to the sae.py runs"],
           loc="lower center", ncols=4, handlelength=1.8, fontsize=6.6,
           bbox_to_anchor=(0.5, 0.0))
letters((a, b))
save(fig, "fig-autointerp-density")

# GPT-2 concatenated-head seed pair by layer (tab:gpt2seeds). partition.py
# and headcontrib.py print rather than save; these are their outputs as
# recorded in experiments/ste-arm/notes.md, "The same picture on GPT-2".
LAYERS = [0, 1, 2, 3, 4]
SEEDS = {"partition NMI": ([0.359, 0.188, 0.186, 0.186, 0.186],
                           [0.976, 0.766, 0.602, 0.538, 0.510], 0.179, None),
         "contribution cosine": ([0.171, 0.017, 0.007, 0.005, 0.004],
                                 [0.956, 0.758, 0.588, 0.516, 0.483],
                                 0.0017, 0.0030)}
fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 2.3))
fig.subplots_adjust(wspace=0.32, bottom=0.30, top=0.88, left=0.10, right=0.99)
for ax, (measure, (seeds, ctrl, null, null_max)) in zip((a, b), SEEDS.items()):
    ax.plot(LAYERS, ctrl, color=INK, lw=1.1, marker="s", ms=3.6, **RING,
            label="same run, step 350k (control)", zorder=3)
    ax.plot(LAYERS, seeds, color=LEAF, lw=1.2, marker="o", ms=4.0, **RING,
            label="across seeds", zorder=4)
    ax.axhline(null, **NULL, label="null", zorder=1)
    if null_max is not None:
        ax.axhspan(null, null_max, color=INKMUT, alpha=0.15, lw=0, zorder=1)
    ax.set_xticks(LAYERS)
    ax.set_xlabel("layer")
    ax.set_ylabel(measure)
    mono_axes(ax)
a.set_ylim(0, 1.04)
b.set_yscale("log")
b.set_ylim(1e-3, 1.5)
eyebrow(a, "what each head separates")
eyebrow(b, "what each head writes")
fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncols=3,
           handlelength=1.8, fontsize=6.6, bbox_to_anchor=(0.5, 0.0))
letters((a, b))
save(fig, "fig-gpt2-seeds")

# classifier geometry from the weights (bilinspec.py): |cos(w, v)| of each
# bilinear feature's pair, 1 when the feature is rank 1, by layer at the final
# checkpoint (a) and over training (b, c). The reference is the median |cos|
# of a random pair in the d = 1024 semantic block, 0.6745 / sqrt(d)
BL = load_rows("bilinspec.csv")
RANDOM_COS = 0.6745 / np.sqrt(1024)
STACKS = [("softmax stack", LEAF, "o"), ("hard-code stack", INK, "s")]
LAYER_COLS = [matplotlib.colors.LinearSegmentedColormap.from_list(
    "layers", [PINE50, LEAF, PINE, INK])(v) for v in np.linspace(0, 1, 5)]


def geometry(model, step=None):
    """{layer: row} at `step` (default: the run's last checkpoint)."""
    rows = [r for r in BL if r["model"] == model]
    step = step or max(int(r["step"]) for r in rows)
    return {int(r["layer"]): r for r in rows if int(r["step"]) == step}


def spread(ys, gap):
    """Label heights for line ends at `ys`, pushed apart to at least `gap`
    while keeping their order."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for a, b in zip(order[:-1], order[1:]):
        out[b] = max(out[b], out[a] + gap)
    return out


fig, axs = plt.subplots(1, 3, figsize=(5.4, 2.45), sharey=True,
                        gridspec_kw=dict(width_ratios=[0.9, 1, 1]))
fig.subplots_adjust(wspace=0.12, bottom=0.30, top=0.88, left=0.10, right=0.98)
a = axs[0]
for (model, col, mk), dx in zip(STACKS, (-0.12, 0.12)):
    g = geometry(model)
    xs = np.arange(5) + dx
    med = [float(g[l]["abscos_med"]) for l in range(5)]
    lo = [med[l] - float(g[l]["abscos_q25"]) for l in range(5)]
    hi = [float(g[l]["abscos_q75"]) - med[l] for l in range(5)]
    a.errorbar(xs, med, yerr=[lo, hi], ls="none", color=col, marker=mk,
               ms=4.0, elinewidth=0.9, capsize=0, label=model, zorder=3,
               **RING)
a.set_xticks(range(5))
a.set_xlabel("layer")
a.set_ylabel(r"median $|\cos(\mathbf{w}, \mathbf{v})|$")
eyebrow(a, "final checkpoint")
for ax, (model, _, _) in zip(axs[1:], STACKS):
    rows = [r for r in BL if r["model"] == model]
    ends = []
    for l in range(5):
        pts = sorted((int(r["step"]), float(r["abscos_med"])) for r in rows
                     if int(r["layer"]) == l)
        ax.plot(*zip(*pts), color=LAYER_COLS[l], lw=1.1, marker="o", ms=2.4,
                zorder=3)
        ends.append(pts[-1])
    # the layer number at each line's end, kept legible where lines converge
    for l, ((x, _), y) in enumerate(zip(ends, spread([e[1] for e in ends],
                                                        0.05))):
        ax.annotate(str(l), xy=(x, y), xytext=(4, 0),
                    textcoords="offset points", fontsize=6.2,
                    family="monospace", color=LAYER_COLS[l], va="center")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1e4, 3e4, 1e5, 3e5]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e3:g}k"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(8e3, 5.2e5)
    ax.set_xlabel("training step")
    eyebrow(ax, model)
# the softmax stack's temperature anneal ends at 50k; the hard-code stack's
# temperature is constant
axs[1].axvline(5e4, color=INKMUT, lw=0.6, ls=DOT, zorder=1)
axs[1].text(5.4e4, 0.97, "anneal\nends", fontsize=6.0, family="monospace",
            color=INKMUT, va="top")
for ax in axs:
    ax.axhline(RANDOM_COS, **NULL, zorder=1)
    ax.set_ylim(0, 1.0)
    mono_axes(ax)
axs[1].text(5.0e5, RANDOM_COS + 0.015, "random pair", fontsize=6.2,
            color=INKMUT, ha="right", va="bottom", family="monospace")
handles = [Line2D([], [], ls="none", color=c, marker=m, ms=4.0, **RING)
           for _, c, m in STACKS]
fig.legend(handles, [s[0] for s in STACKS], loc="lower center", ncols=2,
           handlelength=1.0, fontsize=6.6, bbox_to_anchor=(0.5, 0.0))
letters(axs)
save(fig, "fig-bilinspec")

# head partitions as communities of the input (headstruct.py --modularity):
# soft modularity over its size-matched null, (a) for every model scored on
# the input itself and (b) by depth, each layer or prefix block on the graph
# of its own input. Bars are +-2 null standard deviations carried to the
# ratio; a ratio whose null is within two of its spread of 0 is not drawn
MD = load_rows("modularity.csv")
MOD_STYLE = {
    "Ontologizer": (LEAF, "o"),
    "Matryoshka, trained heads": (INK, "s"),
    "Matryoshka, discovered groups": (INKMUT, "^"),
    "flat, trained heads": (INK, "s"),
    "flat, discovered groups": (INKMUT, "^"),
}


def ratio(r):
    """(real / null, its 2-sigma half-width), or None for an unstable null."""
    real, null, sd = float(r["real"]), float(r["null"]), float(r["null_sd"])
    if null <= 2 * sd:
        return None
    return real / null, 2 * real * sd / null ** 2


fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 2.5),
                           gridspec_kw=dict(width_ratios=[1, 1.15]))
fig.subplots_adjust(wspace=0.55, bottom=0.30, top=0.88, left=0.23, right=0.98)
ON_X = [("Ontologizer", "0", "Ontologizer, layer 0"),
        ("Matryoshka, trained heads", "0", "Matryoshka, trained"),
        ("Matryoshka, discovered groups", "0", "Matryoshka, discovered"),
        ("flat, trained heads", "", "flat, trained"),
        ("flat, discovered groups", "", "flat, discovered")]
for y, (model, block, label) in enumerate(ON_X):
    r = next(r for r in MD if r["model"] == model and r["block"] == block)
    v, e = ratio(r)
    col, mk = MOD_STYLE[model]
    a.errorbar([v], [y], xerr=[e], ls="none", color=col, marker=mk, ms=4.2,
               elinewidth=0.9, capsize=0, zorder=3, **RING)
a.set_yticks(range(len(ON_X)))
a.set_yticklabels([s[2] for s in ON_X], fontsize=6.8)
a.set_ylim(len(ON_X) - 0.5, -0.5)
a.tick_params(axis="y", length=0)
a.axvline(1, **NULL, zorder=1)
a.set_xlim(0.6, 1.45)
a.set_xlabel("modularity / null")
eyebrow(a, "scored on the input")
for model in ("Ontologizer", "Matryoshka, trained heads",
              "Matryoshka, discovered groups"):
    col, mk = MOD_STYLE[model]
    pts = [(int(r["block"]), *ratio(r), float(r["coverage"])) for r in MD
           if r["model"] == model and ratio(r) is not None]
    xs, vs, es, cov = zip(*pts)
    b.plot(xs, vs, color=col, lw=1.0, zorder=2)
    for x, v, e, c in pts:
        b.errorbar([x], [v], yerr=[e], ls="none", color=col, marker=mk,
                   ms=4.2, elinewidth=0.9, capsize=0, zorder=3,
                   mfc=col if c >= 0.1 else "white", mec=col if c < 0.1
                   else "white", mew=0.6 if c >= 0.1 else 0.9)
b.axhline(1, **NULL, zorder=1)
b.set_xticks(range(5))
b.set_xlabel("layer or prefix block")
b.set_ylabel("modularity / null")
b.set_ylim(0.5, 3.3)
eyebrow(b, "on each layer's own input")
for ax in (a, b):
    mono_axes(ax)
a.tick_params(axis="y", labelfontfamily="serif", labelsize=6.8)
handles = [Line2D([], [], ls="none", color=c, marker=m, ms=4.2, **RING)
           for c, m in (MOD_STYLE["Ontologizer"],
                        MOD_STYLE["Matryoshka, trained heads"],
                        MOD_STYLE["Matryoshka, discovered groups"])]
fig.legend(handles, ["Ontologizer heads", "SAE trained heads",
                     "SAE discovered groups"], loc="lower center", ncols=3,
           handlelength=1.0, fontsize=6.6, bbox_to_anchor=(0.6, 0.0))
letters((a, b))
save(fig, "fig-modularity")


# ---- per-sample and per-entry views (assignmap, entrydrift, enrich,
# headnmi, selection), from their extracted outputs -------------------------

def seq_cmap(color, name):
    """White at 0 rising to `color`, as anatomy.py's blocks."""
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        name, ["#FFFFFF", color])


DIV = matplotlib.colors.LinearSegmentedColormap.from_list(
    "inkpine", [INK, "#FFFFFF", PINE])


def load_npz(name):
    return np.load(os.path.join(RESULTS, name))


def hairlines(ax, edges, axis, **kw):
    """Separator lines between consecutive blocks ending at `edges`."""
    line = ax.axhline if axis == "y" else ax.axvline
    for e in edges[:-1]:
        line(e - 0.5, **{"color": INKMUT, "lw": 0.4, **kw})


# assignmap.py --sae: a hard-code stack's layer-0 assignments for the three
# heads most informative about the label (a, c), beside an SAE's code on the
# same rows in the same order (b, d). Rows are 24 per label slice, ordered
# within a slice by their argmax entries; entries and latents by usage
Z = load_npz("assignmap.npz")
fig = plt.figure(figsize=(5.4, 5.0))
n_rows = [len(Z[f"H{i}"]) for i in range(2)]
outer = fig.add_gridspec(2, 1, height_ratios=[1, 0.03], hspace=0.2,
                         left=0.10, right=0.98, top=0.95, bottom=0.07)
gs = outer[0].subgridspec(2, 2, height_ratios=n_rows, hspace=0.42,
                          wspace=0.08)
gcb = outer[1].subgridspec(1, 2, wspace=0.2)
axes = []
for i in range(2):
    H, S = Z[f"H{i}"].astype(float), Z[f"S{i}"].astype(float)
    k = H.shape[1] // len(Z[f"heads{i}"])
    use, row = Z[f"use{i}"], Z[f"row{i}"]
    sae_use, sae_row, mass, l0, m, _ = Z[f"sae{i}"]
    edges = np.cumsum(Z[f"slices{i}"])
    a = fig.add_subplot(gs[i, 0])
    im_h = a.imshow(H, aspect="auto", interpolation="nearest",
                    cmap=seq_cmap(PINE, "pine"), vmin=0, vmax=1)
    hairlines(a, np.arange(1, len(use) + 1) * k, "x", color=INK, lw=0.5)
    a.set_xticks([(j + 0.5) * k - 0.5 for j in range(len(use))],
                 [f"h{h}\nuse {u:.1f}" for h, u in zip(Z[f"heads{i}"], use)])
    a.set_yticks(edges - np.asarray(Z[f"slices{i}"]) / 2 - 0.5,
                 Z[f"names{i}"], fontsize=6.0)
    a.tick_params(length=0)
    eyebrow(a, f"{Z[f'panel{i}']}: {Z[f'run{i}']}, layer {int(Z[f'layer{i}'])}")
    b = fig.add_subplot(gs[i, 1], sharey=a)
    vmax_s = 2 / l0
    im_s = b.imshow(S, aspect="auto", interpolation="nearest",
                    cmap=seq_cmap(INK, "ink"), vmin=0, vmax=vmax_s)
    b.set_xticks([S.shape[1] / 2 - 0.5],
                 [f"first {S.shape[1]} of {int(m):,} latents ({mass:.0%} of "
                  f"usage)\nuse {sae_use:,.0f}, row {sae_row:.1f}"])
    b.tick_params(length=0, labelleft=False)
    eyebrow(b, f"SAE {Z[f'sae_run{i}']}, same rows")
    for ax in (a, b):
        hairlines(ax, edges, "y")
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(6.2)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.5)
    axes += [a, b]
for j, (im, label, top) in enumerate(
        [(im_h, "assignment probability", 1.0),
         (im_s, "share of the row's SAE code", None)]):
    cax = fig.add_subplot(gcb[0, j])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      extend="max" if top is None else "neither")
    cb.set_label(label, fontsize=6.6)
    cb.ax.tick_params(labelsize=6.0)
    cb.outline.set_linewidth(0.4)
letters(axes)
save(fig, "fig-assignmap")


# entrydrift.py over resid_nc's checkpoints: two heads' entries step by step
# (a, b: usage, entries ordered by usage at the last checkpoint), and per
# layer the median cosine of each used entry's decoded direction with its
# final one (c) and the share of entries in use (d)
ZD = load_npz("entrydrift.npz")
ED = load_rows("entrydrift.csv")
steps = ZD["steps"]
fig = plt.figure(figsize=(5.4, 4.1))
outer = fig.add_gridspec(2, 1, hspace=0.55, left=0.09, right=0.92, top=0.92,
                         bottom=0.2)
gtop = outer[0].subgridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.32)
gbot = outer[1].subgridspec(1, 2, wspace=0.32)
top_ = 4 / int(ZD["k"])
ticks = [j for j, s in enumerate(steps) if s in (10000, 60000, 200000, 300000)
         or j == len(steps) - 1]
anneal_col = np.searchsorted(steps, int(ZD["mark_step"]), side="right") - 0.5
axes = []
for i in range(2):
    a = fig.add_subplot(gtop[0, i])
    im = a.imshow(ZD[f"usage{i}"], aspect="auto", interpolation="nearest",
                  cmap=seq_cmap(PINE, "pine"), vmin=0, vmax=top_)
    a.axvline(anneal_col, color=INKMUT, lw=0.7, ls=DOT)
    a.set_xticks(ticks, [f"{steps[j] / 1e3:g}k" for j in ticks], rotation=0)
    if i == 0:
        a.set_ylabel("entry (by final usage)")
    a.set_yticks([0, 31])
    L, h = ZD[f"head{i}"]
    eyebrow(a, f"layer {L}, head {h}: usage")
    monoticks(a)
    axes.append(a)
cb = fig.colorbar(im, cax=fig.add_subplot(gtop[0, 2]), extend="max")
cb.set_ticks([0, 0.05, 0.1])
cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
cb.set_label(r"usage $\bar p$", fontsize=7.0)
cb.ax.tick_params(labelsize=6.0)
for j, (col, ylab, lim) in enumerate([
        ("median_drift_used", "median cos to final", (0, 1.02)),
        ("frac_used", "share of entries in use", (0, 1.02))]):
    ax = fig.add_subplot(gbot[0, j])
    for L in range(5):
        pts = [(int(r["step"]), float(r[col])) for r in ED
               if int(r["layer"]) == L]
        ax.plot(*zip(*pts), color=LAYER_COLS[L], lw=1.1, marker="o", ms=2.2)
    ax.axvline(int(ZD["mark_step"]), color=INKMUT, lw=0.6, ls=DOT, zorder=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1e4, 3e4, 1e5, 3e5]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e3:g}k"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(8e3, 4.6e5)
    ax.set_ylim(*lim)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylab)
    eyebrow(ax, "by layer" if j == 0 else "usage >= 1/(4k), by layer")
    mono_axes(ax)
    axes.append(ax)
axes[2].text(5.3e4, 0.05, "anneal\nends", fontsize=6.0, family="monospace",
             color=INKMUT, va="bottom")
fig.legend([Line2D([], [], color=LAYER_COLS[L], lw=1.1, marker="o", ms=2.2)
            for L in range(5)], [f"layer {L}" for L in range(5)],
           loc="lower center", ncols=5, handlelength=1.6, fontsize=6.6,
           bbox_to_anchor=(0.53, 0.0))
letters(axes)
save(fig, "fig-entrydrift")


# enrich.py: null-subtracted NMI between each head's argmax and a label, by
# layer (a; one point per head, floored at 1e-4 where at or below chance), and
# one GPT-2 layer-0 head's entries against token class (b): colour the
# Haldane-corrected log2 odds ratio, area -log10 FDR, filled when FDR < 0.05
EN = load_rows("enrich_nmi.csv")
EC = load_rows("enrich_cells.csv")
ENRICH_STYLE = [("SONAR", "lang", "SONAR resid_nc, language", LEAF, "o"),
                ("GPT-2", "tokclass", "GPT-2 ste_h76, token class", INK, "s"),
                ("GPT-2", "pos", "GPT-2 ste_h76, position", INKMUT, "^")]
FLOOR = 1e-4
fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 3.3),
                           gridspec_kw=dict(width_ratios=[1.35, 1]))
fig.subplots_adjust(wspace=0.55, bottom=0.25, top=0.9, left=0.11, right=0.86)
rng = np.random.default_rng(0)
for (sub, by, label, col, mk), dx in zip(ENRICH_STYLE, (-0.25, 0, 0.25)):
    for L in range(5):
        v = np.array([float(r["excess"]) for r in EN if r["substrate"] == sub
                      and r["by"] == by and int(r["layer"]) == L])
        x = L + dx + rng.uniform(-0.08, 0.08, len(v))
        a.scatter(x, np.maximum(v, FLOOR), s=3.0, color=col, marker=mk,
                  lw=0, alpha=0.55, zorder=2)
        a.plot([L + dx - 0.1, L + dx + 0.1], [max(np.median(v), FLOOR)] * 2,
               color=col, lw=1.4, zorder=3)
a.set_yscale("log")
a.set_ylim(FLOOR * 0.8, 0.3)
a.set_xticks(range(5))
a.set_xlabel("layer")
a.set_ylabel("NMI(head, label) − null")
eyebrow(a, "each head; bar = layer median")
mono_axes(a)
handles = [Line2D([], [], ls="none", color=c, marker=m, ms=3.6)
           for *_, c, m in ENRICH_STYLE]
fig.legend(handles, [s[2] for s in ENRICH_STYLE], loc="lower center",
           ncols=3, handlelength=1.0, fontsize=6.4, bbox_to_anchor=(0.45, 0.0))
labels = sorted({r["label"] for r in EC},
                key=["punct", "word", "Word", "cont", "newline", "digit",
                     "space"].index)
# the data green at the enriched end stays apart from ink by lightness
DOT_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "inkleaf", [INK, "#FFFFFF", LEAF])


def top_label(e):
    """Entry e's sort key: the label it is most enriched for, then usage."""
    best = max((r for r in EC if int(r["entry"]) == e),
               key=lambda r: float(r["log2OR"]))
    return labels.index(best["label"]), -int(best["n_entry"])


entries = sorted({int(r["entry"]) for r in EC}, key=top_label)
for r in EC:
    y, x = entries.index(int(r["entry"])), labels.index(r["label"])
    q = min(float(r["neglog10_FDR"]), 50.0)
    sig = q > -np.log10(0.05)
    c = DOT_CMAP((np.clip(float(r["log2OR"]), -4, 4) + 4) / 8)
    b.scatter([x], [y], s=4 + 40 * q / 50, color=c if sig else "white",
              edgecolors=INK if sig else INKMUT,
              linewidths=0.3 if sig else 0.5, zorder=3)
b.set_xticks(range(len(labels)), labels, rotation=55, ha="right",
             fontsize=6.4)
b.set_yticks(range(len(entries)), [f"e{e}" for e in entries], fontsize=5.6)
b.set_ylim(len(entries) - 0.5, -0.5)
b.set_xlim(-0.6, len(labels) - 0.4)
b.tick_params(length=0)
b.grid(color=PINE25, lw=0.3)
b.set_axisbelow(True)
eyebrow(b, f"GPT-2 layer 0, head {EC[0]['head']}")
sm = matplotlib.cm.ScalarMappable(cmap=DOT_CMAP,
                                  norm=matplotlib.colors.Normalize(-4, 4))
cb = fig.colorbar(sm, ax=b, fraction=0.08, pad=0.04, extend="both")
cb.set_label(r"$\log_2$ odds ratio", fontsize=6.6)
cb.ax.tick_params(labelsize=6.0)
letters((a, b))
save(fig, "fig-enrich")


# headnmi.py --ei on the SONAR hard-code stack ste_h76: adjusted NMI between
# every pair of heads' partitions (a; heads blocked by layer, hierarchically
# ordered within a layer), the on-distribution effective information in the
# same order (b; source rows, effect columns, defined only for later effect
# layers), and one against the other per pair (c), coloured by source layer
ZN = load_npz("headnmi.npz")
adj, ei = ZN["adj"].astype(float), ZN["ei"].astype(float)
lay = ZN["layer"]
m = len(lay)
np.fill_diagonal(adj, np.nan)
fig = plt.figure(figsize=(5.4, 2.3))
gs = fig.add_gridspec(1, 3, wspace=0.75, left=0.06, right=0.97, top=0.86,
                      bottom=0.22)
starts = np.flatnonzero(np.r_[True, lay[1:] != lay[:-1]])
mids = [(s + e) / 2 for s, e in zip(starts, [*starts[1:], m])]
mats = [(adj, "adjusted NMI", seq_cmap(PINE, "pine"),
         matplotlib.colors.LogNorm(1e-3, 1), "adjusted NMI"),
        (ei, "EI (bits)", seq_cmap(INK, "ink"),
         matplotlib.colors.LogNorm(*np.nanpercentile(ei, [1, 99])),
         "effective information")]
axes = []
for j, (M, label, cmap, norm, title) in enumerate(mats):
    ax = fig.add_subplot(gs[0, j])
    cm = cmap.copy()
    cm.set_bad("#E6E6E6")
    im = ax.imshow(np.where(np.isfinite(M), np.maximum(M, norm.vmin), np.nan),
                   cmap=cm, norm=norm, interpolation="nearest")
    for s in starts[1:]:
        ax.axhline(s - 0.5, color=INK, lw=0.4)
        ax.axvline(s - 0.5, color=INK, lw=0.4)
    ax.set_xticks(mids, [f"L{L}" for L in range(len(mids))], fontsize=6.0)
    ax.set_yticks(mids, [f"L{L}" for L in range(len(mids))], fontsize=6.0)
    ax.tick_params(length=0)
    eyebrow(ax, title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    if j == 1:
        cb.set_ticks([3e-3, 1e-2, 2e-2])
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    cb.ax.tick_params(labelsize=5.6)
    axes.append(ax)
c = fig.add_subplot(gs[0, 2])
src = np.repeat(lay[:, None], m, 1)
ok = np.isfinite(ei)
for sel, col, label in [(ok & (src > 0), INKMUT, "from layers 1-3"),
                        (ok & (src == 0), LEAF, "from layer 0")]:
    c.scatter(np.maximum(adj[sel], 1e-4), ei[sel], s=1.0, lw=0, color=col,
              alpha=0.45, label=label, rasterized=True)
c.set_xscale("log")
c.set_yscale("log")
c.set_xlim(1e-4, 1)
c.set_box_aspect(1)
c.set_xlabel("adjusted NMI", labelpad=1)
c.set_ylabel("EI (bits)", labelpad=1)
eyebrow(c, "per head pair")
mono_axes(c)
c.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.0, color=col)
                  for col in (LEAF, INKMUT)],
         labels=["from layer 0", "from layers 1-3"], fontsize=5.8,
         handletextpad=0.1, loc="lower right", borderaxespad=0.2)
axes.append(c)
letters(axes)
save(fig, "fig-headnmi")


# selection.py over the s_Hm selection sweep: the held-out noise2self score of
# each layer's head partitions minus its run's unpartitioned heat-kernel
# smoother (filled; below 0 the partition helps), against a random partition
# with matched cell sizes, the same way (hollow, dashed)
SN = load_rows("selection_n2s.csv")
SN_STYLE = [("3e-05", r"$s_{\mathrm{KL}_m}{=}3{\times}10^{-5}$", LEAF),
            ("1e-06", r"$s_{\mathrm{KL}_m}{=}10^{-6}$", INK)]


def n2s(hue, layer, stat):
    """{n_sel: value} of one stat for one s_Hm family and layer."""
    return {int(r["n_sel"]): float(r["value"]) for r in SN
            if r["s_Hm"] == hue and int(r["layer"]) == layer
            and r["stat"] == stat}


fig, axs = plt.subplots(1, 5, figsize=(5.4, 2.1), sharey=True)
fig.subplots_adjust(wspace=0.12, bottom=0.36, top=0.86, left=0.12,
                    right=0.99)
for L, ax in enumerate(axs):
    for hue, _, col in SN_STYLE:
        base = n2s(hue, L, "unpartitioned")
        for stat, kw in [("partition", dict(ls="-", mfc=col)),
                         ("random", dict(ls=DASH, mfc="white"))]:
            v = n2s(hue, L, stat)
            xs = sorted(v)
            ax.plot(xs, [v[x] - base[x] for x in xs], color=col, lw=0.9,
                    marker="o", ms=2.6, mec=col, mew=0.6, **kw)
    ax.axhline(0, **NULL, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 32], ["1", "4", "32"])
    ax.xaxis.set_minor_locator(FixedLocator([2, 8, 16]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.75, 42)
    eyebrow(ax, f"layer {L}")
    mono_axes(ax)
axs[0].set_ylabel("score − unpartitioned")
axs[2].set_xlabel("entries selected per head")
handles = ([Line2D([], [], color=c, marker="o", ms=2.6, lw=0.9)
            for _, _, c in SN_STYLE]
           + [Line2D([], [], color=INKMUT, marker="o", ms=2.6, mfc="white",
                     lw=0.9, ls=DASH)])
fig.legend(handles, [s[1] for s in SN_STYLE] + ["random partition, same "
                                               "sizes"],
           loc="lower center", ncols=3, handlelength=1.6, fontsize=6.4,
           bbox_to_anchor=(0.55, 0.0))
letters(axs)
save(fig, "fig-selection")

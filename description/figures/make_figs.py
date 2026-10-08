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

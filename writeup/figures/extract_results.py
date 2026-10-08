"""Result tables for make_figs.py, read from the eval scripts' outputs.

Nothing here restores a checkpoint, so unlike extract_curves.py it runs on
CPU in seconds. Each family reads what an eval script already wrote under
data/out and keeps only what a figure plots:

  pareto      pareto.py's reconstruction-capacity points: the reference
              softmax model (resid_nc, its own pareto.py run) and the
              straight-through stack with every SAE (pareto_unified), each
              row tagged with the family it is drawn as
  ratefloor   the Gaussian reference (Eq. ratefloor) over a grid of rates,
              from the whitened covariance spectrum of the eval tail
              pareto.py scores on
  steer       steerembed.py's realization and collateral by direction and
              strength, from the draw that also scored the supervised
              directions (steerembed_sup; tab:steersup reads the same draw)
  headcoh     headcoh.py's per-head z_cos for the four groupings of the
              coherence table
  autointerp  autointerp.py's mean detection F1 per description mode, the
              harvest's mean firing rate, and the code's nominal density
  bilinspec   bilinspec.py's per-(step, layer) classifier geometry for the
              softmax and hard-code stacks
  modularity  headstruct.py --modularity's real and null modularity per
              layer or prefix block, from each run's summary.json
  assignmap   assignmap.py --sae's per-row assignments and SAE shares for
              a few label-informative heads, rows subsampled per label
              and put in the page's order (npz: heatmaps)
  entrydrift  entrydrift.py's per-layer drift and usage over steps, and
              two heads' entries step by step (csv and npz)
  enrich      enrich.py's null-subtracted head x label NMI, and the
              dotplot cells of one head
  headnmi     headnmi.py's adjusted NMI and the effective information it
              was paired with, in the script's head order (npz)
  selection   selection.py's noise2self scores against the selection
              rule, with their unpartitioned and random baselines

  uv run python writeup/figures/extract_results.py [pareto|ratefloor|...]
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = Path(__file__).resolve().parent / "data"
OUT = ROOT / "data" / "out"

PARETO = {
    "softmax": OUT / "sonar" / "pareto_resid_nc" / "pareto.csv",
    "unified": OUT / "sonar" / "pareto_unified" / "pareto.csv",
}
# pareto.py's eval split and objective weights, as its defaults set them
CACHE = ROOT / "data" / "sonar_embeddings" / "mc4_4M.npy"
WEIGHTS = OUT / "sonar" / "mse_weights.npy"
EVAL_ROWS = 32768

# (steerembed_sup run, model) for the panels of the steering figure
STEER = [
    ("sweep_softmax_shm", "softmax stack"),
    ("ste_h76_init01", "hard-code stack"),
    ("ste_l1_h380_i01_hm1e4", "hard-code flat"),
    ("g160top1", "g160top1"),
]

# (grouping, headcoh output dir) in the coherence table's row order
HEADCOH = [
    ("g160top1", OUT / "sonar" / "sae_conv" / "m5120_g160top1" / "headcoh"),
    ("k32 discovered",
     OUT / "sonar" / "sae_conv" / "m11264_k32" / "headcoh_discovered"),
    ("onto heads", OUT / "sonar" / "multilingual" / "resid_nc" / "headcoh"),
    ("g160softmax", OUT / "sonar" / "sae_conv" / "m5120_g160softmax" / "headcoh"),
]

# (autointerp run dir, label, nominal density = active coefficients / code
# width, trained by sae.py) in the description-specificity table's order
AUTOINTERP = [
    ("onto/onto", "onto", 1.0, False),
    ("m5120_k32/sae", "k32", 32 / 5120, True),
    ("m5120_k32_bl/sae", "k32_bl", 32 / 5120, True),
    ("m5120_g160top1/sae", "g160top1", 160 / 5120, True),
    ("m5120_g160softmax/sae", "g160softmax", 1.0, True),
    ("m11264_k5120/sae", "k5120", 5120 / 11264, True),
]


# (model, run) whose bilinspec.py output the classifier-geometry figure draws
BILINSPEC = [
    ("softmax stack", "resid_nc"),
    ("hard-code stack", "ste_h76_init01"),
]

# (model, headstruct.py --modularity output dir) for the modularity figure,
# as experiments/modularity/run.sh writes them
MODULARITY = [
    ("Ontologizer", OUT / "sonar" / "multilingual" / "bl_ctl_full"
     / "headstruct"),
    ("Matryoshka, trained heads", OUT / "sonar" / "sae_conv"
     / "m5120_g160top1_p5" / "headstruct_mod"),
    ("Matryoshka, discovered groups", OUT / "sonar" / "sae_conv"
     / "m5120_k32_p5" / "headstruct_mod"),
    ("flat, trained heads", OUT / "sonar" / "sae_conv" / "m5120_g160top1"
     / "headstruct_mod"),
    ("flat, discovered groups", OUT / "sonar" / "sae_conv" / "m5120_k32"
     / "headstruct_mod"),
]

# (panel, assignmap.py --sae output dir, layer) for the assignment figure,
# drawn by the commands in experiments/ste-arm/notes.md
ASSIGNMAP = [
    ("SONAR", OUT / "sonar" / "multilingual" / "ste_h76_init01" / "assignmap",
     0),
    ("GPT-2", OUT / "gpt2_l8" / "ste_h76" / "assignmap", 0),
]
ASSIGN_HEADS = 3   # the layer's heads with the most null-corrected label NMI
ASSIGN_ROWS = 24   # rows per label slice, evenly spaced through it
ASSIGN_COLS = 96   # SAE latents shown, by usage over the shown rows

# entrydrift.py --ckpt output (steps of one run), and the (layer, head)
# pairs drawn entry by entry
ENTRYDRIFT = (OUT / "sonar" / "multilingual" / "resid_nc" / "entrydrift",
              [(0, 0), (4, 0)])

# (substrate, enrich.py output dir, label variables kept) for the label
# figure; the dotplot is the first substrate's best head for its first
# variable at ENRICH_LAYER
ENRICH = [
    ("GPT-2", OUT / "gpt2_l8" / "ste_h76" / "enrich", ["tokclass", "pos"]),
    ("SONAR", OUT / "sonar" / "multilingual" / "resid_nc" / "enrich",
     ["lang"]),
]
ENRICH_LAYER = 0

# the run whose headnmi.py --ei output (and the effinfo.py output it read)
# the NMI figure draws
HEADNMI = OUT / "sonar" / "multilingual" / "ste_h76"

SELECTION = OUT / "sonar" / "selection" / "n2s.csv"


def read(path: Path) -> list:
    with open(path) as f:
        return list(csv.DictReader(f))


def write(name: str, header: list, rows: list) -> None:
    with open(DATA / name, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def pareto_family(label: str, source: str) -> tuple:
    """(family, kind) a pareto.py row is drawn as. `onto` rows belong to
    the model of the run that wrote them; SAE rows are named by run dir."""
    if label.startswith("onto"):
        family = "softmax" if source == "softmax" else "ste"
        kind = ("origin" if "origin" in label else "hard" if "argmax" in label
                else "soft" if "soft" in label else "dev")
        return family, kind
    run = label.split()[-1]
    if "_g160" in run:
        return "variant", run.split("_")[-1]
    if "_l1" in run:
        return "sae_l1", run
    if run.endswith("_bl"):
        return "sae_bilinear", run
    return "sae_topk", run


def extract_pareto() -> None:
    rows = []
    for source, path in PARETO.items():
        for r in read(path):
            family, kind = pareto_family(r["label"], source)
            # the softmax run's SAE-free table holds only its own rows
            if source == "softmax" and family != "softmax":
                continue
            nu = re.search(r"dev m=(\d+)", r["label"])
            rows.append([family, kind, nu.group(1) if nu else "",
                         r["label"], r["coeffs"], r["index_bits"],
                         f"{float(r['fvu_w']):.6g}"])
    write("pareto_sonar.csv",
          ["family", "kind", "nu", "label", "coeffs", "bits", "fvu"], rows)


def extract_ratefloor() -> None:
    from pareto import gaussian_reference, whitened_spectrum
    mm = np.load(CACHE, mmap_mode="r")
    X = np.asarray(mm[-EVAL_ROWS:], dtype=np.float64)
    lam = whitened_spectrum(X, np.load(WEIGHTS).astype(np.float64))
    bits = np.unique(np.concatenate([np.geomspace(1, 40000, 400),
                                     [800, 1900]]))
    fvu = gaussian_reference(lam, bits)
    write("ratefloor_sonar.csv", ["bits", "fvu"],
          [[f"{b:.6g}", f"{v:.6g}"] for b, v in zip(bits, fvu)])
    for b in (800, 1900):
        print(f"  Gaussian reference at {b} bits: FVU_w "
              f"{float(gaussian_reference(lam, np.array([b]))[0]):.4f}")


def extract_steer() -> None:
    rows = []
    for run, model in STEER:
        d = json.loads((OUT / "sonar" / "steerembed_sup" / run
                        / "summary.json").read_text())
        for key, v in d["results"].items():
            if "@" not in key:
                continue  # the native-step summaries are not a sweep
            kind, s = key.split("@")
            rows.append([model, run, kind, s, f"{v['hit']:.6g}",
                         f"{v['collateral']:.6g}", d["n_features"],
                         d["n_samples"]])
    write("steer_sweep.csv", ["model", "run", "kind", "strength", "hit",
                              "collateral", "n_targets", "n_rows"], rows)


def extract_headcoh() -> None:
    rows = []
    for grouping, path in HEADCOH:
        for r in read(path / "heads.csv"):
            rows.append([grouping, r["group"], r["size"],
                         f"{float(r['z_cos']):.6g}"])
    write("headcoh_z.csv", ["grouping", "group", "size", "z_cos"], rows)


def extract_autointerp() -> None:
    rows = []
    for run, label, density, by_sae in AUTOINTERP:
        path = OUT / "sonar" / "autointerp" / run
        recs = read(path / "scores.csv")
        # params decodes were redone at textfid.SONAR_NORM in a campaign
        # of their own; acts and cacts never touch the decoder
        recs = [r for r in recs if r["mode"] != "params"] + [
            r for r in read(OUT / "sonar" / "autointerp_scalefix" / run
                            / "scores.csv") if r["mode"] == "params"]
        freq = float(np.load(path / "features.npz")["freq"].mean())
        f1 = {m: float(np.mean([float(r["f1"]) for r in recs
                                if r["mode"] == m]))
              for m in ("acts", "cacts", "params")}
        n = len({r["feature"] for r in recs if r["mode"] == "acts"})
        rows.append([label, f"{density:.6g}", f"{freq:.6g}",
                     f"{f1['acts']:.6g}", f"{f1['cacts']:.6g}",
                     f"{f1['params']:.6g}", n, int(by_sae)])
    write("autointerp_density.csv", ["run", "density", "freq", "acts",
                                     "cacts", "params", "n_features",
                                     "sae_py"], rows)


def extract_bilinspec() -> None:
    keep = ("abscos_med", "abscos_q25", "abscos_q75", "frac_negdom",
            "rho_w_med", "odd_med")
    rows = []
    for model, run in BILINSPEC:
        path = OUT / "sonar" / "multilingual" / run / "bilinspec"
        for r in read(path / "bilinspec.csv"):
            rows.append([model, run, r["step"], r["layer"]]
                        + [f"{float(r[k]):.6g}" for k in keep])
    write("bilinspec.csv", ["model", "run", "step", "layer", *keep], rows)


def extract_modularity() -> None:
    rows = []
    for model, path in MODULARITY:
        d = json.loads((path / "summary.json").read_text())
        for r in d["table"]:
            if r["metric"] != "modularity":
                continue
            # a flat model has only the whole-model row; a layered one's
            # whole-model row averages over layers and is not drawn
            block = r.get("block", "")
            if block == "" and any("block" in t for t in d["table"]):
                continue
            rows.append([model, block, f"{r['real']:.6g}",
                         f"{r['null']:.6g}", f"{r['null_sd']:.6g}",
                         f"{r['z']:.6g}", r.get("groups", ""),
                         f"{r['coverage']:.6g}" if "coverage" in r else ""])
    write("modularity.csv", ["model", "block", "real", "null", "null_sd", "z",
                             "groups", "coverage"], rows)


def extract_assignmap() -> None:
    import assignmap
    from enrich import nmi_heads

    out = {}
    for i, (panel, path, L) in enumerate(ASSIGNMAP):
        z = np.load(path / "assign.npz")
        P = z["P"][:, L]  # (n, h, k)
        k = P.shape[-1]
        values, y = np.unique(z["labels"], return_inverse=True)
        st = nmi_heads(P.argmax(-1), y, k, len(values), nulls=20, seed=0)
        heads = np.argsort(-(st["nmi"] - st["null"]), kind="stable")
        heads = heads[:ASSIGN_HEADS]

        starts = np.cumsum([0, *z["slices"]])
        picks = [np.unique(np.linspace(a, b - 1, min(ASSIGN_ROWS, b - a))
                           .round().astype(int))
                 for a, b in zip(starts[:-1], starts[1:])]
        Q = P[np.concatenate(picks)][:, heads]
        order = assignmap.entry_order(Q)
        cut = np.cumsum([0, *map(len, picks)])
        perm = np.concatenate([a + assignmap.row_order(Q[a:b], order)
                               for a, b in zip(cut[:-1], cut[1:])])
        rows = np.concatenate(picks)[perm]
        Q = np.take_along_axis(P[rows][:, heads], order[None], -1)

        sae = json.loads(str(z["sae_meta"]))
        zs = z["sae_z"][rows]
        blk = assignmap.sae_block(zs, 0, ASSIGN_COLS, str(z["sae_run"]), 1.0)
        out |= {
            f"panel{i}": panel, f"run{i}": str(z["run"]),
            f"step{i}": int(z["step"]), f"layer{i}": L, f"by{i}": str(z["by"]),
            f"heads{i}": heads, f"H{i}": Q.reshape(len(rows), -1)
            .astype(np.float16),
            f"use{i}": assignmap.eff_entries(Q), f"row{i}": assignmap.row_eff(Q),
            f"slices{i}": np.asarray([len(p) for p in picks]),
            f"names{i}": np.asarray([str(s) for s in z["shown"]]),
            f"sae_run{i}": str(z["sae_run"]), f"S{i}": blk.Q[:, 0]
            .astype(np.float16),
            # whole-code use and row, the share of usage the shown columns
            # hold, the code's L0 on these rows, and its width and topk
            f"sae{i}": np.asarray([blk.eff[0], blk.reff[0], blk.mass[0],
                                   (zs > 0).sum(-1).mean(), sae["m"],
                                   sae["topk"]]),
        }
        print(f"assignmap {panel}: {len(rows)} rows, heads {heads.tolist()} "
              f"(excess NMI {np.round((st['nmi'] - st['null'])[heads], 3)})")
    np.savez_compressed(DATA / "assignmap.npz", **out)


def extract_entrydrift() -> None:
    path, pairs = ENTRYDRIFT
    z = np.load(path / "drift.npz")
    U, D = z["usage"], z["drift"]  # (columns, l, h, k)
    C, l, h, k = U.shape
    live = U >= 1 / (4 * k)  # entrydrift.py's dead-entry threshold
    ref = int(z["ref"])
    rows = []
    for L in range(l):
        used = live[ref, L]  # entries in use at the reference column
        for c in range(C):
            rows.append([L, int(z["steps"][c]), f"{z['temps'][c]:.6g}",
                         f"{np.median(D[c, L][used]):.6f}",
                         f"{live[c, L].mean():.6f}"])
    write("entrydrift.csv", ["layer", "step", "T", "median_drift_used",
                             "frac_used"], rows)
    heat = {}
    for i, (L, hd) in enumerate(pairs):
        order = np.argsort(-U[ref, L, hd], kind="stable")
        heat |= {f"usage{i}": U[:, L, hd][:, order].T.astype(np.float32),
                 f"drift{i}": D[:, L, hd][:, order].T.astype(np.float32),
                 f"dead{i}": ~live[ref, L, hd][order],
                 f"head{i}": np.asarray([L, hd])}
    np.savez_compressed(DATA / "entrydrift.npz", steps=z["steps"],
                        temps=z["temps"], mark_step=z["mark_step"], k=k,
                        **heat)


def extract_enrich() -> None:
    rows = []
    for sub, path, bys in ENRICH:
        for r in read(path / "nmi.csv"):
            if r["by"] in bys:
                rows.append([sub, r["by"], r["layer"], r["head"],
                             f"{float(r['nmi']) - float(r['null']):.6g}",
                             r["null_sd"], r["z"]])
    write("enrich_nmi.csv", ["substrate", "by", "layer", "head", "excess",
                             "null_sd", "z"], rows)
    sub, path, bys = ENRICH[0]
    best = max((r for r in rows if r[0] == sub and r[1] == bys[0]
                and int(r[2]) == ENRICH_LAYER), key=lambda r: float(r[4]))
    keep = ("entry", "label", "count", "n_entry", "expected", "log2OR",
            "neglog10_FDR")
    cells = [[r[c] for c in keep] for r in read(path / "cells.csv")
             if int(r["layer"]) == ENRICH_LAYER and r["head"] == best[3]]
    write("enrich_cells.csv", ["layer", "head", *keep],
          [[ENRICH_LAYER, best[3], *c] for c in cells])


def extract_headnmi() -> None:
    from headnmi import head_order, load_ei

    z = np.load(HEADNMI / "headnmi" / "headnmi.npz")
    l, h = int(z["l"]), int(z["h"])
    layer = np.repeat(np.arange(l), h)
    order = head_order(z["adj"], layer, z["dead"], "hclust")
    ei = load_ei(HEADNMI / "effinfo", "ei_live", l, h)
    np.savez_compressed(
        DATA / "headnmi.npz", run=str(z["run"]), step=int(z["step"]), l=l,
        h=h, adj=z["adj"][np.ix_(order, order)].astype(np.float16),
        ei=ei[np.ix_(order, order)].astype(np.float16), layer=layer[order],
        dead=z["dead"][order])


def extract_selection() -> None:
    keep = {"partition", "partition_p10", "partition_p90", "random",
            "random_sd", "unpartitioned"}
    rows = [[r["run"], r["x"], r["hue"], r["layer"], r["stat"], r["value"]]
            for r in read(SELECTION) if r["s"] == "1" and r["stat"] in keep]
    write("selection_n2s.csv", ["run", "n_sel", "s_Hm", "layer", "stat",
                                "value"], rows)


FAMILIES = {
    "pareto": extract_pareto,
    "ratefloor": extract_ratefloor,
    "steer": extract_steer,
    "headcoh": extract_headcoh,
    "autointerp": extract_autointerp,
    "bilinspec": extract_bilinspec,
    "modularity": extract_modularity,
    "assignmap": extract_assignmap,
    "entrydrift": extract_entrydrift,
    "enrich": extract_enrich,
    "headnmi": extract_headnmi,
    "selection": extract_selection,
}

if __name__ == "__main__":
    for family in sys.argv[1:] or list(FAMILIES):
        FAMILIES[family]()

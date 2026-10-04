"""Scaling-sweep analysis: does polysemanticity fall with parameter count?

Reads, for every config gen_configs.py wrote (or an explicit --configs
list): the run's loss.csv tail (converged training MSE and KL_m) and its
headcoh.py output (heads.csv / summary.json: per-head within-head
direction-coherence z-scores). The polysemanticity reading: a head whose
entry decode directions cohere (high z_cos) behaves like alternative
values of ONE dimension; a head with null-level coherence is a
polysemantic grab bag. The claim predicts mean z_cos and frac(z_cos > 2)
rise, and usage imbalance (KL_m) falls, as (h, k, l) grow the parameter
count.

Pure CPU + numpy: no checkpoint restore (headcoh.py already did the GPU
work in the queue). Writes results.csv and results.md into this
experiment dir.

  uv run python experiments/scaling-sweep/analyze.py
  uv run python experiments/scaling-sweep/analyze.py --tail 200
"""
import argparse
import csv
import json
import numpy as np
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]

# loss.csv columns (OntoState stats buffer; KL_m present on 9-col runs)
LOSS_COLS = ["loss", "MSE", "MSE_ghost", "L1_K", "L1_F", "entropy",
             "cossim_b", "cossim_h", "KL_m"]

RESULT_COLS = ["name", "h", "k", "l", "n_params", "n_tags", "steps",
               "train_mse_tail", "train_klm_tail", "n_heads_scored",
               "mean_z_cos", "median_z_cos", "frac_z_cos_gt2",
               "mean_z_sv", "frac_z_sv_gt2"]


def n_params(cfg):
    """Analytic parameter count of the Ontologizer a config builds.
    Assumes the live architecture family: bilinear classifier (n weight
    factors, unbiased), abs()'d DictBlock, unbiased Linear decoder, no
    encoder (encoded=False), forward='resid' so upper-layer classifier
    input is d_out + resid_const. Matches ~23M for the live center."""
    if cfg["encoded"] or cfg["scaled"]:
        raise SystemExit("n_params: encoded/scaled configs not supported "
                         "(the sweep never generates them)")
    d, e_dec, h, k, l, n = (cfg["d"], cfg["e_dec"], cfg["h"], cfg["k"],
                            cfg["l"], cfg["n"])
    if cfg["fwd_mode"] == "resid":
        d_next = d + (1 if cfg["resid_const"] else 0)
    else:
        d_next = h * k
    cl = n * h * k * d + (l - 1) * n * h * k * d_next
    return cl + l * h * k * e_dec + e_dec * d


def loss_tail(path, tail):
    """(steps on file, tail-mean MSE, tail-mean KL_m or nan)."""
    if not path.exists():
        return 0, float("nan"), float("nan")
    try:
        dat = np.loadtxt(path, delimiter=",", ndmin=2)
    except ValueError as e:
        print(f"warning: unparseable {path}: {e}")
        return 0, float("nan"), float("nan")
    t = dat[-min(tail, len(dat)):]
    mse = float(t[:, LOSS_COLS.index("MSE")].mean())
    klm = (float(t[:, LOSS_COLS.index("KL_m")].mean())
           if dat.shape[1] > LOSS_COLS.index("KL_m") else float("nan"))
    return len(dat), mse, klm


def headcoh_stats(run_dir):
    """Per-head z arrays + summary from a headcoh.py output dir."""
    hc = Path(run_dir) / "headcoh"
    heads = hc / "heads.csv"
    if not heads.exists():
        return None
    with open(heads) as f:
        rows = list(csv.DictReader(f))
    z_cos = np.array([float(r["z_cos"]) for r in rows])
    z_sv = np.array([float(r["z_sv"]) for r in rows])
    return {"n_heads_scored": len(rows),
            "mean_z_cos": float(z_cos.mean()),
            "median_z_cos": float(np.median(z_cos)),
            "frac_z_cos_gt2": float((z_cos > 2).mean()),
            "mean_z_sv": float(z_sv.mean()),
            "frac_z_sv_gt2": float((z_sv > 2).mean())}


def spearman(a, b):
    """Spearman rank correlation without scipy (ties get arbitrary but
    deterministic ranks; fine at these ns)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def collect(cfg_paths, tail):
    rows = []
    for cp in cfg_paths:
        cfg = json.loads(Path(cp).read_text())
        run = Path(cfg["out"])
        if not run.is_absolute() and not run.exists():
            run = ROOT / cfg["out"]  # robust to running outside repo root
        steps, mse, klm = loss_tail(run / "loss.csv", tail)
        hc = headcoh_stats(run)
        row = {"name": cfg["name"], "h": cfg["h"], "k": cfg["k"],
               "l": cfg["l"], "n_params": n_params(cfg),
               "n_tags": cfg["l"] * cfg["h"] * cfg["k"], "steps": steps,
               "train_mse_tail": mse, "train_klm_tail": klm}
        if hc is None:
            print(f"note: {run}/headcoh missing (queue not finished?); "
                  "headcoh columns stay NaN")
            hc = {c: float("nan") for c in RESULT_COLS[9:]}
            hc["n_heads_scored"] = 0
        rows.append({**row, **hc})
    return sorted(rows, key=lambda r: r["n_params"])


def write_results(rows, out_csv, out_md, tail):
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLS)
        w.writeheader()
        w.writerows(rows)

    scored = [r for r in rows if r["n_heads_scored"]]
    lp = [np.log(r["n_params"]) for r in scored]
    cors = {
        "spearman(log n_params, mean_z_cos)":
            spearman(lp, [r["mean_z_cos"] for r in scored]),
        "spearman(log n_params, frac_z_cos_gt2)":
            spearman(lp, [r["frac_z_cos_gt2"] for r in scored]),
        "spearman(log n_params, train_klm_tail)":
            spearman(lp, [r["train_klm_tail"] for r in scored]),
        "spearman(log n_params, train_mse_tail)":
            spearman(lp, [r["train_mse_tail"] for r in scored]),
    }

    md = ["# scaling-sweep results", "",
          f"Tail window: last {tail} loss.csv rows. Coherence stats from "
          "each run's `headcoh/heads.csv` (z vs size-matched within-layer "
          "nulls).", "",
          "| " + " | ".join(RESULT_COLS) + " |",
          "|" + "---|" * len(RESULT_COLS)]
    for r in rows:
        md.append("| " + " | ".join(
            f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])
            for c in RESULT_COLS) + " |")
    md += ["", "## Claim test (higher coherence = less polysemantic)", ""]
    for k_, v in cors.items():
        md.append(f"- {k_} = {v:+.3f}")
    md += ["",
           "The claim predicts positive correlations for the z columns and",
           "a negative one for KL_m. Caveats: (i) the loss.csv MSE column",
           "is the mean over l prefix reconstructions (deepsup), so it is",
           "not comparable across different l; run `pareto.py --ckpt",
           "<run>` for a final-reconstruction FVU if that axis matters.",
           "(ii) z-scores are against size-k nulls drawn from each model's",
           "OWN pool, so they compare head structure, not absolute",
           "reconstruction quality. (iii) with 7 star points, treat the",
           "correlations as directional, not inferential; per-axis reads",
           "(h rows, k rows, l rows separately) are in the table."]
    Path(out_md).write_text("\n".join(md) + "\n")

    for k_, v in cors.items():
        print(f"{k_} = {v:+.3f}")
    print(f"-> {out_csv}\n-> {out_md}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="*", default=None,
                   help="config JSONs (default: configs/*.json here)")
    p.add_argument("--tail", type=int, default=500,
                   help="loss.csv rows averaged for the converged stats")
    args = p.parse_args()

    cfgs = args.configs or sorted((EXP / "configs").glob("*.json"))
    if not cfgs:
        raise SystemExit("no configs found; run gen_configs.py first")
    rows = collect(cfgs, args.tail)
    write_results(rows, EXP / "results.csv", EXP / "results.md", args.tail)


if __name__ == "__main__":
    main()

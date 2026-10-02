"""Does autointerp F1 track code density rather than architecture?

Two tests, both from artifacts already on disk:

  across runs   mean firing rate (and L0 from the trainer's own loss.csv)
                against mean detection F1 over the six campaign runs
  within runs   per-feature Spearman between firing rate and F1, which the
                harvest/scores join supports directly -- the sharper test,
                since it holds architecture fixed

`freq` is written by `autointerp.harvest` as the fraction of subsampled
rows where the feature is above its method's "fires" threshold, so it is
the same quantity for Ontologizer tags and SAE latents.
"""
import csv
from pathlib import Path

import numpy as np

A = Path("/home/keira/flock/ontologize/data/out/sonar/autointerp")
SAE = Path("/home/keira/flock/ontologize/data/out/sonar/sae")


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def l0_of(run):
    """Final eval L0 from the trainer's loss.csv, where available."""
    p = SAE / run / "loss.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    for key in ("l0_eval", "l0"):
        if rows and key in rows[0]:
            vals = [float(r[key]) for r in rows[-50:] if r[key]]
            return sum(vals) / len(vals) if vals else None
    return None


runs = {}
for p in sorted(A.glob("*/*/scores.csv")):
    run = p.parts[-3]
    recs = list(csv.DictReader(open(p)))
    npz = np.load(p.parent / "features.npz")
    sel, freq = npz["sel"], npz["freq"]
    fmap = dict(zip(sel.tolist(), freq.tolist()))
    runs[run] = {"recs": recs, "fmap": fmap, "m": len(sel)}

print(f"{'run':>20}{'mean freq':>11}{'median':>9}{'L0':>8}"
      f"{'acts F1':>9}{'cacts F1':>10}{'rho(freq,F1)':>14}")
print("-" * 81)
agg = []
for run, d in runs.items():
    f = np.array(list(d["fmap"].values()))
    out = {}
    for mode in ("acts", "cacts"):
        v = [(d["fmap"].get(int(r["feature"])), float(r["f1"]))
             for r in d["recs"] if r["mode"] == mode]
        v = [(a, b) for a, b in v if a is not None]
        out[mode] = (np.mean([b for _, b in v]),
                     spearman([a for a, _ in v], [b for _, b in v]))
    l0 = l0_of(run)
    print(f"{run:>20}{f.mean():>11.4f}{np.median(f):>9.4f}"
          f"{(f'{l0:.0f}' if l0 else '-'):>8}"
          f"{out['acts'][0]:>9.3f}{out['cacts'][0]:>10.3f}"
          f"{out['cacts'][1]:>+14.3f}")
    agg.append((run, f.mean(), out["acts"][0], out["cacts"][0]))

print("\nacross the six runs:")
for i, lab in ((2, "acts"), (3, "cacts")):
    x = [a[1] for a in agg]
    y = [a[i] for a in agg]
    print(f"  Spearman(mean firing rate, {lab} F1) = {spearman(x, y):+.3f}"
          f"   Pearson = {np.corrcoef(x, y)[0, 1]:+.3f}  (n=6)")
print("\nwithin runs, rho(freq, cacts F1) is the last column above: if "
      "density drove\ndescribability, it would be consistently negative "
      "there too.")

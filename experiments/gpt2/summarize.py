"""Tabulate GPT-2 experiment results: Ontologizer runs (diag_<step>.json)
and SAE runs (eval_gpt2.json) under data/out/gpt2.

  uv run python experiments/gpt2/summarize.py [name-glob ...]
"""
import fnmatch
import json
import sys
from pathlib import Path

ROOT = Path("data/out/gpt2")


def onto_rows(pats):
    for run in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name not in ("sae", "logs")):
        if pats and not any(fnmatch.fnmatch(run.name, p) for p in pats):
            continue
        diags = sorted(run.glob("diag_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        if not diags:
            continue
        s = json.loads(diags[-1].read_text())
        cfg = json.loads((run / "config.json").read_text())
        ce = s.get("ce", {})
        L = s["layers"]
        yield {"run": run.name, "step": s["step"], "lr": cfg["lr"],
               "fvu": s["fvu_soft"], "fvu_hard": s["fvu_hard"],
               "rec": ce.get("soft_recovered"), "rec_hard": ce.get("hard_recovered"),
               "dCE": (ce["soft"] - ce["clean"]) if ce else None,
               "H": [round(r["entropy"], 2) for r in L],
               "fvu_prefix": [round(f, 3) for f in s["fvu_prefix"]],
               "corr_nc": [round(r["corr_nc"], 2) for r in L]}


def sae_rows(pats):
    for f in sorted((ROOT / "sae").glob("*/eval_gpt2.json")):
        name = "sae/" + f.parent.name
        if pats and not any(fnmatch.fnmatch(name, p) for p in pats):
            continue
        s = json.loads(f.read_text())
        ce = s["ce"]
        yield {"run": name, "fvu": s["fvu"], "l0": s["l0"],
               "rec": ce["sae_recovered"], "dCE": ce["sae"] - ce["clean"]}


def fmt(x, n=4):
    return "   -   " if x is None else f"{x:.{n}f}"


def main():
    pats = sys.argv[1:]
    print(f"{'run':28s} {'step':>6} {'lr':>7} {'FVU':>7} {'FVUhard':>8} {'CErec':>7} "
          f"{'CErecH':>7} {'dCE':>7}  entropy/layer (max 5)   prefix FVU")
    for r in onto_rows(pats):
        print(f"{r['run']:28s} {r['step']:6d} {r['lr']:7.0e} {fmt(r['fvu'])} {fmt(r['fvu_hard']):>8} "
              f"{fmt(r['rec'])} {fmt(r['rec_hard'])} {fmt(r['dCE'])}  {r['H']}  {r['fvu_prefix']}")
    print()
    print(f"{'run':28s} {'L0':>7} {'FVU':>7} {'CErec':>7} {'dCE':>7}")
    for r in sae_rows(pats):
        print(f"{r['run']:28s} {r['l0']:7.1f} {fmt(r['fvu'])} {fmt(r['rec'])} {fmt(r['dCE'])}")


if __name__ == "__main__":
    main()

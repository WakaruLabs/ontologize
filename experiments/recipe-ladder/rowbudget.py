"""SAE codes held to a fixed per-row budget, read exactly and taken apart.

quantrate.py holds each SAE to a grid of per-row budgets (rowbudget.csv),
keeps only the FVU at each, and interpolates between grid points for its
readouts. This recomputes quantrate.sae_rows's per-row lengths and errors
(one entry per --steps value and row) and reads them at exact budgets,
four ways:

  rate control      each row at its least-error step whose code fits the
                    budget once log2(steps) bits name the step, a row no
                    step fits taking its shortest code (quantrate's
                    rate_control: the reading rowbudget.csv samples)
  longest fit       each row at its longest code that fits, with no search
                    for error: what the budget costs before the search
  one step, mean    every row at the one step whose mean length is nearest
                    the budget: the mean-rate reading, one quantization
                    for every row
  one step, cap     every row at one step, the least-error step at which
                    every row's code fits the budget: one quantization and
                    a hard cap

The two one-step readings name no step, since every row shares it, so no
side bits count against their budget. Each reading also gives the bits
the rows use, and rate control how many rows it moves off their longest
code and how many of those its search improves.

For each --targets FVU it also finds the least per-row budget at which
rate control reaches it. That FVU cannot rise with the budget: a larger
budget only adds steps a row may take, and a row's first step to fit is
its shortest code, which is what it took before. So the least budget is
the smallest breakpoint L[s, n] + side that meets the target, and a
bisection over the sorted breakpoints finds it exactly.

  JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest \\
      uv run python experiments/recipe-ladder/rowbudget.py \\
      --runs data/out/sonar/sae_sent16/<run> ... --targets 0.16 0.17 \\
      --out data/out/sonar/<dir>/rowbudget_exact.json

The defaults score mc4_sent16's tail as the ladder's quantrate.py runs do.
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from jaxtyping import Float, Int

import quantrate as q


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="sae.py run directories, or their params.npz")
    p.add_argument("--budgets", type=float, nargs="+", default=[1900.0],
                   help="per-row budgets, in bits, to read each code at")
    p.add_argument("--targets", type=float, nargs="*", default=[],
                   help="FVUs to find rate control's least per-row budget for")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_sent16.npy")
    p.add_argument("--mse-weights",
                   default="data/out/sonar/sentcache/mse_weights_no837.npy")
    p.add_argument("--eval-rows", type=int, default=32765,
                   help="rows scored, from the tail of --cache")
    p.add_argument("--b", type=int, default=6553)
    p.add_argument("--out", default=None, help="JSON file for the readings")
    return p.parse_args(argv)


# ---------- pure helpers ----------

def row_fvu(e: Float[np.ndarray, "n"], dims: int, base_w: float) -> float:
    """Whitened FVU from per-row whitened squared errors, summed over
    coordinates, as rowbudget.csv computes it."""
    return float(e.mean() / dims / base_w)


def picked(E: Float[np.ndarray, "s n"], pick: Int[np.ndarray, "n"]
           ) -> Float[np.ndarray, "n"]:
    """Each row's entry at its own step."""
    return E[pick, np.arange(E.shape[1])]


def longest_fit(L: Float[np.ndarray, "s n"], budget: float, side: float
                ) -> Int[np.ndarray, "n"]:
    """Per row, the step with the longest code that fits `budget` with
    `side` bits naming it; a row no step fits takes its shortest code, as
    under rate_control."""
    fits = L + side <= budget
    longest = np.where(fits, L, -np.inf).argmax(0)
    return np.where(fits.any(0), longest, L.argmin(0))


def nearest_mean_step(L: Float[np.ndarray, "s n"], budget: float) -> int:
    """The step whose mean code length over rows is nearest `budget`."""
    return int(np.abs(L.mean(1) - budget).argmin())


def cap_step(L: Float[np.ndarray, "s n"], E: Float[np.ndarray, "s n"],
             budget: float) -> Optional[int]:
    """Of the steps at which every row's code fits `budget`, the one with
    the least mean error; None if no step fits every row."""
    ok = np.where(L.max(1) <= budget)[0]
    return int(ok[E[ok].mean(1).argmin()]) if len(ok) else None


def least_budget(L: Float[np.ndarray, "s n"], E: Float[np.ndarray, "s n"],
                 target: float, side: float, dims: int, base_w: float
                 ) -> Optional[float]:
    """The least per-row budget at which rate control's FVU is at most
    `target`, or None if no budget reaches it. Rate control's FVU does not
    rise with the budget (module docstring), so this bisects over the
    sorted breakpoints L + side, where alone it can change."""
    cands = np.unique(L + side)

    def ok(i: int) -> bool:
        pick = q.rate_control(L, E, float(cands[i]), side)
        return row_fvu(picked(E, pick), dims, base_w) <= target

    if not ok(len(cands) - 1):
        return None
    lo, hi = 0, len(cands) - 1      # ok(hi) holds throughout
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return float(cands[hi])


def read(L: Float[np.ndarray, "s n"], E: Float[np.ndarray, "s n"],
         budget: float, side: float, dims: int, base_w: float,
         settings: Sequence[str]) -> dict:
    """The four readings at one per-row budget (module docstring)."""
    fvu = lambda pick: row_fvu(picked(E, pick), dims, base_w)
    idx = np.arange(L.shape[1])
    rc = q.rate_control(L, E, budget, side)
    lf = longest_fit(L, budget, side)
    used = L[rc, idx] + side
    g = nearest_mean_step(L, budget)
    lg = L[g]
    p05, p50, p95 = np.percentile(lg, [5, 50, 95])
    c = cap_step(L, E, budget)
    return {
        "budget": float(budget),
        "rate_control": {
            "fvu": fvu(rc), "bits_mean": float(used.mean()),
            "over_share": float((used > budget).mean()),
            "off_longest_share": float((rc != lf).mean()),
            "improved_share": float((picked(E, lf) > picked(E, rc)).mean())},
        "longest_fit": {
            "fvu": fvu(lf),
            "bits_mean": float((L[lf, idx] + side).mean())},
        "one_step_mean": {
            "step": settings[g], "fvu": row_fvu(E[g], dims, base_w),
            "bits_mean": float(lg.mean()), "bits_p05": float(p05),
            "bits_p50": float(p50), "bits_p95": float(p95),
            "bits_max": float(lg.max()),
            "over_share": float((lg > budget).mean())},
        "one_step_cap": None if c is None else {
            "step": settings[c], "fvu": row_fvu(E[c], dims, base_w),
            "bits_mean": float(L[c].mean()), "bits_max": float(L[c].max())},
    }


# ---------- main ----------

def score(path, cfg, qcfg, fit_batches, X, w, base_w) -> dict:
    rows, pr = q.sae_rows(path, qcfg, fit_batches, X, w, base_w)
    L, E = pr["lengths"], pr["errs"]
    dims, bw = pr["dims"], pr["base_w"]
    side = math.log2(len(L))
    per_step = [row_fvu(E[s], dims, bw) for s in range(len(L))]
    best = int(np.argmin(per_step))
    free = E.argmin(0)
    return {
        "label": pr["label"], "steps": len(L), "side_bits": side,
        "rows": int(L.shape[1]),
        "float_fvu": [r for r in rows if r["setting"] == "float"][0]["fvu_w"],
        "best_one_step": {"step": pr["settings"][best],
                          "fvu": per_step[best],
                          "bits_mean": float(L[best].mean())},
        "best_step_per_row": {"fvu": row_fvu(picked(E, free), dims, bw),
                              "bits_mean": float(picked(L, free).mean())},
        "budgets": [read(L, E, b, side, dims, bw, pr["settings"])
                    for b in cfg.budgets],
        "least_budget": {f"{t:g}": least_budget(L, E, t, side, dims, bw)
                         for t in cfg.targets},
    }


def report(r: dict) -> None:
    print(f"\n== {r['label']}: {r['steps']} steps "
          f"({r['side_bits']:.2f} side bits), {r['rows']} rows")
    b1, bp = r["best_one_step"], r["best_step_per_row"]
    print(f"float FVU {r['float_fvu']:.5f}; best one step {b1['fvu']:.5f} "
          f"({b1['bits_mean']:.0f} bits); each row's best step, no budget, "
          f"{bp['fvu']:.5f} ({bp['bits_mean']:.0f} bits)")
    for b in r["budgets"]:
        rc, lf, om, oc = (b["rate_control"], b["longest_fit"],
                          b["one_step_mean"], b["one_step_cap"])
        print(f"at {b['budget']:.0f} bits per row:")
        print(f"  rate control     {rc['fvu']:.5f}  {rc['bits_mean']:.0f} bits used, "
              f"{rc['over_share']:.2%} over; {rc['off_longest_share']:.1%} of rows "
              f"off their longest code, {rc['improved_share']:.1%} improved")
        print(f"  longest fit      {lf['fvu']:.5f}  {lf['bits_mean']:.0f} bits used")
        print(f"  one step, mean   {om['fvu']:.5f}  mean {om['bits_mean']:.0f}, "
              f"p05 {om['bits_p05']:.0f}, p50 {om['bits_p50']:.0f}, "
              f"p95 {om['bits_p95']:.0f}, max {om['bits_max']:.0f}; "
              f"{om['over_share']:.1%} of rows over")
        print("  one step, cap    " + ("none fits every row" if oc is None else
              f"{oc['fvu']:.5f}  mean {oc['bits_mean']:.0f}, max {oc['bits_max']:.0f}"))
    for t, v in r["least_budget"].items():
        print(f"least budget to FVU {t}: "
              + ("not reached" if v is None else f"{v:.1f} bits"))
    sys.stdout.flush()


def main() -> None:
    cfg = parse_args()
    qcfg = q.parse_args(["--cache", cfg.cache, "--mse-weights", cfg.mse_weights,
                         "--eval-rows", str(cfg.eval_rows), "--b", str(cfg.b)])
    X, w, base_w = q.eval_rows(qcfg)
    fit_batches = q.fit_source(qcfg, X.shape[1])
    out = []
    for run in cfg.runs:
        path = Path(run) / "params.npz" if Path(run).is_dir() else Path(run)
        meta = path.parent / "meta.json"
        if meta.exists() and json.loads(meta.read_text()).get("block"):
            print(f"skip {path} (block-sparse code: not implemented)", flush=True)
            continue
        out.append(score(path, cfg, qcfg, fit_batches, X, w, base_w))
        report(out[-1])
    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out).write_text(json.dumps(
            {"settings": vars(cfg), "codes": out}, indent=2))
        print(f"\n-> {cfg.out}")


if __name__ == "__main__":
    main()

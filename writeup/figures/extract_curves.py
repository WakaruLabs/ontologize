"""Training curves for make_figs.py, scored from checkpoints.

The logged `loss.csv` MSE cannot be plotted as FVU. Under deep supervision
it averages the reconstruction error of every prefix, so a five-layer
stack logs roughly twice its final-output error and plots above a flat
arm it beats held out; it is also taken under classifier noise and
dropout. This restores each checkpoint and scores the quantity the tables
report instead: whitened FVU of the final output on the eval tail, with
`sae.py`'s split and constant-predictor base (the GPT-2 tail snapped to
whole documents by `doc_holdout`). The classification temperature is the
run's own schedule at that step, so a softmax checkpoint taken during the
anneal is scored as it trained. The softmax sweep predates the holdout,
so its scores are in-sample.

Each checkpoint also gives `c`, the mean within-head row cosine
`dictgeom.py` reads from the weights; the sweep's logged `cossim_k`
predates the current stats row and is not that statistic.

Resumable: rows already in an output CSV are skipped. Needs a GPU for the
sharded restores; a small memory fraction leaves a training run alone.

  XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 uv run python \\
      writeup/figures/extract_curves.py [orthant|sweep|depth ...]
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dictgeom import geometry
from ontologize.data.loaders import doc_holdout
from ontologize.training.ontostate import schedules
from pareto import load_onto

DATA = Path(__file__).resolve().parent / "data"
SONAR = ROOT / "data" / "out" / "sonar" / "multilingual"
GPT2 = ROOT / "data" / "out" / "gpt2_l8"
CACHE = {"sonar": ROOT / "data" / "sonar_embeddings" / "mc4_4M.npy",
         "gpt2": ROOT / "data" / "activations" / "gpt2_l8.npy"}
HOLDOUT = 32768

FAMILIES = {
    # tab:orthant: non-negativity x dictionary width, GPT-2 layer 8
    "orthant": ("gpt2", [
        ("ste_h76_sgn", "signed, e=1536"),
        ("ste_h76_sgn768", "signed, e=768"),
        ("ste_h76", "abs, e=1536"),
        ("ste_h76_e768", "abs, e=768"),
        ("ste_h76_cat32", "concat, d_head=32"),
    ]),
    # the s_Hm selection sweep: hard arms keep decorrelating after FVU flattens
    "sweep": ("sonar", [
        ("sweep_top2_shm", "top2"),
        ("sweep_top4_shm", "top4"),
        ("sweep_top8_shm", "top8"),
        ("sweep_top16_shm", "top16"),
        ("sweep_softmax_shm", "softmax"),
    ]),
    # depth vs. flat at the fixed initialization, with seed replicas
    "depth": ("sonar", [
        ("ste_h76_init01", "5x76"),
        ("ste_h76_i01_s43", "5x76, seed 43"),
        ("ste_l1_h380_i01_hm1e4", "1x380, s_Hm 1e-4"),
        ("ste_l1_h380_i01", "1x380"),
        ("ste_l1_h380_i01_s43", "1x380, seed 43"),
    ]),
}


def hyper(run: Path) -> dict:
    """The run's recorded hyperparameters: the last `env_config` in its log."""
    last = None
    with open(run / "log.jsonl") as f:
        for line in f:
            if '"env_config"' in line:
                last = line
    return json.loads(last)["hyper"]


def local(path: str) -> Path:
    """A path recorded on the training host, re-rooted at this checkout."""
    p = Path(path)
    if p.exists():
        return p
    parts = p.parts
    return ROOT.joinpath(*parts[parts.index("data"):])


def checkpoints(run: Path) -> list:
    """Every 10k checkpoint plus the last; the dense run-out at the end of a
    log is one point, not ten."""
    steps = sorted(int(d.name) for d in run.iterdir() if d.name.isdigit())
    return sorted({s for s in steps if s % 10000 == 0} | {steps[-1]})


def temperature(h: dict, step: int) -> float:
    return schedules(step, h.get("anneal_steps", 0), h["temperature"],
                     h.get("temperature_end"), h.get("p_drop", 0.0),
                     h.get("p_drop_start"), h.get("sd_K", 0.0),
                     h.get("sd_K_end"))[0]


def eval_split(substrate: str):
    mm = np.load(CACHE[substrate], mmap_mode="r")
    n = doc_holdout(CACHE[substrate], HOLDOUT)
    return np.asarray(mm[len(mm) - n:], dtype=np.float32)


def fvu(model, params, T: float, X_eval: Float[np.ndarray, "n d"],
        w: Float[np.ndarray, "d"], base_w: float, b: int = 1024) -> float:
    """Whitened FVU of the final output, `pareto.py`'s convention."""
    run = jax.jit(lambda X: model.apply({"params": params}, X,
                                        temperature=T))
    err = np.zeros(X_eval.shape[1])
    nb = 0
    for i in range(0, len(X_eval) - b + 1, b):
        X = jnp.asarray(X_eval[i:i + b])
        err += np.asarray(((run(X) - X) ** 2).mean(0))
        nb += 1
    return float(((err / nb) * w).mean() / base_w)


def done(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path) as f:
        return {(r["arm"], int(r["step"])) for r in csv.DictReader(f)}


def extract(family: str) -> None:
    substrate, arms = FAMILIES[family]
    root = GPT2 if substrate == "gpt2" else SONAR
    out = DATA / f"curves_{family}.csv"
    seen = done(out)
    X_eval = eval_split(substrate)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["arm", "label", "step", "fvu", "c", "T"])
        for arm, label in arms:
            run = root / arm
            h = hyper(run)
            w = np.load(local(h["mse_weights"])).astype(np.float64)
            base_w = float((X_eval.var(0) * w).mean())
            for step in checkpoints(run):
                if (arm, step) in seen:
                    continue
                model, params, step = load_onto(run, step)
                T = temperature(h, step)
                c = np.mean([geometry(np.asarray(
                    params[f"dictencs_{i}"]["dict"]["weights"]),
                    getattr(model, "signed", False))[0]
                    for i in range(model.l)])
                v = fvu(model, params, T, X_eval, w, base_w)
                wr.writerow([arm, label, step, f"{v:.6f}", f"{c:.6f}",
                             f"{T:.6g}"])
                f.flush()
                print(f"{family} {arm} {step} fvu={v:.4f} c={c:+.4f} T={T:.3g}",
                      flush=True)
                jax.clear_caches()


if __name__ == "__main__":
    for family in sys.argv[1:] or list(FAMILIES):
        extract(family)

"""Which part of the stack's structure keeps its heads from reproducing?

A single k-means partition of mc4_sent16 reproduces across seeds
(kmeans_seeds.py: NMI 0.61 at k = 32) while the stack's matched layer-0
heads reach 0.025. The stack differs from one k-means in two ways: its
layers are a residual cascade, and each layer sums many heads. This
separates the two with k-means alone, in the loss's whitened frame.

- `--mode rvq`: residual k-means, one head per stage. Stage s is a k-means
  of what stages 0..s-1 leave. Every seed runs the whole cascade
  (`independent`), and again with stage 0 taken from the first seed
  (`shared`), so its later stages see the same inputs as the first seed's.
  Per stage, the NMI between two seeds' tail assignments.
- `--mode aq`: additive quantization, h codebooks summed at one stage, as a
  layer of the stack sums its heads, fit by FAISS's LSQ++ (Martinez et al.),
  a strong solver for it; the simple block descent this replaced mixed even
  two planted factors. Between two runs, heads are matched one-to-one by NMI
  (Hungarian, as partition.py matches the stack's heads), beside a null with
  one run's rows shuffled. On planted factors LSQ's wrong answers have far
  higher error than its right ones, so the runs are read against their
  error: agreement among the lowest-error runs, and its rank correlation
  with the pair's worse error. FAISS is not a project dependency:
  `uv run --with faiss-cpu`.

`--gauss` repeats either on the Gaussian copy (gausscopy.py).

    JAX_PLATFORMS=cpu uv run --with faiss-cpu python experiments/seed-stability/kmeans_structure.py --mode aq --gauss data/out/sonar/gausscopy
    JAX_PLATFORMS=cpu uv run python experiments/seed-stability/kmeans_structure.py --mode rvq
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))
sys.path.insert(0, str(HERE))

from kmeans_seeds import assign, kmeans  # noqa: E402


def total_ss(X: Float[np.ndarray, "n d"]) -> float:
    X64 = X.astype(np.float64)
    return float(((X64 - X64.mean(0)) ** 2).sum())


# ---------- residual k-means ----------

def rvq_fit(X: Float[np.ndarray, "n d"], k: int, stages: int, seed: int,
            first: Float[np.ndarray, "k d"] | None = None, iters: int = 100
            ) -> list:
    """Codebooks of a residual k-means cascade; `first` fixes stage 0."""
    R = X.copy()
    Cs = []
    for s in range(stages):
        C = first if (s == 0 and first is not None) else \
            kmeans(R, k, seed * 1000 + s, iters)
        R = R - C[assign(R, C)]
        Cs.append(C)
    return Cs


def rvq_assign(X: Float[np.ndarray, "n d"], Cs: list
               ) -> tuple[Int[np.ndarray, "s n"], list]:
    """Each stage's cell for every row, and the FVU after each prefix."""
    R = X.copy()
    tot = total_ss(X)
    labs, fv = [], []
    for C in Cs:
        lab = assign(R, C)
        R = R - C[lab]
        labs.append(lab)
        fv.append(float((R.astype(np.float64) ** 2).sum() / tot))
    return np.stack(labs), fv


# ---------- additive quantization (FAISS's LSQ) ----------

def unpack(codes: Int[np.ndarray, "n b"], heads: int, nbits: int
           ) -> Int[np.ndarray, "heads n"]:
    """FAISS's packed codes as one entry per head: `heads` codes of `nbits`
    bits each, packed least significant bit first."""
    bits = np.unpackbits(np.asarray(codes, np.uint8), axis=1,
                         bitorder="little")[:, :heads * nbits]
    w = 1 << np.arange(nbits)
    return (bits.reshape(len(codes), heads, nbits) * w).sum(-1).T


def lsq_run(train: Float[np.ndarray, "n d"], tail: Float[np.ndarray, "m d"],
            heads: int, nbits: int, seed: int) -> dict:
    """One additive quantizer of `heads` codebooks of 2**nbits entries
    (LSQ++: alternating codebook least squares and ICM encoding with
    perturbations), its FVU on the training rows and the tail, and the
    tail's entry per head."""
    import faiss
    q = faiss.LocalSearchQuantizer(train.shape[1], heads, nbits)
    q.random_seed = seed
    q.train(np.ascontiguousarray(train))
    out = {}
    for name, X in (("train", train), ("tail", tail)):
        X = np.ascontiguousarray(X)
        c = q.compute_codes(X)
        out[f"{name}_fvu"] = float(((X - q.decode(c)).astype(np.float64) ** 2).sum()
                                   / total_ss(X))
        if name == "tail":
            out["labels"] = unpack(c, heads, nbits)
    return out


# ---------- agreement ----------

def matched_nmi(A: Int[np.ndarray, "h n"], B: Int[np.ndarray, "h n"], k: int
                ) -> tuple[float, list]:
    """Heads of A matched one-to-one to heads of B by NMI (Hungarian); the
    mean matched NMI and the matched values."""
    from scipy.optimize import linear_sum_assignment
    from partition import nmi_matrix
    M = nmi_matrix(A, k, B, k)
    r, c = linear_sum_assignment(-M)
    v = M[r, c]
    return float(v.mean()), [float(x) for x in v]


def stage_nmi(A: Int[np.ndarray, "s n"], B: Int[np.ndarray, "s n"], k: int
              ) -> list:
    """NMI between the same stage of two cascades."""
    from partition import nmi_matrix
    return [float(nmi_matrix(A[s][None], k, B[s][None], k)[0, 0])
            for s in range(len(A))]


# ---------- runs ----------

def run_rvq(train, tail, cfg) -> dict:
    out = {}
    base = rvq_fit(train, cfg.k, cfg.stages, cfg.seeds[0], iters=cfg.iters)
    for kind in ("independent", "shared"):
        fits = {cfg.seeds[0]: base}
        for s in cfg.seeds[1:]:
            fits[s] = rvq_fit(train, cfg.k, cfg.stages, s,
                              first=base[0] if kind == "shared" else None,
                              iters=cfg.iters)
        codes = {s: rvq_assign(tail, fits[s]) for s in cfg.seeds}
        pairs = list(itertools.combinations(cfg.seeds, 2))
        if kind == "shared":
            pairs = [p for p in pairs if p[0] == cfg.seeds[0]]
        M = np.array([stage_nmi(codes[a][0], codes[b][0], cfg.k) for a, b in pairs])
        out[kind] = {"pairs": [list(p) for p in pairs], "nmi": M.tolist(),
                     "nmi_mean": M.mean(0).tolist(),
                     "fvu": {str(s): codes[s][1] for s in cfg.seeds}}
        print(f"  rvq {kind:<11} NMI by stage "
              f"{np.round(M.mean(0), 4).tolist()}  prefix FVU "
              f"{np.round(np.mean([codes[s][1] for s in cfg.seeds], 0), 4).tolist()}",
              flush=True)
    return out


def run_aq(train, tail, cfg) -> dict:
    """`--restarts` runs per number of heads. Every pair's matched head NMI,
    and the same among the `--best` runs with the lowest training FVU: if
    restarts land in optima of different quality and the good ones agree,
    disagreement is the search's; if they reach the same error and still
    disagree, the optimum is not unique."""
    nbits = int(np.log2(cfg.k))
    assert 1 << nbits == cfg.k, "--k must be a power of two for LSQ"
    out = {}
    rng = np.random.default_rng(0)
    for h in cfg.hs:
        runs = [lsq_run(train, tail, h, nbits, s) for s in range(cfg.restarts)]
        tr = np.array([r["train_fvu"] for r in runs])
        best = set(np.argsort(tr)[:cfg.best].tolist())
        pairs, vals, nulls, inbest, worse = [], [], [], [], []
        for a, b in itertools.combinations(range(len(runs)), 2):
            v = matched_nmi(runs[a]["labels"], runs[b]["labels"], cfg.k)[0]
            perm = rng.permutation(tail.shape[0])
            nulls.append(matched_nmi(runs[a]["labels"], runs[b]["labels"][:, perm],
                                     cfg.k)[0])
            pairs.append([a, b])
            vals.append(v)
            inbest.append(a in best and b in best)
            worse.append(float(max(tr[a], tr[b])))
        vals, inbest = np.array(vals), np.array(inbest)
        rho = float(np.corrcoef(np.argsort(np.argsort(vals)),
                                np.argsort(np.argsort(worse)))[0, 1]) \
            if len(vals) > 2 else float("nan")
        out[str(h)] = {"train_fvu": tr.tolist(),
                       "tail_fvu": [r["tail_fvu"] for r in runs],
                       "pairs": pairs, "nmi": vals.tolist(), "null": nulls,
                       "best": sorted(best), "nmi_best": vals[inbest].tolist(),
                       "spearman_nmi_vs_worse_fvu": rho}
        print(f"  aq h={h:<2} train FVU {tr.min():.4f}-{tr.max():.4f}  matched head "
              f"NMI all {vals.mean():.4f} [{vals.min():.4f}, {vals.max():.4f}], best "
              f"{cfg.best} {vals[inbest].mean():.4f}  null {np.mean(nulls):.4f}  "
              f"Spearman(NMI, worse FVU) {rho:+.2f}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["rvq", "aq"], required=True)
    p.add_argument("--cache", default=str(ROOT / "data/sonar_embeddings/mc4_sent16.npy"))
    p.add_argument("--mse-weights",
                   default=str(ROOT / "data/out/sonar/sentcache/mse_weights_med837.npy"))
    p.add_argument("--prefix", type=int, default=12_800_000)
    p.add_argument("--train-rows", type=int, default=65536)
    p.add_argument("--tail-rows", type=int, default=32768)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--stages", type=int, default=4)
    p.add_argument("--hs", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--restarts", type=int, default=8, help="aq: LSQ runs per h")
    p.add_argument("--best", type=int, default=3,
                   help="aq: the lowest-error runs whose agreement is reported apart")
    p.add_argument("--iters", type=int, default=100,
                   help="rvq: Lloyd iterations per k-means at most")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45],
                   help="rvq: cascade seeds")
    p.add_argument("--gauss", default=None)
    p.add_argument("--out", default=str(ROOT / "data/out/sonar/kmeans_structure"))
    cfg = p.parse_args()
    from kmeans_seeds import rows
    data = rows(cfg)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"config": vars(cfg)}
    for name, (train, tail, _) in data.items():
        print(f"{name} ({cfg.mode}, k={cfg.k}, seeds {cfg.seeds})", flush=True)
        summary[name] = (run_rvq if cfg.mode == "rvq" else run_aq)(train, tail, cfg)
    (out / f"{cfg.mode}.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {out / (cfg.mode + '.json')}")


if __name__ == "__main__":
    main()

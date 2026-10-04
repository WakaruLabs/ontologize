"""Is the strongest layer-0 head the same head across runs?

For each run, rank layer-0 heads by eta^2 (between-group over total
variance of the embedding, grouping rows by the head's argmax entry), and
report language NMI for the head index given by --head. Then, for every
pair of runs, compare that head's partition with the same index in the
other run and with its best match there, against the median same-index NMI
of the other heads.

eta^2 is reported raw and against a random-partition baseline with the
nominal k groups, (k-1)/(n-1), as `headeta.py` does. Dividing by the groups
a head actually uses instead inflates heads that use few entries, which the
shipped initialization's layer 0 does (7--10 of 32), so raw eta^2 is the
ranking to read.

    uv run python experiments/ste-arm/topichead.py \\
        --model data/out/sonar/multilingual/ste_h76 \\
                data/out/sonar/multilingual/ste_h76_init01 \\
                data/out/sonar/multilingual/ste_h76_i01_s43
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from jaxtyping import Float, Int

from headlang import codes


def nmi(a: Int[np.ndarray, "n"], b: Int[np.ndarray, "n"]) -> float:
    """Plug-in MI normalized by the arithmetic mean of the entropies."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    J = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(J, (a, b), 1)
    J /= J.sum()
    pa, pb = J.sum(1), J.sum(0)
    nz = J > 0
    mi = (J[nz] * np.log(J[nz] / np.outer(pa, pb)[nz])).sum()
    H = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    return float(mi / max((H(pa) + H(pb)) / 2, 1e-12))


def eta2(Xc: Float[np.ndarray, "n d"], g: Int[np.ndarray, "n"],
         total: float) -> float:
    n = int(g.max()) + 1
    cnt = np.bincount(g, minlength=n)
    S = np.zeros((n, Xc.shape[1]))
    np.add.at(S, g, Xc)
    S /= np.maximum(cnt[:, None], 1)
    return float(((cnt / len(g))[:, None] * S ** 2).sum()) / total


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--rows", type=int, default=65536)
    p.add_argument("--head", type=int, default=54)
    cfg = p.parse_args()

    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:],
                   dtype=np.float32)
    raw = np.load(cfg.cache.replace(".npy", ".langs.npy"))[-cfg.rows:]
    _, y = np.unique(raw, return_inverse=True)
    Xc = (X - X.mean(0)).astype(np.float64)
    total = float((Xc ** 2).sum(1).mean())
    N, q = cfg.rows, cfg.head

    runs = {}
    for m in cfg.model:
        A, model, step = codes(m, 0, cfg.temperature, X, 256)
        A = A.astype(int)[:, :model.h]
        base = (model.k - 1) / (N - 1)
        E = np.array([eta2(Xc, A[:, j], total) for j in range(model.h)])
        used = np.array([len(np.unique(A[:, j])) for j in range(model.h)])
        order = np.argsort(-E)
        name = Path(m).name
        print(f"{name} step {step}: layer 0 by eta^2")
        for j in order[:5]:
            print(f"   h{j:<3d} eta^2 {E[j]:.4f}  {E[j] / base:5.0f}x  "
                  f"{used[j]:2d} entries used")
        rank = int(np.where(order == q)[0][0]) + 1
        print(f"   h{q}: eta^2 {E[q]:.4f} (rank {rank}/{model.h}), "
              f"language NMI {nmi(y, A[:, q]):.3f}\n")
        runs[name] = A

    names = list(runs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = runs[names[i]], runs[names[j]]
            h = a.shape[1]
            scores = [nmi(a[:, q], b[:, r]) for r in range(h)]
            best = int(np.argmax(scores))
            others = np.median([nmi(a[:, r], b[:, r])
                                for r in range(h) if r != q])
            print(f"{names[i]} h{q} vs {names[j]}: same index "
                  f"{scores[q]:.3f}, best match h{best} {scores[best]:.3f}, "
                  f"other heads' same-index median {others:.3f}")


if __name__ == "__main__":
    main()

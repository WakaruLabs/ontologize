"""How are a layer's large atoms distributed among its heads?

`entryshare.py` finds layer 0 the one layer whose entries contribute
unequally, and atom norm the factor that makes them so. This asks where
the large atoms sit: spread across every head or concentrated in a few,
whether they are a head's frequent outcomes or rare ones, whether
different heads' largest atoms share a direction, and whether the heads
holding them are the ones that reproduce across seeds.

Atoms are decoded and whitened as in `headcontrib.py`; "large" is
relative to the layer's median atom norm, so counts compare within a
model, not across datasets. Head agreement is `headcontrib.py`'s
matched centered contribution cosine against `--b`.

  uv run python experiments/ste-arm/bigatoms.py \\
      --a data/out/gpt2_l8/ste_h20_cat128 \\
      --b data/out/gpt2_l8/ste_h20_cat128-43
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
from jaxtyping import Float
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from headcontrib import atoms_and_codes, contributions, cosine_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="checkpoint dir")
    p.add_argument("--b", default=None,
                   help="seed replica of --a, for head agreement")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.00015,
                   help="irrelevant under select='ste'")
    p.add_argument("--cache", default="data/activations/gpt2_l8.npy")
    p.add_argument("--mse-weights",
                   default="data/activations/gpt2_l8.mse_weights_matched.npy")
    p.add_argument("--rows", type=int, default=32768,
                   help="rows for usage and principal directions")
    # contributions are (heads, rows, d_out) per model, so agreement runs
    # on a smaller slice of the same tail
    p.add_argument("--agree-rows", type=int, default=8192)
    p.add_argument("--batch", type=int, default=256)
    return p.parse_args()


def unit(V: Float[np.ndarray, "m d"]) -> Float[np.ndarray, "m d"]:
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-30)


def main() -> None:
    cfg = parse_args()
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float32)
    w_sqrt = np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
    D, I, _, layer, used = atoms_and_codes(cfg.a, 0, cfg.temperature, X,
                                           w_sqrt, cfg.batch)
    sel = layer == cfg.layer
    D, I = D[sel], I[sel]
    h, k = D.shape[:2]
    n = I.shape[1]
    usage = np.stack([np.bincount(I[j], minlength=k) for j in range(h)]) / n
    norm = np.linalg.norm(D, axis=-1)                          # (h, k)
    med = np.median(norm)
    print(f"{Path(cfg.a).name} step {used} layer {cfg.layer}: h={h} k={k}; "
          f"median atom norm {med:.4g}\n")

    prof = np.sort(norm, 1)[:, ::-1] / np.median(norm, 1, keepdims=True)
    ranks = sorted({0, 1, 2, 3, 4, 7, 15, k // 2 - 1, k - 1})
    print(" within-head profile, atom norm / head median, mean over heads:")
    print("   " + "  ".join(f"#{r + 1}: {prof[:, r].mean():.2f}" for r in ranks))
    print(f"   largest / head median across heads: min {prof[:, 0].min():.2f}"
          f"  median {np.median(prof[:, 0]):.2f}  max {prof[:, 0].max():.2f}")

    print("\n atoms above multiples of the layer median:")
    for t in (2, 3, 5, 10):
        cnt = (norm > t * med).sum(1)
        top = np.sort(cnt)[::-1][:max(1, h // 10)].sum() / max(cnt.sum(), 1)
        print(f"   > {t:>2}x: {cnt.sum():5d} ({cnt.sum() / norm.size:.1%}) in "
              f"{(cnt > 0).sum()}/{h} heads, at most {cnt.max()} per head; "
              f"top 10% of heads hold {top:.0%}")

    e = usage * norm ** 2
    share = e.sum(1) / e.sum()
    order = np.sort(e, 1)[:, ::-1]
    print(f"\n head share of layer energy: max {share.max():.3f}  median "
          f"{np.median(share):.4f}  (uniform {1 / h:.4f}); top 10% of heads "
          f"carry {np.sort(share)[::-1][:max(1, h // 10)].sum():.0%}")
    print(f" largest entry's share of its head's energy: median "
          f"{np.median(order[:, 0] / e.sum(1)):.3f}  max "
          f"{(order[:, 0] / e.sum(1)).max():.3f}  (uniform {1 / k:.4f})")

    rho = np.array([spearmanr(norm[j], usage[j])[0] for j in range(h)])
    big = norm > 3 * med
    print(f"\n within-head Spearman(atom norm, usage): median "
          f"{np.median(rho):+.3f} [{rho.min():+.2f}, {rho.max():+.2f}]")
    if big.any():
        print(f" usage of atoms > 3x median {np.median(usage[big]):.4f} "
              f"against {np.median(usage[~big]):.4f} for the rest "
              f"(1/k {1 / k:.4f})")

    tops = unit(D[np.arange(h), norm.argmax(1)])
    iu = np.triu_indices(h, 1)
    pool = unit(D.reshape(-1, D.shape[-1]))
    rng = np.random.default_rng(0)
    s = rng.choice(len(pool), min(2000, len(pool)), replace=False)
    iur = np.triu_indices(len(s), 1)
    print(f"\n |cos| between heads' largest atoms: median "
          f"{np.median(np.abs((tops @ tops.T)[iu])):.3f}  (random atom pairs "
          f"{np.median(np.abs((pool[s] @ pool[s].T)[iur])):.3f})")
    Xw = X * w_sqrt
    Vt = np.linalg.svd(Xw - Xw.mean(0), full_matrices=False)[2][:10]
    print(f" energy in the data's top 10 principal directions: largest atoms "
          f"{np.median(((tops @ Vt.T) ** 2).sum(1)):.3f}  all atoms "
          f"{np.median(((pool @ Vt.T) ** 2).sum(1)):.3f}  (random "
          f"{10 / D.shape[-1]:.4f})")

    if cfg.b is None:
        return
    Xa = X[-cfg.agree_rows:]
    A, la, _ = contributions(cfg.a, 0, cfg.temperature, Xa, w_sqrt, cfg.batch)
    B, lb, _ = contributions(cfg.b, 0, cfg.temperature, Xa, w_sqrt, cfg.batch)
    M = cosine_matrix(A, B, True)
    del A, B
    r, c = linear_sum_assignment(-M)
    v = np.empty(len(la))
    v[r] = M[r, c]
    v = v[la == cfg.layer]
    print(f"\n head agreement against {Path(cfg.b).name} "
          f"({cfg.agree_rows} rows): mean {v.mean():+.4f}")
    for name, x in (("largest atom", norm.max(1)),
                    ("median atom", np.median(norm, 1)),
                    ("largest / head median", prof[:, 0]),
                    ("atoms > 3x layer median", big.sum(1)),
                    ("energy share", share)):
        print(f"   Spearman(agreement, {name}): {spearmanr(v, x)[0]:+.3f}")
    best = np.argsort(v)[::-1]
    print(f"   without the best head ({best[0]}): mean "
          f"{np.delete(v, best[0]).mean():+.4f}")
    print("\n   head  agreement  energy share  largest/layer med  #>3x  "
          "largest/head med")
    for j in list(best[:6]) + [None] + list(best[-4:]):
        if j is None:
            print("   ...")
            continue
        print(f"   {j:>4}  {v[j]:+.3f}     {share[j]:.4f}        "
              f"{norm[j].max() / med:6.2f}       {big[j].sum():3d}   "
              f"{prof[j, 0]:.2f}")


if __name__ == "__main__":
    main()

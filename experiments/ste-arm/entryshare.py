"""Do all entries in a layer contribute about equally to the output?

An entry writes only on the rows where it wins its head, and then writes
its layer's input gain times its decoded atom. Its energy over the data
is therefore

    E_rows[ 1{selected} * gain^2 ] * |decoded atom|^2

in the whitened frame. Atoms are decoded as `headcontrib.py` decodes
them -- each head isolated, less `decode(0)` -- and since a head's
usage-weighted mean decoded atom is near zero, this uncentered energy
and the centered one agree (both Ginis are printed).

Per layer: dead entries, Gini over entries, effective number of entries
as a fraction of the total (exp of the entropy of energy shares), the
share carried by the top 10% and 50%, largest and smallest live entry
against the median, and two splits of the variance of log energy --
how much of it sits within heads rather than between them, and how
much each factor carries: usage, gain conditional on selection, and
atom norm (covariance shares, which sum to 1).

  uv run python experiments/ste-arm/entryshare.py \\
      --a data/out/gpt2_l8/ste_h20_cat128
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

from headcontrib import atoms_and_codes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="checkpoint dir")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.00015,
                   help="irrelevant under select='ste'")
    p.add_argument("--cache", default="data/activations/gpt2_l8.npy")
    p.add_argument("--mse-weights",
                   default="data/activations/gpt2_l8.mse_weights_matched.npy")
    p.add_argument("--rows", type=int, default=32768)
    p.add_argument("--batch", type=int, default=256)
    return p.parse_args()


def gini(x: Float[np.ndarray, "m"]) -> float:
    """0 when every entry is equal, approaching 1 when one holds all."""
    x = np.sort(x)
    m = len(x)
    return float((2 * np.arange(1, m + 1) - m - 1).dot(x) / (m * x.sum()))


def main() -> None:
    cfg = parse_args()
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float32)
    w_sqrt = np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
    D, I, G, layer, used = atoms_and_codes(cfg.a, cfg.step, cfg.temperature,
                                           X, w_sqrt, cfg.batch)
    H, k = D.shape[:2]
    n = I.shape[1]
    nl = int(layer.max()) + 1
    Gh = G[layer]                                              # (H, n)
    usage = np.stack([np.bincount(I[j], minlength=k) for j in range(H)]) / n
    g2sel = np.stack([np.bincount(I[j], weights=Gh[j] ** 2, minlength=k)
                      for j in range(H)]) / n
    norm2 = (D ** 2).sum(-1)
    energy = g2sel * norm2                                     # (H, k)
    mean = (usage[..., None] * D).sum(1)
    cenergy = g2sel * ((D - mean[:, None]) ** 2).sum(-1)
    print(f"{Path(cfg.a).name} step {used}: l={nl} h={H // nl} k={k}, "
          f"{n} rows\n")

    print(" layer  dead  gini   gini(cen)  eff/N  top10%  top50%  max/med  "
          "min/med  within-head | log-var: usage  gain|sel  atom")
    for i in range(nl):
        e = energy[layer == i]
        flat = e.ravel()
        live = flat > 0
        s = np.sort(flat)[::-1] / flat.sum()
        q = s[s > 0]
        eff = np.exp(-(q * np.log(q)).sum()) / flat.size
        med = np.median(flat[live])
        m = e > 0
        le = np.log(e[m])
        within = np.mean([np.log(r[r > 0]).var() for r in e]) / le.var()
        u, g, a = usage[layer == i], g2sel[layer == i], norm2[layer == i]
        parts = [np.cov(x, le)[0, 1] / le.var()
                 for x in (np.log(u[m]), np.log(g[m] / u[m]), np.log(a[m]))]
        print(f"  {i:>3}  {(~live).sum():4d}  {gini(flat):.3f}  "
              f"{gini(cenergy[layer == i].ravel()):.3f}      {eff:.3f}  "
              f"{s[:max(1, len(s) // 10)].sum():.3f}   {s[:len(s) // 2].sum():.3f}"
              f"  {flat.max() / med:7.1f}  {flat[live].min() / med:.2e}"
              f"     {within:.2f}     |        {parts[0]:+.2f}   "
              f"{parts[1]:+.2f}    {parts[2]:+.2f}")

    print(f"\n usage and atom norm per entry (uniform usage 1/k = {1 / k:.4f})")
    for i in range(nl):
        u = usage[layer == i].ravel()
        a = np.sqrt(norm2[layer == i]).ravel()
        print(f"  layer {i}: usage CV {u.std() / u.mean():.2f}  "
              f"[{u.min():.2e}, {u.max():.3f}]   atom norm CV "
              f"{a.std() / a.mean():.2f}")
    print("\n layer share of total entry energy: " + "  ".join(
        f"{i}: {energy[layer == i].sum() / energy.sum():.3f}"
        for i in range(nl)))


if __name__ == "__main__":
    main()

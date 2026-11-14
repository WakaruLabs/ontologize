"""Where a code's held-out error sits over the whitened spectrum.

`pareto.gaussian_reference` reverse water-fills a rate over the
eigenvalues lam_i of the whitened data covariance: every direction above
the water level theta is left with error theta, every one below it with
its whole variance. This script splits a model's held-out error over the
same eigenbasis and sets it beside that allocation, direction by
direction, so an excess over the reference can be located:

  - spread over the directions water-filling codes (error above theta
    where lam_i > theta): the code is inefficient across the board;
  - error above a direction's own variance (e_i > lam_i): the code adds
    noise there, worse than reconstructing the mean, the failure of a
    predictive coder below the water level (Kim & Ramstad 2003; Andersen
    & Kleijn 1999);
  - concentrated in a few directions: what the atoms miss.

Models, on the cache's held-out tail (sae.py's holdout):
  - the Ontologizer at --ckpt with every layer's gain pinned to its
    held-out mean (gainablate.py's forward), so the code is the discrete
    index alone, and each prefix of its layers at its own rate
    (l * h * log2 k bits per layer);
  - any --sae runs, unquantized (their rate is not a fixed number of
    bits, so they get no reference of their own).

For each, a per-direction affine rescaling of the reconstruction (a
Wiener-style shrinkage, y_i ~ a_i yhat_i + b_i in the eigenbasis) is fitted
on one random half of the rows and scored on the other, which says how much
of the error a diagonal post-filter could remove.

    JAX_PLATFORMS=cpu uv run python experiments/ste-arm/errspectrum.py \\
        --ckpt data/out/sonar/multilingual/ste_h76_init01 \\
        --sae data/out/sonar/sae_conv/m11264_k160 --sae-topk 160
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from pareto import load_onto

# eigen-rank bands the summary reports, as half-open [lo, hi) ranks
BANDS = ((0, 8), (8, 32), (32, 128), (128, 512), (512, 1024))


def water_level(lam: Float[np.ndarray, "d"], bits: float) -> float:
    """The reverse water-filling level theta at which the rate
    sum max(0, log2(lam_i / theta) / 2) equals `bits`. Bisects log theta,
    as `pareto.gaussian_reference` does."""
    lam = np.asarray(lam, np.float64)
    lam = lam[lam > 0]
    lo, hi = math.log(lam.min()) - 60.0, math.log(lam.max())
    for _ in range(200):
        mid = (lo + hi) / 2
        rate = np.maximum(0.0, np.log2(lam / math.exp(mid)) / 2).sum()
        lo, hi = (mid, hi) if rate > bits else (lo, mid)
    return math.exp(hi)


def effective_bits(lam: Float[np.ndarray, "d"], fvu: float) -> float:
    """The rate at which reverse water-filling over `lam` reaches `fvu`:
    what the code's error would cost a Gaussian-optimal code."""
    lam = np.asarray(lam, np.float64)
    lam = lam[lam > 0]
    lo, hi = 0.0, float(np.log2(lam.max() / lam.min()) / 2 * len(lam) + 1)
    for _ in range(100):
        mid = (lo + hi) / 2
        ref = np.minimum(water_level(lam, mid), lam).sum() / lam.sum()
        lo, hi = (mid, hi) if ref > fvu else (lo, mid)
    return hi


def eigenbasis(Y: Float[np.ndarray, "n d"]
               ) -> Tuple[Float[np.ndarray, "d"], Float[np.ndarray, "d d"]]:
    """Eigenvalues (descending) and eigenvectors (columns) of the
    covariance of Y's rows, in float64."""
    Yc = Y - Y.mean(0)
    C = (Yc.T @ Yc).astype(np.float64) / len(Y)
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    return np.clip(lam[order], 0.0, None), V[:, order]


def direction_errors(E: Float[np.ndarray, "n d"], V: Float[np.ndarray, "d d"]
                     ) -> Float[np.ndarray, "d"]:
    """Mean squared error along each eigenvector: the error's second moment
    (bias included), so the entries sum to the mean squared error."""
    P = E.astype(np.float64) @ V
    return (P ** 2).mean(0)


def shrink_fit(Z: Float[np.ndarray, "n d"], Zh: Float[np.ndarray, "n d"]
               ) -> Tuple[Float[np.ndarray, "d"], Float[np.ndarray, "d"]]:
    """Per-coordinate least-squares affine map Zh -> Z: z ~ a * zh + b.
    A coordinate whose reconstruction does not vary gets a = 0, b = mean."""
    mz, mh = Z.mean(0), Zh.mean(0)
    vh = ((Zh - mh) ** 2).mean(0)
    cov = ((Zh - mh) * (Z - mz)).mean(0)
    a = np.where(vh > 1e-30, cov / np.maximum(vh, 1e-30), 0.0)
    return a, mz - a * mh


def band_summary(lam: Float[np.ndarray, "d"], e: Float[np.ndarray, "d"],
                 D: Float[np.ndarray, "d"] = None,
                 bands: Sequence[Tuple[int, int]] = BANDS) -> List[Dict]:
    """Per eigen-rank band: the band's share of the variance, of the error,
    and (given the reference allocation D) of the reference error and of
    the excess over it, the error over the reference, and how many of the
    band's directions have error above their own variance."""
    tot_lam, tot_e = lam.sum(), e.sum()
    excess = np.maximum(e - D, 0.0) if D is not None else None
    out = []
    for lo, hi in bands:
        hi = min(hi, len(lam))
        if lo >= hi:
            continue
        s = slice(lo, hi)
        row = {"ranks": f"{lo + 1}-{hi}",
               "var_share": float(lam[s].sum() / tot_lam),
               "err_share": float(e[s].sum() / tot_e),
               "err_over_var": float(e[s].sum() / lam[s].sum()),
               "n_err_above_var": int((e[s] > lam[s]).sum())}
        if D is not None:
            row["ref_share"] = float(D[s].sum() / D.sum())
            row["err_over_ref"] = float(e[s].sum() / D[s].sum())
            row["excess_share"] = float(excess[s].sum()
                                        / max(excess.sum(), 1e-30))
        out.append(row)
    return out


def onto_recons(ckpt: str, T: float, X: Float[np.ndarray, "n d"], batch: int
                ) -> Tuple[Float[np.ndarray, "l n d"], Dict]:
    """Each layer prefix's reconstruction with every layer's gain pinned
    to its mean over these rows: the code is then the argmax index alone."""
    model, raw, step = load_onto(ckpt, 0)
    params = {"params": raw}
    l = model.l

    def fwd(module, Xb: Float[Array, "b d"], const: Float[Array, "l"],
            pin: bool) -> Tuple[Float[Array, "l b d"], Float[Array, "l b"]]:
        E, _ = module.encode(Xb, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        outs, gains = [], []
        for i, de in enumerate(module.dictencs):
            U, G = de.gainshape_in(Ein)
            P = de.dict.cluster(de.classifier(U), T)
            if G is None:
                g, gains_i = None, jnp.ones(Xb.shape[:1])
            else:
                g = jnp.where(pin, jnp.full_like(G, const[i]), G)
                gains_i = G[..., 0]
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), g)
            outs.append(module.decode(R))
            gains.append(gains_i)
            if i < module.l - 1:
                Ein = module.nextinput(Xb, R, P.reshape(Xb.shape[0], -1))
        return jnp.stack(outs), jnp.stack(gains)

    f = jax.jit(lambda p, x, c, pin: model.apply(p, x, c, pin, method=fwd),
                static_argnums=3)
    zero = jnp.zeros(l)
    G = np.concatenate([np.asarray(f(params, jnp.asarray(X[i:i + batch]),
                                     zero, False)[1])
                        for i in range(0, len(X), batch)], axis=1)
    gmean = jnp.asarray(G.mean(1))
    Y = np.concatenate([np.asarray(f(params, jnp.asarray(X[i:i + batch]),
                                     gmean, True)[0])
                        for i in range(0, len(X), batch)], axis=1)
    bits = model.h * math.log2(model.k)
    return Y, {"ckpt": ckpt, "step": step, "l": l, "h": model.h,
               "k": model.k, "bits_per_layer": bits,
               "gain_means": [float(x) for x in np.asarray(gmean)]}


def sae_recon(run: str, topk: int, X: Float[np.ndarray, "n d"], batch: int
              ) -> Float[np.ndarray, "n d"]:
    import sae
    params = dict(np.load(Path(run) / "params.npz"))
    out = []
    for i in range(0, len(X), batch):
        z = sae.encode(params, jnp.asarray(X[i:i + batch]), topk)
        out.append(np.asarray(sae.decode(params, z)))
    return np.concatenate(out)


def score(name: str, Xh: Float[np.ndarray, "n d"], Z: Float[np.ndarray, "n d"],
          rw: Float[np.ndarray, "d"], mean: Float[np.ndarray, "d"],
          lam: Float[np.ndarray, "d"], V: Float[np.ndarray, "d d"],
          bits: float, half: Int[np.ndarray, "n"]) -> Tuple[Dict, np.ndarray]:
    """The summary row for one reconstruction `Xh`, and its per-direction
    error. `Z` is the data in the whitened eigenbasis, (X - mean) * rw @ V."""
    Zh = (Xh - mean) * rw @ V
    e = ((Zh - Z) ** 2).mean(0)
    row = {"model": name, "fvu": float(e.sum() / lam.sum()),
           "n_err_above_var": int((e > lam).sum())}
    row["effective_bits"] = effective_bits(lam, row["fvu"])
    D = None
    if bits:
        theta = water_level(lam, bits)
        D = np.minimum(theta, lam)
        row.update({"bits": bits, "theta": theta,
                    "n_above_water": int((lam > theta).sum()),
                    "ref_fvu": float(D.sum() / lam.sum()),
                    "over_ref": float(e.sum() / D.sum())})
    row["bands"] = band_summary(lam, e, D)
    # the diagonal post-filter: fitted on one half, scored on the other
    fit, test = half == 0, half == 1
    a, b = shrink_fit(Z[fit], Zh[fit])
    Zt = Z[test]
    den = ((Zt - Zt.mean(0)) ** 2).sum()
    row["fvu_test_half"] = float(((Zh[test] - Zt) ** 2).sum() / den)
    row["fvu_test_half_shrunk"] = float(
        ((Zh[test] * a + b - Zt) ** 2).sum() / den)
    row["shrink_gain_quartiles"] = [float(x) for x in
                                    np.quantile(a, [0.25, 0.5, 0.75])]
    return row, e


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="data/out/sonar/multilingual/ste_h76_init01")
    p.add_argument("--temperature", type=float, default=1.5e-4)
    p.add_argument("--sae", nargs="*", default=[],
                   help="sae.py run dirs, scored unquantized")
    p.add_argument("--sae-topk", type=int, nargs="*", default=[],
                   help="each --sae run's topk, in order")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--rows", type=int, default=32768,
                   help="the cache's held-out tail")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/out/sonar/errspectrum")
    cfg = p.parse_args()
    if len(cfg.sae_topk) != len(cfg.sae):
        raise SystemExit("--sae-topk needs one value per --sae run")

    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float32)
    rw = np.sqrt(np.load(cfg.mse_weights)).astype(np.float64)
    mean = X.mean(0).astype(np.float64)
    lam, V = eigenbasis((X - mean) * rw)
    Z = (X - mean) * rw @ V
    half = np.random.default_rng(cfg.seed).permutation(len(X)) % 2
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    Y, info = onto_recons(cfg.ckpt, cfg.temperature, X, cfg.batch)
    name = Path(cfg.ckpt).name
    print(f"{name} step {info['step']}: {info['l']} x {info['h']} heads, "
          f"k {info['k']}, gains pinned to {info['gain_means']}")
    rows, errs = [], {}
    for i in range(info["l"]):
        bits = (i + 1) * info["bits_per_layer"]
        row, e = score(f"{name}[:{i + 1}]", Y[i], Z, rw, mean, lam, V, bits,
                       half)
        rows.append(row)
        errs[row["model"]] = e
    for run, topk in zip(cfg.sae, cfg.sae_topk):
        row, e = score(Path(run).name, sae_recon(run, topk, X, cfg.batch),
                       Z, rw, mean, lam, V, 0.0, half)
        rows.append(row)
        errs[row["model"]] = e

    print(f"\n{len(X)} rows; whitened variance in ranks 1-8 "
          f"{lam[:8].sum() / lam.sum():.1%}, 1-128 "
          f"{lam[:128].sum() / lam.sum():.1%}")
    for r in rows:
        ref = (f"  bits {r['bits']:.0f}, reference {r['ref_fvu']:.4f}, "
               f"over it {r['over_ref']:.2f}x, {r['n_above_water']} "
               f"directions above water" if "bits" in r else "")
        print(f"\n{r['model']}: FVU_w {r['fvu']:.4f}{ref}")
        print(f"  the reference reaches this FVU at "
              f"{r['effective_bits']:.0f} bits"
              + (f" ({r['effective_bits'] / r['bits']:.0%} of the code's)"
                 if "bits" in r else ""))
        print(f"  directions with error above their variance: "
              f"{r['n_err_above_var']}")
        print(f"  test half {r['fvu_test_half']:.4f} -> "
              f"{r['fvu_test_half_shrunk']:.4f} after the diagonal "
              f"rescaling (gain quartiles "
              + ", ".join(f"{x:.3f}" for x in r["shrink_gain_quartiles"])
              + ")")
        hdr = f"  {'ranks':>9} {'var':>7} {'err':>7} {'err/var':>8}"
        if "bits" in r:
            hdr += f" {'ref':>7} {'err/ref':>8} {'excess':>7}"
        print(hdr + f" {'e>var':>6}")
        for b in r["bands"]:
            line = (f"  {b['ranks']:>9} {b['var_share']:7.1%} "
                    f"{b['err_share']:7.1%} {b['err_over_var']:8.3f}")
            if "bits" in r:
                line += (f" {b['ref_share']:7.1%} {b['err_over_ref']:8.2f}"
                         f" {b['excess_share']:7.1%}")
            print(line + f" {b['n_err_above_var']:6d}")

    (out / "summary.json").write_text(json.dumps(
        {"onto": info, "rows": len(X), "cache": cfg.cache,
         "mse_weights": cfg.mse_weights, "models": rows}, indent=1))
    with open(out / "directions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "lam"] + list(errs))
        for j in range(len(lam)):
            w.writerow([j + 1, f"{lam[j]:.6e}"]
                       + [f"{errs[m][j]:.6e}" for m in errs])
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

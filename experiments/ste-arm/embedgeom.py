"""Is language a dominant factor in SONAR embedding geometry, or a cluster
structure riding on top of something else?

Separates the two readings of "dense and uniformly distributed":
  - how many directions the cloud actually occupies (effective dimension of
    the covariance, against the 1024 a uniform sphere would give)
  - how much of the variance language explains at all (eta^2 = between-class
    over total), which is what an MSE objective allocates capacity by
  - whether languages are tight clusters or overlapping regions (within- vs
    between-language mean cosine, against the cloud's own baseline)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/keira/flock/ontologize")
from ontologize.data.langs import MC4_TO_SONAR                # noqa: E402

CACHE = "/home/keira/flock/ontologize/data/sonar_embeddings/mc4_4M.npy"
N = 131072


def main():
    mm = np.load(CACHE, mmap_mode="r")
    X = np.asarray(mm[-N:], dtype=np.float64)
    raw = np.load(CACHE.replace(".npy", ".langs.npy"))[-N:]
    names, y = np.unique(raw, return_inverse=True)
    scripts = np.array([("Latn" if c.endswith("-Latn")
                         else MC4_TO_SONAR.get(c, "?_?").split("_")[1])
                        for c in names])
    _, s = np.unique(scripts, return_inverse=True)
    y_scr = s[y]

    nrm = np.linalg.norm(X, axis=1)
    print(f"{N} rows, d={X.shape[1]}: ||x|| mean {nrm.mean():.4f} "
          f"sd {nrm.std():.2e}  (SONAR is unit-normed)")

    mu = X.mean(0)
    Xc = X - mu
    total = float((Xc ** 2).sum(1).mean())
    print(f"||mean embedding|| = {np.linalg.norm(mu):.4f}, so the cloud sits "
          f"on a cap, not centred on the origin")
    print(f"total variance (mean ||x-mu||^2) = {total:.4f}\n")

    # effective dimension of the covariance
    C = (Xc.T @ Xc) / len(Xc)
    lam = np.linalg.eigvalsh(C).clip(0)[::-1]
    eff = lam.sum() ** 2 / (lam ** 2).sum()
    cum = np.cumsum(lam) / lam.sum()
    print(f"effective dimension (participation ratio) {eff:.1f} of "
          f"{X.shape[1]}")
    for q in (0.5, 0.9, 0.99):
        print(f"  {int(q*100)}% of variance in the top "
              f"{int(np.searchsorted(cum, q)) + 1} PCs")

    # how much variance does the label explain?
    print()
    for lab, yy, nn in (("language", y, len(names)),
                        ("script", y_scr, int(y_scr.max()) + 1)):
        p = np.bincount(yy, minlength=nn) / len(yy)
        M = np.zeros((nn, X.shape[1]))
        np.add.at(M, yy, Xc)
        M /= np.maximum(np.bincount(yy, minlength=nn)[:, None], 1)
        between = float((p[:, None] * M ** 2).sum())
        print(f"{lab:>9}: eta^2 = between/total = {between/total:.4f} "
              f"({100*between/total:.2f}% of variance), {nn} classes")

    # cluster tightness: within- vs between-label cosine, on a subsample
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), 8192, replace=False)
    U = X[idx] / np.linalg.norm(X[idx], axis=1, keepdims=True)
    G = U @ U.T
    same = y[idx][:, None] == y[idx][None, :]
    off = ~np.eye(len(idx), dtype=bool)
    wi = float(G[same & off].mean())
    be = float(G[~same & off].mean())
    print(f"\nmean cosine: within-language {wi:+.4f}  "
          f"between-language {be:+.4f}  gap {wi-be:+.4f}")
    print(f"  (a uniform cloud on the sphere would give ~0 for both; the "
          f"shared mean direction puts the floor at {np.linalg.norm(mu)**2:+.4f})")
    sd = float(G[~same & off].std())
    print(f"  between-language sd {sd:.4f}, so the within/between gap is "
          f"{(wi-be)/sd:.2f} sd of the pair distribution")


if __name__ == "__main__":
    main()

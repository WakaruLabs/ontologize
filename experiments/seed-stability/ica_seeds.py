"""Do FastICA's sources of SONAR reproduce on another sample of its rows?

ICA is the linear code that non-Gaussian data identify: sources that are
independent and not Gaussian are fixed up to order, sign and scale. This
fits `kmeans_seeds.py`'s symmetric FastICA twice, in the `med837` whitened
frame: on `kmeans_seeds.py`'s training rows (A, usually continued from a
saved `frame_ica.npz`) and on as many rows drawn from the same prefix and
disjoint from them (B, from another random start). Both are applied to the
same held-out tail, and each of A's sources is matched one to one to B's
(Hungarian) by
- `corr`: |correlation| between the two sources over the tail rows;
- `mixing`: |cosine| between their directions in the whitened frame (each
  source's row of `mixing`), as an SAE's decoder columns are compared.

Beside them: B's sources rotated at random (in its sphered frame), which
no data pin, and the principal frames' coordinates matched the same way
(the covariance's own basis, pinned only where eigenvalues are apart). The
per-source table gives each of A's sources its kurtosis and variance share
on A's training rows, its partner, both matched values and the runner-up
|correlation|; the summary bins them by kurtosis, since a source near
Gaussian is not identified.

    JAX_PLATFORMS=cpu uv run python experiments/seed-stability/ica_seeds.py \\
        --frame-a data/out/sonar/kmeans_seeds_ica/frame_ica.npz \\
        --out data/out/sonar/ica_seeds
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from kmeans_seeds import (ica_frame, mixing, shape_stats, to_sources,  # noqa: E402
                          train_index)

KURTOSIS_BINS = (1.0, 2.0, 5.0, 10.0, np.inf)


def abs_corr(SA: Float[np.ndarray, "n p"], SB: Float[np.ndarray, "n q"]
             ) -> Float[np.ndarray, "p q"]:
    """|Pearson correlation| between every column of SA and of SB."""
    a = (SA - SA.mean(0)) / SA.std(0)
    b = (SB - SB.mean(0)) / SB.std(0)
    return np.abs(a.T @ b / len(SA))


def abs_cos(DA: Float[np.ndarray, "p d"], DB: Float[np.ndarray, "q d"]
            ) -> Float[np.ndarray, "p q"]:
    """|cosine| between every row of DA and of DB."""
    a = DA / np.linalg.norm(DA, axis=1, keepdims=True)
    b = DB / np.linalg.norm(DB, axis=1, keepdims=True)
    return np.abs(a @ b.T)


def match(C: Float[np.ndarray, "p q"]) -> tuple[Int[np.ndarray, "p"],
                                                 Float[np.ndarray, "p"],
                                                 Float[np.ndarray, "p"]]:
    """One-to-one matching maximizing the total of C (Hungarian): each
    row's partner, its value, and the row's best value among the other
    columns (the runner-up)."""
    r, c = linear_sum_assignment(-C)
    partner = np.empty(C.shape[0], int)
    partner[r] = c
    val = C[np.arange(len(C)), partner]
    rest = C.copy()
    rest[np.arange(len(C)), partner] = -np.inf
    return partner, val, rest.max(1)


def random_rotation(p: int, seed: int) -> Float[np.ndarray, "p p"]:
    Q, Rm = np.linalg.qr(np.random.default_rng(seed).standard_normal((p, p)))
    return Q * np.sign(np.diag(Rm))


def describe(v: Float[np.ndarray, "p"]) -> dict:
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "q10": float(np.quantile(v, 0.1)), "q90": float(np.quantile(v, 0.9)),
            "over_0.9": int((v > 0.9).sum()), "over_0.99": int((v > 0.99).sum())}


def compare(fa: dict, fb: dict, tail: Float[np.ndarray, "m d"], seed: int = 0) -> dict:
    """The matched values of A's sources against B's, against B's rotated
    at random, and of the two principal frames' sphered coordinates."""
    SA, SB = to_sources(tail, fa), to_sources(tail, fb)
    pa, ca, ra = match(abs_corr(SA, SB))
    mix_partner, cm, _ = match(abs_cos(mixing(fa), mixing(fb)))
    _, cr, _ = match(abs_corr(SA, SB @ random_rotation(SB.shape[1], seed)))
    sph = lambda f: ((tail.astype(np.float64) - f["mean"]) @ f["eigenvectors"]
                     / np.sqrt(f["eigenvalues"]))
    PA, PB = sph(fa), sph(fb)
    _, cp, _ = match(abs_corr(PA, PB))
    return {"partner": pa, "corr": ca, "runner_up": ra, "mixing": cm,
            "same_partner": float((mix_partner == pa).mean()),
            "rotated": cr, "principal": cp}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=str(ROOT / "data/sonar_embeddings/mc4_sent16.npy"))
    p.add_argument("--mse-weights",
                   default=str(ROOT / "data/out/sonar/sentcache/mse_weights_med837.npy"))
    p.add_argument("--prefix", type=int, default=12_800_000)
    p.add_argument("--train-rows", type=int, default=131072)
    p.add_argument("--tail-rows", type=int, default=32768)
    p.add_argument("--frame-a", default=None,
                   help="kmeans_seeds.py's frame_ica.npz on its training rows "
                        "(continued until converged; fitted afresh without it)")
    p.add_argument("--frame-b", default=None,
                   help="an earlier run's frame_b.npz, to continue B's fit")
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--patience", type=int, default=50,
                   help="kmeans_seeds.py's --ica-patience (0: plain FastICA)")
    p.add_argument("--out", default=str(ROOT / "data/out/sonar/ica_seeds"))
    cfg = p.parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    sw = np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
    X = np.load(cfg.cache, mmap_mode="r")
    ia = train_index(cfg.prefix, cfg.train_rows)
    ib = train_index(cfg.prefix, cfg.train_rows, seed=1, exclude=ia)
    TA = np.asarray(X[ia], np.float32) * sw
    TB = np.asarray(X[ib], np.float32) * sw
    tail = np.asarray(X[-cfg.tail_rows:], np.float32) * sw
    load = lambda f: dict(np.load(f)) if f else None
    print("A: FastICA", flush=True)
    fa = ica_frame(TA, 0, cfg.iters, load(cfg.frame_a), log_every=25,
                   patience=cfg.patience, checkpoint_path=out / "frame_a.partial.npz")
    np.savez(out / "frame_a.npz", **fa)
    print(f"A: {fa['iterations']} iterations, last change {fa['last_change']:.2e}\n"
          "B: FastICA", flush=True)
    fb = ica_frame(TB, 1, cfg.iters, load(cfg.frame_b), log_every=25,
                   patience=cfg.patience, checkpoint_path=out / "frame_b.partial.npz")
    np.savez(out / "frame_b.npz", **fb)
    print(f"B: {fb['iterations']} iterations, last change {fb['last_change']:.2e}",
          flush=True)

    r = compare(fa, fb, tail)
    SA = to_sources(TA, fa)
    st = shape_stats(SA)
    var = (mixing(fa) ** 2).sum(1) * SA.var(0)
    var = var / var.sum()
    with open(out / "sources.csv", "w") as f:
        f.write("source,kurtosis,skew,variance_share,partner,corr,runner_up,mixing\n")
        for i in range(len(var)):
            f.write(f"{i},{st['kurtosis'][i]:.6g},{st['skew'][i]:.6g},{var[i]:.6g},"
                    f"{r['partner'][i]},{r['corr'][i]:.6g},{r['runner_up'][i]:.6g},"
                    f"{r['mixing'][i]:.6g}\n")
    summary = {"rows": {"train": cfg.train_rows, "tail": cfg.tail_rows},
               "fits": {n: {"iterations": int(f["iterations"]),
                            "last_change": float(f["last_change"])}
                        for n, f in (("a", fa), ("b", fb))},
               "corr": describe(r["corr"]), "mixing": describe(r["mixing"]),
               "runner_up": describe(r["runner_up"]),
               "same_partner_by_mixing": r["same_partner"],
               "rotated": describe(r["rotated"]), "principal": describe(r["principal"]),
               "by_kurtosis": {}}
    kurt = st["kurtosis"]
    for lo, hi in zip((-np.inf,) + KURTOSIS_BINS[:-1], KURTOSIS_BINS):
        m = (kurt >= lo) & (kurt < hi)
        if m.any():
            summary["by_kurtosis"][f"[{lo:g}, {hi:g})"] = {
                "sources": int(m.sum()), "variance": float(var[m].sum()),
                "corr_median": float(np.median(r["corr"][m])),
                "mixing_median": float(np.median(r["mixing"][m]))}
    for name in ("corr", "mixing", "rotated", "principal"):
        s = summary[name]
        print(f"{name:<9} matched: mean {s['mean']:.3f}, median {s['median']:.3f}, "
              f"[q10 {s['q10']:.3f}, q90 {s['q90']:.3f}], over 0.9: {s['over_0.9']}, "
              f"over 0.99: {s['over_0.99']}", flush=True)
    for b, s in summary["by_kurtosis"].items():
        print(f"kurtosis {b}: {s['sources']} sources, {s['variance']:.1%} of the "
              f"variance, matched corr median {s['corr_median']:.3f}, mixing "
              f"{s['mixing_median']:.3f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {out / 'summary.json'}")


if __name__ == "__main__":
    main()

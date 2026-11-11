"""Which heads and latents does the source tag move?

`retag.py compare` finds that the tag shift reassigns a hard code's heads
about as often as an isotropic random shift of each row's own size: the
code is no more stable to the tag than to noise. That is a mean over a
layer's heads. This asks the question head by head, and of the SAEs
latent by latent, since the tag shift is the one perturbation here whose
meaning is known: the same text under another source tag.

For every head, over the changed rows, its keep rate is the share of rows
on which it selects the same entry under the cached tag and the intended
one. The same rate under random shifts of the rows' own sizes is its
reference, and their difference, `excess = keep_random - keep_tag`, is
positive for a head the tag moves more than noise does and negative for
one it spares. Two independent random shifts give the rates' sampling
spread, so an excess is read against the noise it would have with no tag
effect.

For an SAE the unit is a latent: of the changed rows on which it is
active under the cached tag, the share on which it stays active under
the intended tag, against the same under a random shift. The support
Jaccard that `retag.py` reports for the tag shift is given beside a
random shift's, which that report lacks.

Each unit's excess is set beside how much it knows about language (a
head's NMI with the language label on the cached rows; for a latent, the
largest share of its active rows that one language holds).

    uv run python experiments/lang-tags/tagflip.py
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))

import numpy as np
from jaxtyping import Bool, Float, Int

from retag import (cached_tag, onto_runner, random_shift,  # noqa: E402
                   run_batched)

from ontologize.data.langs import MC4_TO_SONAR  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--encodes", default="data/out/sonar/langtags/encodes.npz")
    p.add_argument("--onto", nargs="*", default=[
        "data/out/sonar/multilingual/ste_h76_init01",
        "data/out/sonar/multilingual/ste_l1_h380_i01",
        "data/out/sonar/multilingual/ste_k2_h380"])
    p.add_argument("--sae", nargs="*", default=[
        "data/out/sonar/sae_conv/m11264_k32",
        "data/out/sonar/sae_conv/m11264_k160"])
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--min-active", type=int, default=200,
                   help="score a latent only if it is active on this many "
                        "changed rows")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/out/sonar/langtags/tagflip")
    return p.parse_args()


# ---------- statistics (pure) ----------

def keep_rate(A: Int[np.ndarray, "n u"], B: Int[np.ndarray, "n u"]
              ) -> Float[np.ndarray, "u"]:
    """Per unit (a head), the share of rows on which it selects the same
    entry in `A` and `B`."""
    return (A == B).mean(0)


def stay_rate(Z: Bool[np.ndarray, "n m"], Z2: Bool[np.ndarray, "n m"]
              ) -> Tuple[Float[np.ndarray, "m"], Int[np.ndarray, "m"]]:
    """Per latent, of the rows on which it is active in `Z`, the share on
    which it is active in `Z2`, and the number of such rows (the rate is
    NaN where that is zero)."""
    n = Z.sum(0)
    both = (Z & Z2).sum(0)
    return np.where(n > 0, both / np.maximum(n, 1), np.nan), n


def jaccard(Z: Bool[np.ndarray, "n m"], Z2: Bool[np.ndarray, "n m"]
            ) -> Float[np.ndarray, "n"]:
    """Per row, |active in both| / |active in either|."""
    inter = (Z & Z2).sum(1)
    union = (Z | Z2).sum(1)
    return inter / np.maximum(union, 1)


def excess(keep_tag: Float[np.ndarray, "u"],
           keep_r1: Float[np.ndarray, "u"], keep_r2: Float[np.ndarray, "u"]
           ) -> Tuple[Float[np.ndarray, "u"], float]:
    """Each unit's `mean(keep_r1, keep_r2) - keep_tag`, and the standard
    deviation it would have with no tag effect.

    Write s for one rate's sampling spread, taken to be the same for the
    tag's rate as for a random draw's. The difference of the two draws
    scatters by s sqrt(2), which is what the data give; the excess is one
    rate minus the mean of two, which scatters by s sqrt(3/2). So the
    noise is the standard deviation over units of `keep_r1 - keep_r2`
    times sqrt(3)/2."""
    ref = 0.5 * (keep_r1 + keep_r2)
    ok = np.isfinite(keep_r1) & np.isfinite(keep_r2)
    return ref - keep_tag, float(np.std((keep_r1 - keep_r2)[ok])
                                 * np.sqrt(3.0) / 2.0)


def label_nmi(a: Int[np.ndarray, "n"], y: Int[np.ndarray, "n"], ka: int,
              ky: int) -> float:
    """Symmetric normalized mutual information of two labelings."""
    C = np.zeros((ka, ky))
    np.add.at(C, (a, y), 1.0)
    C /= C.sum()
    pa, py = C.sum(1), C.sum(0)
    H = lambda p: float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    nz = C > 0
    I = float((C[nz] * np.log2(C[nz] / np.outer(pa, py)[nz])).sum())
    den = 0.5 * (H(pa) + H(py))
    return I / den if den > 0 else 0.0


def top_label_share(Z: Bool[np.ndarray, "n m"], y: Int[np.ndarray, "n"],
                    n_cls: int) -> Float[np.ndarray, "m"]:
    """Per latent, the largest share of its active rows that one label
    holds (NaN for a latent that is never active)."""
    Y = np.eye(n_cls, dtype=np.float32)[y]                     # (n, c)
    counts = Z.astype(np.float32).T @ Y                        # (m, c)
    tot = counts.sum(1)
    return np.where(tot > 0, counts.max(1) / np.maximum(tot, 1), np.nan)


def spearman(a: Float[np.ndarray, "u"], b: Float[np.ndarray, "u"]) -> float:
    """Rank correlation over the units where both are finite."""
    ok = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[ok])).astype(np.float64)
    rb = np.argsort(np.argsort(b[ok])).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def describe(ex: Float[np.ndarray, "u"], noise: float) -> Dict[str, float]:
    """Summary of a set of excesses against the reference's noise."""
    e = ex[np.isfinite(ex)]
    return {"mean": float(e.mean()), "sd": float(e.std()), "noise": noise,
            "min": float(e.min()), "max": float(e.max()),
            "above_3": float((e > 3 * noise).mean()),
            "below_3": float((e < -3 * noise).mean()), "n": int(len(e))}


# ---------- the experiment ----------

def main() -> None:
    cfg = parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    enc = np.load(cfg.encodes)
    rows, langs = enc["rows"], enc["langs"]
    X_int = np.asarray(enc["X_int"], np.float32)
    X_c = np.asarray(np.load(cfg.cache, mmap_mode="r")[rows], np.float32)
    changed = np.array([cached_tag(g) != MC4_TO_SONAR[g] for g in langs])
    names = sorted(set(langs.tolist()))
    y = np.array([names.index(g) for g in langs])
    rng = np.random.default_rng(cfg.seed)
    D = X_int - X_c
    X_r1 = random_shift(X_c, D, rng)
    X_r2 = random_shift(X_c, D, rng)
    print(f"{int(changed.sum())} changed rows of {len(rows)}; shift "
          f"|x_int - x_cached| median {np.median(np.linalg.norm(D[changed], axis=1)):.3f}\n")
    summary: Dict[str, Any] = {"changed_rows": int(changed.sum())}

    for ckpt in cfg.onto:
        f, model, step = onto_runner(ckpt, cfg.temperature)
        l, h, k = model.l, model.h, model.k
        A = {name: run_batched(f, X, cfg.batch)[1].reshape(len(X), l * h)
             for name, X in (("cached", X_c), ("intended", X_int),
                             ("r1", X_r1), ("r2", X_r2))}
        c = changed
        kt = keep_rate(A["cached"][c], A["intended"][c])
        k1 = keep_rate(A["cached"][c], A["r1"][c])
        k2 = keep_rate(A["cached"][c], A["r2"][c])
        ex, noise = excess(kt, k1, k2)
        nmi = np.array([label_nmi(A["cached"][:, j], y, k, len(names))
                        for j in range(l * h)])
        layer = np.repeat(np.arange(l), h)
        name = Path(ckpt).name
        res = {"step": int(step), "all": describe(ex, noise),
               "keep_tag": [float(kt[layer == i].mean()) for i in range(l)],
               "keep_random": [float(0.5 * (k1 + k2)[layer == i].mean())
                               for i in range(l)],
               "by_layer": [describe(ex[layer == i], noise)
                            for i in range(l)],
               "spearman_excess_nmi": [spearman(ex[layer == i],
                                                nmi[layer == i])
                                       for i in range(l)]}
        order = np.argsort(-ex)
        res["most_moved"] = [
            {"layer": int(layer[j]), "head": int(j % h),
             "keep_tag": float(kt[j]), "keep_random": float(0.5 * (k1 + k2)[j]),
             "excess": float(ex[j]), "nmi_lang": float(nmi[j])}
            for j in order[:8]]
        res["most_spared"] = [
            {"layer": int(layer[j]), "head": int(j % h),
             "keep_tag": float(kt[j]), "keep_random": float(0.5 * (k1 + k2)[j]),
             "excess": float(ex[j]), "nmi_lang": float(nmi[j])}
            for j in order[::-1][:8]]
        summary[name] = res
        with open(out / f"{name}.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["layer", "head", "keep_tag", "keep_r1", "keep_r2",
                         "excess", "nmi_lang"])
            for j in range(l * h):
                wr.writerow([int(layer[j]), j % h] + [
                    f"{v:.5f}" for v in (kt[j], k1[j], k2[j], ex[j], nmi[j])])
        print(f"{name} (step {step}; {l}x{h}, k={k}); reference noise "
              f"{noise:.4f}")
        print(f"  {'layer':<6}{'keep tag':>9}{'keep rnd':>9}{'excess':>8}"
              f"{'sd':>7}{'min':>8}{'max':>8}{'>3n':>6}{'<-3n':>6}"
              f"{'rho(nmi)':>10}")
        for i in range(l):
            b = res["by_layer"][i]
            print(f"  {i:<6}{res['keep_tag'][i]:>9.3f}"
                  f"{res['keep_random'][i]:>9.3f}{b['mean']:>8.3f}"
                  f"{b['sd']:>7.3f}{b['min']:>8.3f}{b['max']:>8.3f}"
                  f"{b['above_3']:>6.0%}{b['below_3']:>6.0%}"
                  f"{res['spearman_excess_nmi'][i]:>10.2f}")
        for tag in ("most_moved", "most_spared"):
            print(f"  {tag}: " + "; ".join(
                f"L{r['layer']}h{r['head']} {r['excess']:+.3f} "
                f"(nmi {r['nmi_lang']:.3f})" for r in res[tag][:5]))
        print()

    import jax
    import jax.numpy as jnp
    import sae
    from pareto import parse_sae_name
    for run in cfg.sae:
        if (Path(run) / "meta.json").exists():
            meta = json.loads((Path(run) / "meta.json").read_text())
        else:
            m_, k_ = parse_sae_name(Path(run) / "params.npz")
            meta = {"m": m_, "topk": k_}
        params = {k_: jnp.asarray(v)
                  for k_, v in np.load(Path(run) / "params.npz").items()}
        topk, groups, gfn = (meta["topk"], meta.get("groups", 0),
                             meta.get("group_fn", "top1"))
        g = jax.jit(lambda p, X: sae.encode(p, X, topk, groups, gfn) > 0)
        act = lambda X: run_batched(lambda x: g(params, jnp.asarray(x)), X,
                                    cfg.batch)[0]
        c = changed
        Zc, Zi, Z1, Z2 = (act(X[c]) for X in (X_c, X_int, X_r1, X_r2))
        st, n_act = stay_rate(Zc, Zi)
        s1, _ = stay_rate(Zc, Z1)
        s2, _ = stay_rate(Zc, Z2)
        scored = n_act >= cfg.min_active
        st, s1, s2 = (np.where(scored, v, np.nan) for v in (st, s1, s2))
        ex, noise = excess(st, s1, s2)
        share = top_label_share(Zc, y[c], len(names))
        name = Path(run).name
        res = {"scored_latents": int(scored.sum()),
               "jaccard_tag": float(np.median(jaccard(Zc, Zi))),
               "jaccard_random": float(np.median(jaccard(Zc, Z1))),
               "stay_tag": float(np.nanmean(st)),
               "stay_random": float(np.nanmean(0.5 * (s1 + s2))),
               "all": describe(ex, noise),
               "spearman_excess_langshare": spearman(ex, share)}
        order = np.argsort(-np.where(np.isfinite(ex), ex, -np.inf))
        res["most_moved"] = [
            {"latent": int(j), "active_rows": int(n_act[j]),
             "stay_tag": float(st[j]), "stay_random": float(0.5 * (s1 + s2)[j]),
             "excess": float(ex[j]), "top_lang_share": float(share[j])}
            for j in order[:8]]
        fin = np.flatnonzero(np.isfinite(ex))
        res["most_spared"] = [
            {"latent": int(j), "active_rows": int(n_act[j]),
             "stay_tag": float(st[j]), "stay_random": float(0.5 * (s1 + s2)[j]),
             "excess": float(ex[j]), "top_lang_share": float(share[j])}
            for j in fin[np.argsort(ex[fin])][:8]]
        summary[name] = res
        a = res["all"]
        print(f"{name}: {res['scored_latents']} latents active on >= "
              f"{cfg.min_active} changed rows; reference noise {noise:.4f}")
        print(f"  support Jaccard (median): tag {res['jaccard_tag']:.3f}, "
              f"random shift {res['jaccard_random']:.3f}")
        print(f"  stay rate: tag {res['stay_tag']:.3f}, random "
              f"{res['stay_random']:.3f}; excess mean {a['mean']:+.3f} sd "
              f"{a['sd']:.3f} min {a['min']:+.3f} max {a['max']:+.3f}; "
              f"{a['above_3']:.0%} above 3 noise, {a['below_3']:.0%} below; "
              f"rho with top-language share "
              f"{res['spearman_excess_langshare']:+.2f}")
        for tag in ("most_moved", "most_spared"):
            print(f"  {tag}: " + "; ".join(
                f"{r['latent']} {r['excess']:+.3f} (lang share "
                f"{r['top_lang_share']:.2f})" for r in res[tag][:5]))
        print()

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

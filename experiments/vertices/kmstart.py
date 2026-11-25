"""A one-head seed pair started from spherical k-means.

On mc4_sent16, one trained head of 32 entries reproduces across seeds at
partition NMI 0.418. k-means at the same 32 cells reproduces at 0.607, and
spherical k-means at 0.732, at about the same fit
(experiments/seed-stability/notes.md, 2026-11-22). The archetypal SAE
(Fel et al. 2025) reports its stability with atoms started at fixed
k-means centroids, so its gain is confounded with its start. This separates
the start from training. The stock one-head recipe is started from each
seed's own spherical k-means, and both seeds also from one seed's start,
then scored against the stock pair and against k-means itself.

The start is fitted where the head can represent it exactly. The classifier
reads the input's shape u = x / |x| with a constant 1 appended, and its
logits are the product of two linear maps of that. With one factor on the
constant and the other on a centroid, the logits are
lam * c_j . (w * (u - ubar)), whose argmax is spherical k-means' nearest
centre in the frame w * (u - ubar): the shape, centred on its training mean
and weighted by the loss's sqrt(MSE weights).

The decoder is moved by the least change that makes each entry decode to
its cell's least-squares target, sum |x| x / sum |x|^2 (the head writes
|x| times its atom). The dictionary keeps its initial atoms. `lam` sets the
logits' scale: the start's median top-two gap matches a trained stock
head's (--margin-from), so the straight-through surrogate starts no
softer or harder than a trained head's.

  fit    spherical k-means per seed on the training rows that
         kmeans_seeds.py uses; per seed, the start (npz), with the fits'
         own agreement on the tail and the stock heads' agreement with them
  score  each trained head against its start and against the other runs,
         each pair beside a rows-shuffled null

  JAX_PLATFORMS=cpu uv run python experiments/vertices/kmstart.py fit \\
      --margin-from data/out/sonar/width_seeds/l1_h1_s42 \\
      --stock data/out/sonar/width_seeds/l1_h1_s42 \\
              data/out/sonar/width_seeds/l1_h1_s43 \\
      --out data/out/sonar/vertices/kmstart/fit

experiments/vertices/train_kmstart.py trains from a start, and
kmstart_seeds.sh queues the whole test.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Tuple  # noqa: E402

import numpy as np  # noqa: E402
from jaxtyping import Float, Int  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))
sys.path.insert(0, str(ROOT / "experiments" / "seed-stability"))

CACHE = ROOT / "data/sonar_embeddings/mc4_sent16.npy"
WEIGHTS = ROOT / "data/out/sonar/sentcache/mse_weights_med837.npy"


# ---- pure helpers -----------------------------------------------------------

def shape_frame(X: Float[np.ndarray, "n d"], w: Float[np.ndarray, "d"],
                ubar: Float[np.ndarray, "d"]) -> Float[np.ndarray, "n d"]:
    """w * (u - ubar) for the rows' shapes u = x / |x|: the frame the start
    is fitted in, a linear function of the shape the classifier reads."""
    X = X.astype(np.float64)
    U = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
    return (w * (U - ubar)).astype(np.float32)


def unit(F: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "n d"]:
    return F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-30)


def cell_targets(X: Float[np.ndarray, "n d"], lab: Int[np.ndarray, "n"],
                 k: int) -> Float[np.ndarray, "k d"]:
    """Each cell's least-squares atom for a head that writes |x| times its
    atom: sum |x| x / sum |x|^2 over the cell."""
    X = X.astype(np.float64)
    g = np.linalg.norm(X, axis=1)
    H = np.zeros((len(X), k))
    H[np.arange(len(X)), lab] = 1.0
    return (H.T @ (g[:, None] * X)) / (H.T @ (g * g))[:, None]


def start_weights(W: Float[np.ndarray, "2 1 k m"],
                  A: Float[np.ndarray, "k e"], Wd: Float[np.ndarray, "d e"],
                  C: Float[np.ndarray, "k d"], w: Float[np.ndarray, "d"],
                  ubar: Float[np.ndarray, "d"], T: Float[np.ndarray, "k d"],
                  lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """The classifier weight and decoder of the start. `W` is a one-head
    bilinear classifier (two factors) over the shape with one constant
    coordinate appended (m = d + 1); its logits become
    lam * C (w * (u - ubar)). `A` holds the head's atoms |D| and `Wd` the
    decoder; the decoder moves by the least change (in Frobenius norm) for
    which Wd A^T = T^T, so entry j decodes to T[j]."""
    n, h, k, m = W.shape
    d = C.shape[1]
    if n != 2 or h != 1 or m != d + 1:
        raise ValueError(f"the start needs one head of a two-factor classifier "
                         f"over d + 1 = {d + 1} inputs; got weight {W.shape}")
    a = (w * C).astype(np.float64)
    W0 = np.zeros((1, k, m))
    W0[0, :, :d] = a
    W0[0, :, d] = -(a @ ubar)
    W1 = np.zeros((1, k, m))
    W1[0, :, d] = lam
    A64, Wd64 = A.astype(np.float64), Wd.astype(np.float64)
    R = T.astype(np.float64).T - Wd64 @ A64.T                  # (d, k)
    Wd_new = Wd64 + R @ np.linalg.solve(A64 @ A64.T, A64)
    return (np.stack([W0, W1]).astype(W.dtype), Wd_new.astype(Wd.dtype))


def gaps(K: Float[np.ndarray, "n k"]) -> Float[np.ndarray, "n"]:
    """Each row's top-two logit gap."""
    top = np.sort(K, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def pair_nmi(a: Int[np.ndarray, "n"], b: Int[np.ndarray, "n"], k: int,
             rng: np.random.Generator) -> Tuple[float, float]:
    """Partition NMI of two labellings and of `a` against `b` shuffled."""
    from partition import nmi_matrix
    v = float(nmi_matrix(a[None], k, b[None], k)[0, 0])
    null = float(nmi_matrix(a[None], k, rng.permutation(b)[None], k)[0, 0])
    return v, null


# ---- the model --------------------------------------------------------------

def layer0_logits(module, X):
    """Layer 0's classifier logits for rows X, as the forward computes
    them (b, h, k)."""
    E, _ = module.encode(X, 0.0, None)
    Ein = module.constinput(E)
    U, _ = module.dictencs[0].gainshape_in(Ein)
    return module.dictencs[0].classifier(U)


def logits_of(model, params, X: np.ndarray, batch: int = 1024) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    f = jax.jit(lambda p, x: model.apply(p, x, method=layer0_logits))
    return np.concatenate([np.asarray(f(params, jnp.asarray(X[i:i + batch])))
                           for i in range(0, len(X), batch)])


def with_kmstart(base: type) -> type:
    """`base` whose `init` replaces a one-head model's classifier and
    decoder with the start in `km_start` (an npz from `fit`). It checks that
    the model then selects the start's cells on the probe rows and decodes
    each entry to its target. A resumed run restores its own checkpoint over
    whatever `init` built."""
    @dataclass
    class KMStartHyperparams(base):
        km_start: str = ""

        def init(self, model, save_each: int = 1000):
            state = super().init(model, save_each)
            if not self.km_start:
                return state
            import jax
            import jax.numpy as jnp
            from flax.core import freeze, unfreeze
            from flax.core.frozen_dict import FrozenDict
            if model.l != 1 or model.h != 1:
                raise ValueError("the k-means start is for one layer of one head")
            z = np.load(self.km_start)
            frozen = isinstance(state.params, FrozenDict)
            p = unfreeze(jax.tree_util.tree_map(np.asarray, state.params))
            cls = p["params"]["dictencs_0"]["classifier"]
            A = np.abs(p["params"]["dictencs_0"]["dict"]["weights"][0])
            W, Wd = start_weights(cls["weight"], A,
                                  p["params"]["decoder"]["weights"],
                                  z["centroids"], z["w"], z["ubar"],
                                  z["targets"], float(z["scale"]))
            cls["weight"] = W
            p["params"]["decoder"]["weights"] = Wd
            new = jax.tree_util.tree_map(jnp.asarray, p)
            state = state.replace(params=freeze(new) if frozen else new)
            K = logits_of(model, state.params, z["probe_x"])[:, 0]
            agree = float((K.argmax(1) == z["probe_lab"]).mean())
            dec = Wd.astype(np.float64) @ A.T.astype(np.float64)
            err = float(np.abs(dec - z["targets"].T).max()
                        / np.abs(z["targets"]).max())
            print(f"  k-means start from {self.km_start}: selects the start's "
                  f"cell on {agree:.4f} of {len(K)} probe rows; atoms decode "
                  f"to their targets within {err:.1e}; median gap "
                  f"{np.median(gaps(K)):.3g}")
            if agree < 0.99 or err > 1e-3:
                raise RuntimeError("the start did not take")
            return state

    KMStartHyperparams.__name__ = KMStartHyperparams.__qualname__ = \
        f"KMStart{base.__name__}"
    return KMStartHyperparams


# ---- fit and score ------------------------------------------------------------

def fit(cfg) -> None:
    from kmeans_seeds import assign_cos, spherical_kmeans, train_index
    from pareto import load_onto
    from partition import codes
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    w = np.sqrt(np.load(cfg.mse_weights).astype(np.float64))
    mm = np.load(cfg.cache, mmap_mode="r")
    Xf = np.asarray(mm[train_index(cfg.prefix, cfg.train_rows)], np.float32)
    Xt = np.asarray(mm[-cfg.tail_rows:], np.float32)
    Uf = Xf / np.linalg.norm(Xf, axis=1, keepdims=True)
    ubar = Uf.astype(np.float64).mean(0)
    Ff, Ft = shape_frame(Xf, w, ubar), shape_frame(Xt, w, ubar)
    Ffu, Ftu = unit(Ff), unit(Ft)

    model, raw, step = load_onto(cfg.margin_from, 0)
    Ks = logits_of(model, {"params": raw}, Xt)[:, 0]
    target = float(np.median(gaps(Ks)))
    print(f"stock head {Path(cfg.margin_from).name} step {step}: median top-two "
          f"gap {target:.4g} ({target / cfg.temperature:.1f} T)", flush=True)

    res = {"cache": cfg.cache, "train_rows": len(Xf), "tail_rows": len(Xt),
           "k": cfg.k, "seeds": cfg.seeds, "target_gap": target,
           "temperature": cfg.temperature, "fits": {}}
    labs = {}
    for s in cfg.seeds:
        C = spherical_kmeans(Ffu, cfg.k, s)
        lf, lt = assign_cos(Ffu, C), assign_cos(Ftu, C)
        T = cell_targets(Xf, lf, cfg.k)
        raw_gap = float(np.median(gaps(Ft @ C.T)))
        lam = target / raw_gap
        np.savez(out / f"start_s{s}.npz", centroids=C.astype(np.float32),
                 w=w.astype(np.float32), ubar=ubar.astype(np.float32),
                 targets=T.astype(np.float32), scale=np.float64(lam),
                 tail_labels=lt.astype(np.int16),
                 probe_x=Xt[:cfg.probe], probe_lab=lt[:cfg.probe])
        cnt = np.bincount(lf, minlength=cfg.k)
        res["fits"][s] = {"scale": lam, "raw_median_gap": raw_gap,
                          "largest_cell": float(cnt.max() / cnt.sum()),
                          "smallest_cell": float(cnt.min() / cnt.sum())}
        labs[s] = lt
        print(f"  fit seed {s}: cells {cnt.min()}..{cnt.max()} of {len(Xf)}, "
              f"scale {lam:.4g}", flush=True)
    rng = np.random.default_rng(0)
    res["fit_pairs"] = {}
    for a, b in itertools.combinations(cfg.seeds, 2):
        v, null = pair_nmi(labs[a], labs[b], cfg.k, rng)
        res["fit_pairs"][f"{a}-{b}"] = {"nmi": v, "null": null}
        print(f"  k-means seeds {a} and {b}: NMI {v:.4f} (null {null:.4f})")
    res["stock"] = {}
    for ckpt in cfg.stock:
        A, _, _, _ = codes(ckpt, 0, cfg.temperature, Xt, 256)
        res["stock"][Path(ckpt).name] = {
            str(s): pair_nmi(A[0], labs[s], cfg.k, rng)[0] for s in cfg.seeds}
        print(f"  stock {Path(ckpt).name} against the fits: "
              + ", ".join(f"{s} {v:.4f}" for s, v in
                          res["stock"][Path(ckpt).name].items()))
    (out / "summary.json").write_text(json.dumps(res, indent=1))


def score(cfg) -> None:
    from partition import codes
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.tail_rows:],
                   np.float32)
    rng = np.random.default_rng(0)
    heads, res = {}, {"runs": {}, "pairs": {}}
    for spec in cfg.runs:
        name, ckpt, start = spec.split(":")
        A, _, _, step = codes(ckpt, 0, cfg.temperature, X, 256)
        heads[name] = A[0]
        lab = np.load(start)["tail_labels"].astype(np.int64)
        v, null = pair_nmi(A[0], lab, cfg.k, rng)
        res["runs"][name] = {"ckpt": ckpt, "step": int(step), "start": start,
                             "nmi_start": v, "null": null}
        print(f"{name} (step {step}) against its start: NMI {v:.4f} "
              f"(null {null:.4f})", flush=True)
    for a, b in itertools.combinations(heads, 2):
        v, null = pair_nmi(heads[a], heads[b], cfg.k, rng)
        res["pairs"][f"{a} {b}"] = {"nmi": v, "null": null}
        print(f"{a} against {b}: NMI {v:.4f} (null {null:.4f})")
    if cfg.out:
        Path(cfg.out).write_text(json.dumps(res, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "score"):
        s = sub.add_parser(name)
        s.add_argument("--cache", default=str(CACHE))
        s.add_argument("--tail-rows", type=int, default=32768,
                       help="the cache's last rows, as partition.py scores")
        s.add_argument("--k", type=int, default=32)
        s.add_argument("--temperature", type=float, default=0.00015)
    f = sub.choices["fit"]
    f.add_argument("--mse-weights", default=str(WEIGHTS))
    f.add_argument("--prefix", type=int, default=12_800_000,
                   help="the training rows are drawn from this many")
    f.add_argument("--train-rows", type=int, default=131072)
    f.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    f.add_argument("--probe", type=int, default=512,
                   help="tail rows the trainer checks the start on")
    f.add_argument("--margin-from", required=True,
                   help="a trained stock one-head run whose median top-two "
                        "gap the start's logits are scaled to")
    f.add_argument("--stock", nargs="*", default=[],
                   help="stock one-head runs scored against the fits")
    f.add_argument("--out", required=True)
    s = sub.choices["score"]
    s.add_argument("--runs", nargs="+", required=True,
                   metavar="NAME:RUN_DIR:START_NPZ")
    s.add_argument("--out", default=None)
    cfg = p.parse_args()
    fit(cfg) if cfg.cmd == "fit" else score(cfg)


if __name__ == "__main__":
    main()

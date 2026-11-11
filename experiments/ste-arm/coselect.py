"""Are the atoms one sample selects together less alike than the
dictionary they come from?

`cossim_k` and `support` describe a dictionary as a whole, weighted by
usage at most. What a sample's reconstruction sums is narrower: the one
atom each head selects on that sample. If heads avoid one another's
directions on the same input, the atoms that actually meet are less
alike than the dictionary's average pair, and dictionary-wide figures
overstate the interference a sample sees. Costa et al. (2025) report
this for a matching-pursuit SAE: a more coherent dictionary than a
standard SAE's whose co-selected atoms are less so.

For an Ontologizer, over held-out rows, the cosine between the decoded
atoms of two different heads, averaged three ways:

  selected    over the pairs one row selects together
  dictionary  over all pairs of the two heads' atoms
  usage       over pairs drawn from each head's own usage frequencies,
              independently: what `selected` would be if heads chose
              without regard to one another. This is the null. It is
              computed exactly, and `--draws` permutations of each
              head's selections across rows give the spread `selected`
              is read against.

Head pairs are grouped as within each layer and across layers. Each is
reported for the absolute and the signed cosine, in two frames: the
decoded atoms as they are, and centered on their head's usage-weighted
mean atom. A non-negative dictionary gives every head a shared offset
that the first frame reads as likeness between any two atoms; the
second removes it, and is the frame in which a head's choice matters.

For an SAE (`--sae`), the same over its decoder directions: `selected`
over the pairs of latents active on one row, `dictionary` over all
pairs, and the null over active sets of the same size drawn from the
latents' firing frequencies. Its decoder bias plays the part of the
heads' mean atoms, so its one frame corresponds to the centered one.

  uv run python experiments/ste-arm/coselect.py \\
      --ckpt data/out/sonar/multilingual/ste_h76_init01 \\
      --sae data/out/sonar/sae_conv/m5120_k32/params.npz
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from jaxtyping import Bool, Float, Int  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def unit(A: Float[np.ndarray, "... d"]) -> Float[np.ndarray, "... d"]:
    """Rows scaled to unit length; a zero row stays zero."""
    n = np.linalg.norm(A, axis=-1, keepdims=True)
    return A / np.where(n > 0, n, 1.0)


def center(D: Float[np.ndarray, "s k d"], usage: Float[np.ndarray, "s k"]
           ) -> Float[np.ndarray, "s k d"]:
    """Each head's atoms less the head's usage-weighted mean atom."""
    return D - np.einsum("sk,skd->sd", usage, D)[:, None, :]


def usage_of(I: Int[np.ndarray, "s n"], k: int) -> Float[np.ndarray, "s k"]:
    """How often each head selects each entry."""
    return np.stack([np.bincount(row, minlength=k) for row in I]
                    ).astype(np.float64) / I.shape[1]


def head_pair_means(G: Float[np.ndarray, "sk sk"],
                    W: Float[np.ndarray, "s k"]) -> Float[np.ndarray, "s s"]:
    """For every pair of heads, the mean of `G` over their atoms under
    independent weights `W` (rows summing to 1): the usage null with the
    usage frequencies, the dictionary mean with uniform ones."""
    s, k = W.shape
    P = np.zeros((s, s * k), G.dtype)
    for i in range(s):
        P[i, i * k:(i + 1) * k] = W[i]
    return P @ G @ P.T


def selected_means(G: Float[np.ndarray, "sk sk"], I: Int[np.ndarray, "s n"],
                   k: int, chunk: int = 64) -> Float[np.ndarray, "s s"]:
    """For every pair of heads, the mean of `G` over the pairs of atoms
    the rows select together."""
    s, n = I.shape
    flat = (I + np.arange(s)[:, None] * k).T               # (n, s)
    out = np.zeros((s, s), np.float64)
    for a in range(0, n, chunk):
        f = flat[a:a + chunk]
        out += G[f[:, :, None], f[:, None, :]].sum(0, dtype=np.float64)
    return out / n


def pair_groups(layer: Int[np.ndarray, "s"]) -> Dict[str, Bool[np.ndarray, "s s"]]:
    """Which pairs of distinct heads each reported figure averages:
    within each layer, across layers, and all."""
    same = layer[:, None] == layer[None, :]
    off = ~np.eye(len(layer), dtype=bool)
    out = {f"layer {i}": same & off & (layer[:, None] == i)
           for i in range(int(layer.max()) + 1)}
    if not same.all():
        out["across layers"] = ~same
    out["all"] = off
    return {name: m for name, m in out.items() if m.any()}


def coselection(D: Float[np.ndarray, "s k d"], I: Int[np.ndarray, "s n"],
                layer: Int[np.ndarray, "s"], draws: int,
                rng: np.random.Generator, absolute: bool
                ) -> Dict[str, Dict[str, float]]:
    """`selected`, `dictionary` and the `usage` null with its spread over
    `draws` permutations, for each group of head pairs."""
    s, k, d = D.shape
    U = unit(D).reshape(s * k, d)
    G = U @ U.T
    if absolute:
        G = np.abs(G)
    usage = usage_of(I, k)
    S = selected_means(G, I, k)
    N = head_pair_means(G, usage)
    Dm = head_pair_means(G, np.full((s, k), 1.0 / k))
    perm = [selected_means(G, np.stack([rng.permutation(r) for r in I]), k)
            for _ in range(draws)]
    out = {}
    for name, m in pair_groups(layer).items():
        null = np.array([p[m].mean() for p in perm])
        out[name] = {"selected": float(S[m].mean()),
                     "dictionary": float(Dm[m].mean()),
                     "usage": float(N[m].mean()),
                     "usage_sd": float(null.std(ddof=1)) if draws > 1
                     else float("nan"),
                     "pairs": int(m.sum())}
    return out


def active_sets(Z: Float[np.ndarray, "n m"]) -> Tuple[Int[np.ndarray, "n a"],
                                                    Float[np.ndarray, "m"]]:
    """The latents active on each row, for a code with the same number
    active on every row, and each latent's firing frequency."""
    on = Z != 0
    a = int(on.sum(1).min())
    assert a == int(on.sum(1).max()), "rows differ in their active count"
    return np.argsort(~on, axis=1, kind="stable")[:, :a], on.mean(0)


def set_mean(G: Float[np.ndarray, "m m"], S: Int[np.ndarray, "n a"],
             chunk: int = 256) -> float:
    """Mean of `G` over the pairs of distinct members of each row's set."""
    n, a = S.shape
    tot = 0.0
    for i in range(0, n, chunk):
        f = S[i:i + chunk]
        B = G[f[:, :, None], f[:, None, :]]
        tot += float(B.sum(dtype=np.float64)
                     - np.trace(B, axis1=1, axis2=2).sum(dtype=np.float64))
    return tot / (n * a * (a - 1))


def draw_sets(freq: Float[np.ndarray, "m"], n: int, a: int,
              rng: np.random.Generator, chunk: int = 1024
              ) -> Int[np.ndarray, "n a"]:
    """`n` sets of `a` distinct latents, drawn without replacement in
    proportion to `freq` (Gumbel top-a)."""
    logp = np.log(np.maximum(freq, np.finfo(np.float64).tiny))
    logp = np.where(freq > 0, logp, -np.inf)
    out = []
    for i in range(0, n, chunk):
        g = logp + rng.gumbel(size=(min(chunk, n - i), len(freq)))
        out.append(np.argpartition(-g, a - 1, axis=1)[:, :a])
    return np.concatenate(out)


def sae_coselection(W: Float[np.ndarray, "m d"], Z: Float[np.ndarray, "n m"],
                    draws: int, rng: np.random.Generator, absolute: bool
                    ) -> Dict[str, float]:
    """The SAE's `selected`, `dictionary` and firing-frequency null."""
    U = unit(W)
    G = U @ U.T
    if absolute:
        G = np.abs(G)
    S, freq = active_sets(Z)
    m = len(G)
    null = np.array([set_mean(G, draw_sets(freq, len(S), S.shape[1], rng))
                     for _ in range(draws)])
    return {"selected": set_mean(G, S),
            "dictionary": float((G.sum(dtype=np.float64) - np.trace(G))
                                / (m * (m - 1))),
            "usage": float(null.mean()) if draws else float("nan"),
            "usage_sd": float(null.std(ddof=1)) if draws > 1
            else float("nan"),
            "active": int(S.shape[1]), "live": int((freq > 0).sum())}


def show(name: str, r: Dict[str, float]) -> None:
    z = (r["selected"] - r["usage"]) / r["usage_sd"] if r["usage_sd"] > 0 \
        else float("nan")
    print(f"    {name:<14} selected {r['selected']:+.4f}  usage "
          f"{r['usage']:+.4f} (sd {r['usage_sd']:.4f}, z {z:+.1f})  "
          f"dictionary {r['dictionary']:+.4f}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="*", default=[], help="Ontologizer runs")
    p.add_argument("--sae", nargs="*", default=[], help="sae.py params.npz")
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--rows", type=int, default=8192,
                   help="the cache's last rows, held out from training")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--draws", type=int, default=20,
                   help="draws of the null, for its spread")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON of the results")
    cfg = p.parse_args()

    arr = np.load(cfg.cache, mmap_mode="r")
    X = np.asarray(arr[len(arr) - cfg.rows:], np.float32)
    w_sqrt = np.sqrt(np.load(cfg.mse_weights).astype(np.float64)
                     ).astype(np.float32)
    rng = np.random.default_rng(cfg.seed)
    res = {"rows": cfg.rows, "cache": cfg.cache, "draws": cfg.draws,
           "onto": {}, "sae": {}}

    if cfg.ckpt:
        from headcontrib import atoms_and_codes
    for ckpt in cfg.ckpt:
        D, I, _, layer, step = atoms_and_codes(
            ckpt, cfg.step, cfg.temperature, X, w_sqrt, cfg.batch)
        D = D.astype(np.float32)
        k = D.shape[1]
        frames = {"decoded": D, "centered": center(D, usage_of(I, k)
                                                   ).astype(np.float32)}
        print(f"{Path(ckpt).name} step {step}: {len(D)} heads of {k} entries")
        out = {}
        for frame, A in frames.items():
            for kind, absolute in (("abs", True), ("signed", False)):
                r = coselection(A, I, layer, cfg.draws, rng, absolute)
                out[f"{frame}_{kind}"] = r
                print(f"  {frame}, {kind} cosine")
                for name, v in r.items():
                    show(name, v)
        res["onto"][ckpt] = {"step": int(step), **out}

    if cfg.sae:
        import jax.numpy as jnp
        import sae
    for path in cfg.sae:
        meta = json.loads((Path(path).parent / "meta.json").read_text())
        assert meta["topk"] and not meta.get("groups") \
            and not meta.get("block"), f"{path}: a flat top-k SAE is assumed"
        params = {key: jnp.asarray(v) for key, v in np.load(path).items()}
        Z = np.concatenate([
            np.asarray(sae.encode(params, jnp.asarray(X[i:i + 1024]),
                                  meta["topk"]))
            for i in range(0, len(X), 1024)])
        W = np.asarray(params["W_dec"]) * w_sqrt
        print(f"{Path(path).parent.name}: {W.shape[0]} latents, "
              f"{meta['topk']} active")
        out = {}
        for kind, absolute in (("abs", True), ("signed", False)):
            out[kind] = sae_coselection(W, Z, cfg.draws, rng, absolute)
            show(kind, out[kind])
        res["sae"][path] = out

    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

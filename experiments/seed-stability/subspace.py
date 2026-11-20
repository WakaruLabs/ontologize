"""Seed agreement as subspaces, with the data's own principal directions as
the reference.

Everything is in the frame the loss uses: coordinates scaled by the square
root of the per-coordinate MSE weights (`--mse-weights`).

--mode sae (Gerasimov et al. 2026's subspace transfer, with our control).
  Live latents of each SAE (any activation on the tail) are matched by
  decoder cosine: a latent is matched if some live latent of the other
  seed is within --theta (0.7, after Leask et al.). Each seed splits into
  its matched set S and unmatched set U, subsampled to one size (--draws
  random draws averaged). For a set X of decoder directions (rows, unit
  norm) and an orthonormal basis Q of r columns, EV(X, Q) is the share of
  ||X||_F^2 inside span(Q). Per seed s and kind (U or S):
    own      EV of X_s in its own top-r right singular subspace (the
             Eckart-Young ceiling)
    cross    EV of X_s in the other seed's top-r subspace for the same kind
    data     EV of X_s in the whitened data's top-r principal subspace
    bottom   the same in its bottom-r subspace
    chance   r/d, a random r-dimensional subspace
  "Reproducible subspaces" means cross well above data and chance.

--mode stack (Hu et al. 2021's normalized projection similarity).
  Per layer, the contribution x_hat_{l+1} - x_hat_l of each row (whitened,
  centered over rows); its covariance's eigenvectors give each seed an
  ordered basis. phi(A, B, i, j) = ||A_i^T B_j||_F^2 / min(i, j), A_i the
  top-i columns, is 1 for nested subspaces and max(i, j)/d in expectation
  for random ones. Per layer and rank r: phi between the seeds, and
  between each seed and the whitened data's top-r principal subspace, each
  also over chance (r/d). The full prefix grid phi(i, j) between the seeds
  goes to the npz.

  uv run python experiments/seed-stability/subspace.py --mode sae \\
      --a data/out/sonar/sae_conv/m5120_k32/params.npz \\
      --b data/out/sonar/sae_conv/m5120_k32_s43/params.npz
  uv run python experiments/seed-stability/subspace.py --mode stack \\
      --a data/out/sonar/multilingual/ste_h76_init01 \\
      --b data/out/sonar/multilingual/ste_h76_i01_s43 --step 369500

Writes <out>/subspace_<mode>.json and .npz.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from jaxtyping import Float  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def basis(X: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "d q"]:
    """Right singular vectors of a row matrix, largest first: the ordered
    orthonormal basis of the rows' span (q = min(n, d))."""
    return np.linalg.svd(X, full_matrices=False)[2].T


def eig_basis(C: Float[np.ndarray, "d d"]) -> Float[np.ndarray, "d d"]:
    """Eigenvectors of a symmetric matrix as columns, largest eigenvalue
    first."""
    lam, V = np.linalg.eigh(C)
    return V[:, ::-1]


def ev(X: Float[np.ndarray, "n d"], Q: Float[np.ndarray, "d r"]) -> float:
    """Share of ||X||_F^2 inside span(Q), Q orthonormal columns."""
    return float(((X @ Q) ** 2).sum() / (X ** 2).sum())


def phi(A: Float[np.ndarray, "d p"], B: Float[np.ndarray, "d q"],
        i: int, j: int) -> float:
    """||A_i^T B_j||_F^2 / min(i, j) for the top-i and top-j columns of two
    orthonormal bases (Hu et al. 2021, Eq. 4)."""
    return float(((A[:, :i].T @ B[:, :j]) ** 2).sum() / min(i, j))


def best_cos(D_a: Float[np.ndarray, "m d"],
             D_b: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "m"]:
    """Each unit row of D_a's best cosine with any unit row of D_b."""
    return np.concatenate([(D_a[i:i + 1024] @ D_b.T).max(1)
                           for i in range(0, len(D_a), 1024)])


def unit_rows(D: Float[np.ndarray, "m d"]) -> Float[np.ndarray, "m d"]:
    return D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)


def transfer(sets: dict, data_V: Float[np.ndarray, "d d"], ranks, draws: int,
             seed: int) -> dict:
    """EV curves for {(seed, kind): rows} with seeds 'a' and 'b' and kinds
    'U' and 'S'. Within each kind both seeds' sets are subsampled to the
    smaller size, `draws` times; values are means over draws."""
    d = data_V.shape[0]
    rng = np.random.default_rng(seed)
    out = {}
    for kind in ("U", "S"):
        n = min(len(sets[("a", kind)]), len(sets[("b", kind)]))
        acc = {}
        for _ in range(draws):
            sub = {s: sets[(s, kind)][rng.choice(len(sets[(s, kind)]), n,
                                                 replace=False)]
                   for s in ("a", "b")}
            Q = {s: basis(sub[s]) for s in sub}
            for s, o in (("a", "b"), ("b", "a")):
                for r in ranks:
                    if r > n:
                        continue
                    row = {"own": ev(sub[s], Q[s][:, :r]),
                           "cross": ev(sub[s], Q[o][:, :r]),
                           "data": ev(sub[s], data_V[:, :r]),
                           "bottom": ev(sub[s], data_V[:, -r:]),
                           "chance": r / d}
                    for key, v in row.items():
                        acc.setdefault((s, r, key), []).append(v)
        out[kind] = {"n": n, "curves": {
            s: {r: {key: float(np.mean(acc[(s, r, key)]))
                    for key in ("own", "cross", "data", "bottom", "chance")}
                for r in ranks if (s, r, "own") in acc}
            for s in ("a", "b")}}
    return out


def phi_table(A: Float[np.ndarray, "d d"], B: Float[np.ndarray, "d d"],
              data_V: Float[np.ndarray, "d d"], ranks) -> dict:
    """phi(r, r) between A and B and between each and the data, with each
    over chance r/d."""
    d = A.shape[0]
    rows = {}
    for r in ranks:
        v = {"ab": phi(A, B, r, r), "a_data": phi(A, data_V, r, r),
             "b_data": phi(B, data_V, r, r), "chance": r / d}
        v.update({f"{key}_x": v[key] / v["chance"]
                  for key in ("ab", "a_data", "b_data")})
        rows[r] = v
    return rows


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["sae", "stack"], required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--step", type=int, default=0, help="stack: 0 = latest")
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--eval-rows", type=int, default=32768)
    p.add_argument("--theta", type=float, default=0.7)
    p.add_argument("--draws", type=int, default=10)
    p.add_argument("--b-rows", type=int, default=2048, help="batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/out/sonar/seed_subspace")
    return p.parse_args()


def data_basis(X: Float[np.ndarray, "n d"], sw: Float[np.ndarray, "d"]):
    Xw = (X - X.mean(0)) * sw
    return eig_basis(Xw.T.astype(np.float64) @ Xw.astype(np.float64))


def run_sae(cfg, X, sw, data_V) -> dict:
    sys.path.insert(0, str(ROOT))
    import jax.numpy as jnp
    import sae
    D, live = {}, {}
    for s, path in (("a", cfg.a), ("b", cfg.b)):
        raw = np.load(path)
        params = {k: jnp.asarray(v) for k, v in raw.items()}
        meta = json.loads((Path(path).parent / "meta.json").read_text())
        fired = np.zeros(raw["W_dec"].shape[0], bool)
        for i in range(0, len(X), cfg.b_rows):
            Z = np.asarray(sae.encode(params, jnp.asarray(X[i:i + cfg.b_rows]),
                                      meta["topk"], meta.get("groups", 0),
                                      meta.get("group_fn", "top1")))
            fired |= (Z > 0).any(0)
        live[s] = fired
        D[s] = unit_rows(raw["W_dec"][fired] * sw)
    cos_ab, cos_ba = best_cos(D["a"], D["b"]), best_cos(D["b"], D["a"])
    match = {"a": cos_ab >= cfg.theta, "b": cos_ba >= cfg.theta}
    sets = {(s, kind): D[s][m if kind == "S" else ~m]
            for s, m in match.items() for kind in ("U", "S")}
    res = {"live": {s: int(v.sum()) for s, v in live.items()},
           "matched_share": {s: float(m.mean()) for s, m in match.items()},
           "theta": cfg.theta,
           "transfer": transfer(sets, data_V, RANKS, cfg.draws, cfg.seed)}
    print(f"live latents a {res['live']['a']}, b {res['live']['b']}; matched at "
          f"cos >= {cfg.theta}: a {res['matched_share']['a']:.3f}, "
          f"b {res['matched_share']['b']:.3f}")
    for kind, t in res["transfer"].items():
        print(f"{kind} (n = {t['n']} per seed), seed a:")
        print(f"  {'r':>4} {'own':>7} {'cross':>7} {'data':>7} {'bottom':>7} {'chance':>7}")
        for r, v in t["curves"]["a"].items():
            print(f"  {r:>4} " + " ".join(f"{v[k]:7.4f}" for k in
                                          ("own", "cross", "data", "bottom", "chance")))
    return res


def run_stack(cfg, X, sw, data_V) -> dict:
    sys.path.insert(0, str(ROOT))
    import jax
    import jax.numpy as jnp
    from pareto import load_onto

    def prefixes(module, Xb):
        """The decode after each layer, (l + 1, b, d), from the model's
        own selection rule."""
        E, _ = module.encode(Xb, 0.0, None)
        R = module.resid(E)
        E_in = module.constinput(E)
        out = [module.decode(R)]
        for i, de in enumerate(module.dictencs):
            U, G = de.gainshape_in(E_in)
            P = de.dict.cluster(de.classifier(U), cfg.temperature)
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
            out.append(module.decode(R))
            if i < module.l - 1:
                E_in = module.nextinput(Xb, R, P.reshape(P.shape[0], -1))
        return jnp.stack(out)

    bases, steps = {}, {}
    for s, path in (("a", cfg.a), ("b", cfg.b)):
        model, params, step = load_onto(path, cfg.step)
        steps[s] = int(step)
        f = jax.jit(lambda x, p=params, m=model: m.apply({"params": p}, x,
                                                          method=prefixes))
        l = model.l
        S1 = np.zeros((l, X.shape[1]))
        S2 = np.zeros((l, X.shape[1], X.shape[1]))
        for i in range(0, len(X), cfg.b_rows):
            Y = np.asarray(f(jnp.asarray(X[i:i + cfg.b_rows])), np.float64)
            W = np.diff(Y, axis=0) * sw                     # (l, b, d)
            S1 += W.sum(1)
            S2 += np.einsum("lbi,lbj->lij", W, W)
        n = len(X)
        C = S2 / n - np.einsum("li,lj->lij", S1 / n, S1 / n)
        bases[s] = [eig_basis(C[L]) for L in range(l)]
    res = {"steps": steps, "layers": {}}
    grid = {}
    print(f"phi(r, r) over chance (r/d), per layer: a vs b | a vs data | b vs data")
    for L in range(len(bases["a"])):
        A, B = bases["a"][L], bases["b"][L]
        res["layers"][L] = phi_table(A, B, data_V, RANKS)
        grid[f"layer{L}"] = np.array([[phi(A, B, i, j) for j in RANKS] for i in RANKS])
        print(f"  layer {L}: " + "  ".join(
            f"r{r}: {v['ab_x']:.1f}|{v['a_data_x']:.1f}|{v['b_data_x']:.1f}"
            for r, v in res["layers"][L].items() if r in (1, 8, 32, 128)))
    res["grid_ranks"] = list(RANKS)
    return res, grid


def main():
    cfg = parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.eval_rows:], np.float32)
    sw = np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
    data_V = data_basis(X, sw)
    meta = {"mode": cfg.mode, "a": cfg.a, "b": cfg.b, "cache": cfg.cache,
            "mse_weights": cfg.mse_weights, "rows": len(X)}
    if cfg.mode == "sae":
        res = run_sae(cfg, X, sw, data_V)
        np.savez(out / "subspace_sae.npz", data_V=data_V[:, :512])
    else:
        res, grid = run_stack(cfg, X, sw, data_V)
        np.savez(out / "subspace_stack.npz", **grid)
    (out / f"subspace_{cfg.mode}.json").write_text(
        json.dumps({**meta, **res}, indent=1))


if __name__ == "__main__":
    main()

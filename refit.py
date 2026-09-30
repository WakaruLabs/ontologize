"""Shrinkage / refit-FVU: decompose reconstruction error into selection
vs magnitude error (Gao et al. 2024's "refinement").

For each sample, freeze the code's support -- WHICH features are active --
and solve the whitened least-squares coefficients on that support. The
refit FVU isolates selection quality; the gap between raw and refit FVU
is the magnitude (shrinkage) error the activation rule imposes. ReLU+topk
systematically shrinks coefficients; group-top1 forces the winner's
magnitude; the Ontologizer's truncated deviation codes renormalize --
this benchmark says how much each costs relative to what its own support
could achieve. Coefficients are unconstrained (sign flips allowed), so
refit is an upper bound on what magnitude correction can recover.

SAE runs: support = active latents, directions = W_dec rows, offset =
b_dec. Group-softmax runs are skipped (dense support, nothing frozen).

Ontologizer (--onto, evaluated at each --ms deviation count): the output
is jointly linear in the flattened per-layer classifications (zero
residual seed + linear decoder), so the model has a fixed (l*h*k, d)
direction matrix G -- measured by decoding one-hot constant codes -- and
Y = P @ G exactly. Support = the top-m |p - E[p]| entries per head; raw
truncates the SOFT forward's P and decodes linearly, so it is the
NON-adaptive truncation -- pareto.py truncates inside the forward, where
later layers reclassify against the truncated prefix residual, and its
FVUs are accordingly lower (the gap measures the residual stack's
downstream compensation). Points with l*h*m >= d are skipped: an
underdetermined per-sample LS fits anything exactly, which is not a
shrinkage measurement (m=4 at K=640/1024 is already generous -- compare
against a random-support control before quoting it). Requires GPU JAX
for checkpoint restore.

  uv run python refit.py                          # all sae.py runs
  uv run python refit.py --onto data/out/sonar/multilingual/resid_nc

Scores the cache tail (the sae.py eval split). Writes <out>/refit.csv.
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import json
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

import sae


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sae", nargs="*", default=None,
                   help="params.npz paths (default: data/out/sonar/sae/*/)")
    p.add_argument("--onto", default=None, help="Ontologizer checkpoint dir")
    p.add_argument("--ms", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                   help="onto deviation counts per head")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--rows", type=int, default=32768,
                   help="cache tail rows (the sae.py eval split)")
    p.add_argument("--origin-rows", type=int, default=65536,
                   help="head rows for the onto E[p] origin")
    p.add_argument("--b", type=int, default=1024)
    p.add_argument("--max-support", type=int, default=2600,
                   help="skip codes with more active slots than this (the "
                        "per-sample LS is O(K^2) memory / O(K^3) time, and "
                        "refitting a near-dense support is uninformative)")
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--out", default="data/out/sonar/refit")
    return p.parse_args()


def refit_recon(X, dirs, idx, mask, base, w_sqrt, ridge=1e-6):
    """Batched masked weighted least squares on a frozen support.
    X (b, d); dirs (F, d) global directions; idx (b, K) selected feature
    ids; mask (b, K) with 0 = slot excluded (its coefficient is forced to
    0 via the ridge); base (d,) or (b, d) constant part. Returns the
    refit reconstruction (b, d)."""
    A = dirs[idx] * mask[..., None] * w_sqrt          # (b, K, d) whitened
    r = ((X - base) * w_sqrt)[..., None]              # (b, d, 1)
    G = A @ jnp.swapaxes(A, 1, 2)                     # (b, K, K)
    G = G + ridge * jnp.eye(A.shape[1])
    p = jnp.linalg.solve(G, A @ r)                    # (b, K, 1)
    return base + jnp.einsum("bko,bkd->bd", p * mask[..., None],
                             dirs[idx])


def eff_batch(K, d, b, budget=2 << 30):
    """Largest batch whose per-sample LS workspace (the (K, K) normal
    matrix plus the (K, d) design and its whitened copy) fits the budget."""
    per = 4 * K * (K + 2 * d)
    return max(1, min(b, budget // per))


def fvu_pair(X, recon_raw, recon_fit, w, base_w):
    e_raw = (((recon_raw - X) ** 2) * w).mean()
    e_fit = (((recon_fit - X) ** 2) * w).mean()
    return float(e_raw / base_w), float(e_fit / base_w)


def sae_supports(params, X, topk, groups, group_fn):
    """(z, idx, mask): the activation rule's support, fixed-width."""
    z = sae.encode(params, X, topk, groups, group_fn)
    if groups:  # one winner slot per group (silent groups masked out)
        m = z.shape[-1]
        g = z.reshape(*z.shape[:-1], groups, m // groups)
        off = jnp.arange(groups) * (m // groups)
        idx = g.argmax(-1) + off
        mask = (jnp.take_along_axis(g, (idx - off)[..., None], -1)
                .squeeze(-1) > 0).astype(jnp.float32)
    else:
        vals, idx = jax.lax.top_k(z, topk)
        mask = (vals > 0).astype(jnp.float32)
    return z, idx, mask


def onto_linear_model(ckpt, step, temperature):
    """acts fn + the fixed direction matrix G with Y = P_flat @ G."""
    from autointerp import onto_acts_fn
    acts, embed, F, meta = onto_acts_fn(ckpt, step, temperature)
    l, h, k = meta["l"], meta["h"], meta["k"]
    codes = np.eye(F, dtype=np.float32).reshape(F, l, h, k)
    G = jnp.asarray(embed(codes))                     # (F, d)
    return acts, G, meta


def onto_supports(P, origin, m, l, h, k):
    """Top-m |p - origin| entries per head: flat ids + the raw truncated
    (renormalized, pareto convention) code."""
    Ph = P.reshape(-1, l * h, k)
    D = jnp.abs(Ph - origin.reshape(1, l * h, k))
    _, ki = jax.lax.top_k(D, m)                        # (b, l*h, m)
    idx = (jnp.arange(l * h)[None, :, None] * k + ki).reshape(P.shape[0], -1)
    thr = jax.lax.top_k(D, m)[0][..., -1:]
    kept = D >= thr
    Pt = jnp.where(kept, Ph, origin.reshape(1, l * h, k))
    Pt = Pt / (Pt.sum(-1, keepdims=True) + 1e-9)
    return idx, Pt.reshape(P.shape[0], -1)


def main():
    cfg = parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    mm = np.load(cfg.cache, mmap_mode="r")
    n = cfg.rows // cfg.b * cfg.b
    X_all = np.asarray(mm[-n:], dtype=np.float32)
    w = np.load(cfg.mse_weights) if cfg.mse_weights else np.ones(mm.shape[1])
    w_sqrt = jnp.asarray(np.sqrt(w), jnp.float32)
    wj = jnp.asarray(w, jnp.float32)
    base_w = float((X_all.var(0) * w).mean())
    rows = []

    paths = cfg.sae
    if paths is None:
        paths = sorted(Path("data/out/sonar/sae").glob("*/params.npz"))
    for path in paths:
        meta_path = Path(path).parent / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            topk, groups, gfn = meta["topk"], meta["groups"], meta["group_fn"]
        else:
            from pareto import parse_sae_name
            _, topk = parse_sae_name(path)
            groups, gfn = 0, "top1"
        if groups and gfn == "softmax":
            print(f"skip {path} (dense softmax support)")
            continue
        if not topk and not groups:
            print(f"skip {path} (L1 mode: variable support)")
            continue
        K_sup = groups if groups else topk
        if K_sup > cfg.max_support:
            print(f"skip {path} (support {K_sup} > --max-support)")
            continue
        params = {k_: jnp.asarray(v) for k_, v in np.load(path).items()}

        @jax.jit
        def run(X):
            z, idx, mask = sae_supports(params, X, topk, groups, gfn)
            raw = sae.decode(params, z)
            fit = refit_recon(X, params["W_dec"], idx, mask,
                              params["b_dec"], w_sqrt, cfg.ridge)
            return (((raw - X) ** 2) * wj).mean(), \
                   (((fit - X) ** 2) * wj).mean()

        be = eff_batch(K_sup, X_all.shape[1], cfg.b)
        e = np.mean([np.asarray(run(jnp.asarray(X_all[i:i + be])))
                     for i in range(0, n - be + 1, be)], 0)
        name = Path(path).parent.name
        rows.append((f"sae {name}", e[0] / base_w, e[1] / base_w))
        print(f"sae {name}: raw {rows[-1][1]:.4f}  refit {rows[-1][2]:.4f}")

    if cfg.onto:
        acts, G, meta = onto_linear_model(cfg.onto, cfg.step, cfg.temperature)
        l, h, k = meta["l"], meta["h"], meta["k"]
        acc = np.zeros(l * h * k)
        nb = 0
        for i in range(0, min(cfg.origin_rows, mm.shape[0]), cfg.b):
            A = acts(jnp.asarray(np.asarray(mm[i:i + cfg.b], np.float32)))
            acc += np.asarray(A).mean(0)
            nb += 1
        origin = (acc / nb).reshape(l * h, k)
        origin = jnp.asarray(origin / origin.sum(-1, keepdims=True))
        c = (origin.reshape(-1) @ G)[None]            # pinned-code decode

        d_out = X_all.shape[1]
        for m in cfg.ms:
            if not 0 < m < k or l * h * m > cfg.max_support:
                continue
            if l * h * m >= d_out:
                # K >= d: the per-sample LS is underdetermined, so refit
                # is trivially exact (or numerically garbage) regardless
                # of direction quality -- not a shrinkage measurement
                print(f"skip onto m={m} (support {l * h * m} >= d={d_out})")
                continue

            @jax.jit
            def run(X, m=m):
                P = acts(X)
                idx, Pt = onto_supports(P, origin, m, l, h, k)
                raw = Pt @ G
                base = c + jnp.zeros_like(X)
                # kept slots refit their deviation from the origin
                fit = refit_recon(X, G, idx,
                                  jnp.ones(idx.shape, jnp.float32),
                                  base, w_sqrt, cfg.ridge)
                return (((raw - X) ** 2) * wj).mean(), \
                       (((fit - X) ** 2) * wj).mean()

            be = eff_batch(l * h * m, X_all.shape[1], cfg.b)
            e = np.mean([np.asarray(run(jnp.asarray(X_all[i:i + be])))
                         for i in range(0, n - be + 1, be)], 0)
            rows.append((f"onto dev m={m}", e[0] / base_w, e[1] / base_w))
            print(f"onto dev m={m}: raw {rows[-1][1]:.4f}  "
                  f"refit {rows[-1][2]:.4f}")

    print(f"\n{'point':<28} {'FVU raw':>9} {'FVU refit':>10} {'magn share':>11}")
    for name, raw, fit in rows:
        share = (raw - fit) / raw if raw else 0.0
        print(f"{name:<28} {raw:>9.4f} {fit:>10.4f} {share:>11.3f}")
    with open(out / "refit.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["point", "fvu_raw", "fvu_refit", "magnitude_share"])
        for name, raw, fit in rows:
            wr.writerow([name, raw, fit, (raw - fit) / raw if raw else 0.0])
    print(f"-> {out / 'refit.csv'}")


if __name__ == "__main__":
    main()

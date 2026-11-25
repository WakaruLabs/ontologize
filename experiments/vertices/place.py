"""Where a head's vertices sit: at its cells' centroids, at extremes of the
data, or outside it.

A head's assignment lies on a simplex whose vertices are its atoms. Under a
hard code each row writes one vertex; under a soft code rows write mixtures,
and a vertex is written only when an intervention forces it. This reads a
trained model as it is, retraining nothing, and asks where those vertices
sit.

For each head, with every other head, layer and gain held as the model has
them, the head's target on row n is the final residual plus the head's own
write, t_n = r_n + g_n w_n, in the whitened frame. Per unit of gain it is
z_n = t_n / g_n, and rows weigh g_n^2. A row belongs to the cell of the entry
its head's argmax selects.

Atoms are read about the head's mass-weighted mean atom abar, and targets
about the head's one-atom least-squares write c. A vector shared by every
atom of a head is a component any head can carry, so it is reported
separately (`common`) and not counted as placement.

  centroid  the cell's least-squares atom, mu = sum g t / sum g^2, which the
            entry would write if each row wrote one atom. Under a hard
            decode, atom a costs mass |a - mu|^2 more than mu does, with
            mass = sum g^2 (`price`, here of a - abar against mu - c)
  off       |(a - abar) - (mu - c)| over the cell's spread, the
            g^2-weighted RMS of z about mu; `off_raw` without the centring
  common    |abar - c| over the head's RMS spread
  beyond    along the ray from c in the direction a - abar: the weighted
            share of the cell's rows that project past c + (a - abar).
            That is about half for an atom at a symmetric cell's centroid,
            near 0 for one at the cell's edge, and 0 for one past every row
  reach     that projection over the cell's 95th-percentile projection: 1
            at the cell's trimmed edge, above 1 past it
  extreme   the trimmed extreme: the weighted mean of z over the cell's
            rows projecting between the 80th and 95th percentiles. Its
            hard-decode price is given as FVU, per head; per-head prices
            do not add across heads

Two tests of "in distribution", in the whitened frame:

  nn   a row's distance to its nearest held-out data row (the query rows
       are excluded from the reference set), each read as a ratio to the
       real rows' own median. Rows scored:
       - the query rows themselves;
       - the model's reconstructions;
       - the reconstructions with one head per row forced to another
         entry's atom, or to that entry's trimmed extreme. Forcing is
         frozen: nothing downstream reclassifies;
       - for a soft model, one head forced to its own argmax vertex, and
         the end-to-end argmax decode;
       - each prefix's norm against the rows'.
  ood  the archetypal SAE's score (Fel et al. 2025): one minus an atom's
       best cosine to a data row. Here it compares each atom's offset from
       its head's centre with the offsets of the layer's input rows from
       their mean. Random directions with the rows' covariance are scored
       beside it, and at layer 0 an SAE's decoder directions too (`--sae`)

  JAX_PLATFORMS=cpu uv run python experiments/vertices/place.py \\
      --ckpt data/out/sonar/multilingual/ste_h76_init01 \\
      --sae data/out/sonar/sae_conv/m5120_k32 --out data/out/sonar/vertices
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from jaxtyping import Float, Int  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BAND = (0.80, 0.95)


# ---- pure helpers -----------------------------------------------------------

def centroids(t: Float[np.ndarray, "n d"], g: Float[np.ndarray, "n"],
              sel: Int[np.ndarray, "n"], k: int
              ) -> Tuple[Float[np.ndarray, "k"], Float[np.ndarray, "k d"],
                         Float[np.ndarray, "k"]]:
    """Per entry of one head: the mass sum g^2 of its cell, its
    least-squares atom mu = sum g t / sum g^2, and the g^2-weighted RMS of
    z = t / g about mu. An empty cell gives mass 0 and NaNs."""
    oh = np.zeros((len(sel), k))
    oh[np.arange(len(sel)), sel] = 1.0
    g = g.astype(np.float64)
    t = t.astype(np.float64)
    mass = oh.T @ (g * g)
    s1 = oh.T @ (g[:, None] * t)
    ss = oh.T @ (t * t).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = s1 / mass[:, None]
        sse = ss - mass * (mu * mu).sum(1)
        spread = np.sqrt(np.maximum(sse, 0.0) / mass)
    return mass, mu, spread


def price(mass: Float[np.ndarray, "..."], mu: Float[np.ndarray, "... d"],
          a: Float[np.ndarray, "... d"]) -> Float[np.ndarray, "..."]:
    """Squared error a hard decode adds when the entry writes `a` rather
    than its cell's least-squares atom `mu`: mass |a - mu|^2. The cross
    terms over the cell's rows vanish at mu."""
    return mass * ((a - mu) ** 2).sum(-1)


def wquantile(x: Float[np.ndarray, "n"], w: Float[np.ndarray, "n"],
              q: float) -> float:
    """The smallest x whose weighted cumulative share reaches q."""
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(cw, q * cw[-1]), len(x) - 1)])


def ray(z: Float[np.ndarray, "n d"], w: Float[np.ndarray, "n"],
        c: Float[np.ndarray, "d"], a: Float[np.ndarray, "d"],
        band: Tuple[float, float] = BAND
        ) -> Tuple[float, float, Float[np.ndarray, "d"]]:
    """Along the ray from the head's centre `c` through the atom `a`: the
    weighted share of the cell's rows `z` that project past the atom, the
    atom's reach (its projection over the rows' `band[1]` quantile), and
    the trimmed extreme, the weighted mean of the rows between the band's
    quantiles. NaNs when the atom sits at the centre, or when the rows do
    not extend along the ray."""
    v = a - c
    s_a = float(np.linalg.norm(v))
    nan = (np.nan, np.nan, np.full_like(c, np.nan))
    if s_a == 0.0 or len(z) == 0:
        return nan
    s = (z - c) @ (v / s_a)
    beyond = float(w[s > s_a].sum() / w.sum())
    lo, hi = wquantile(s, w, band[0]), wquantile(s, w, band[1])
    inband = (s >= lo) & (s <= hi)
    if hi <= 0.0 or not inband.any():
        return beyond, np.nan, np.full_like(c, np.nan)
    e = (w[inband, None] * z[inband]).sum(0) / w[inband].sum()
    return beyond, s_a / hi, e


def nn_dist(Q: Float[np.ndarray, "m d"], R: Float[np.ndarray, "N d"],
            chunk: int = 512) -> Float[np.ndarray, "m"]:
    """Each query row's Euclidean distance to its nearest reference row."""
    R = R.astype(np.float64)
    rr = (R * R).sum(1)
    out = np.empty(len(Q))
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk].astype(np.float64)
        d2 = (q * q).sum(1)[:, None] + rr[None, :] - 2.0 * q @ R.T
        out[i:i + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def best_cos(A: Float[np.ndarray, "a d"], R: Float[np.ndarray, "N d"],
             chunk: int = 2048) -> Float[np.ndarray, "a"]:
    """Each vector's largest signed cosine with any reference row."""
    Rn = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    return np.concatenate([(An[i:i + chunk] @ Rn.T).max(1)
                           for i in range(0, len(An), chunk)])


def quartiles(x: Float[np.ndarray, "n"]) -> Dict[str, float]:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    q = np.quantile(x, [0.25, 0.5, 0.75])
    return {"n": int(len(x)), "q25": float(q[0]), "median": float(q[1]),
            "q75": float(q[2])}


# ---- the model --------------------------------------------------------------

def run_temperature(ckpt: str) -> float:
    """The temperature a run ended at: its last `env_config`'s
    `temperature_end`, else its `temperature`."""
    hyper = {}
    for line in (Path(ckpt) / "log.jsonl").read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") == "env_config":
            hyper = rec.get("hyper", hyper)
    t = hyper.get("temperature_end") or hyper.get("temperature")
    if t is None:
        raise ValueError(f"{ckpt}: no temperature in log.jsonl; pass one")
    return float(t)


def model_view(ckpt: str, step: int, T: float, X: Float[np.ndarray, "n d"],
               w_sqrt: Float[np.ndarray, "d"], batch: int, keep_p: bool):
    """Everything the measures read, in the whitened frame: the decoded
    atoms (l, h, k, d), each head's argmax entry (n, l, h), the soft
    assignments when `keep_p` (n, l, h, k), the layers' gains (n, l), each
    layer's input residual x - x_hat_l (n, l, d), the reconstruction (n, d),
    the end-to-end argmax reconstruction (n, d), and the step loaded."""
    import jax
    import jax.numpy as jnp
    from pareto import load_onto

    model, raw, used = load_onto(ckpt, step)
    if model.scaled or model.fiber_rank:
        raise ValueError(f"{ckpt}: a router or fibers break "
                         "contribution = gain x atom")
    params = {"params": raw}
    k = model.k

    def atoms(module):
        zero = module.decode(jnp.zeros((1, module.e_lat), module.dtype))
        out = []
        for de in module.dictencs:
            h = de.dict.h
            P = jnp.zeros((k, h, k), module.dtype)
            P = P.at[jnp.arange(k), :, jnp.arange(k)].set(1.0)
            Y = de.dict.hfwd(P)
            Yh = Y[:, None] * jnp.eye(h, dtype=Y.dtype)[None, :, :, None]
            out.append(module.decode(de.place(de.dict.combine(Yh))) - zero)
        return jnp.stack(out).transpose(0, 2, 1, 3)          # (l, h, k, d)

    def forward(module, Xb, hard):
        E, _ = module.encode(Xb, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        Ps, Gs, pre = [], [], []
        for i, de in enumerate(module.dictencs):
            U, G = de.gainshape_in(Ein)
            K = de.classifier(U)
            P = (jax.nn.one_hot(jnp.argmax(K, -1), k, dtype=K.dtype) if hard
                 else de.dict.cluster(K, T))
            pre.append(module.decode(R))
            Ps.append(P)
            Gs.append(jnp.ones(Xb.shape[:1], module.dtype) if G is None
                      else G[:, 0])
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
            if i < module.l - 1:
                Ein = module.nextinput(Xb, R, P.reshape(Xb.shape[0], -1))
        return (module.decode(R), jnp.stack(Ps, 1), jnp.stack(Gs, 1),
                jnp.stack(pre, 1))

    A = np.asarray(jax.jit(lambda p: model.apply(p, method=atoms))(params)
                   ) * w_sqrt
    fwd = jax.jit(lambda p, x, hard: model.apply(p, x, hard, method=forward),
                  static_argnums=2)
    sel, Ps, Gs, pre, Ys, Yh = [], [], [], [], [], []
    for i in range(0, len(X), batch):
        xb = jnp.asarray(X[i:i + batch])
        Y, P, G, Pre = fwd(params, xb, False)
        P = np.asarray(P)
        sel.append(P.argmax(-1).astype(np.int16))
        if keep_p:
            Ps.append(P.astype(np.float32))
        Gs.append(np.asarray(G))
        pre.append((X[i:i + batch, None] - np.asarray(Pre)) * w_sqrt)
        Ys.append(np.asarray(Y) * w_sqrt)
        Yh.append(np.asarray(fwd(params, xb, True)[0]) * w_sqrt)
    return dict(A=A.astype(np.float32), sel=np.concatenate(sel),
                P=np.concatenate(Ps) if keep_p else None,
                G=np.concatenate(Gs).astype(np.float32),
                Yin=np.concatenate(pre).astype(np.float32),
                Y=np.concatenate(Ys).astype(np.float32),
                Yhard=np.concatenate(Yh).astype(np.float32), step=used)


def head_write(view: dict, l: int, i: int, rows: slice = slice(None)
               ) -> Float[np.ndarray, "n d"]:
    """Head i of layer l's write per row, gain included: its soft mixture
    when the view kept assignments, else the selected atom."""
    A, g = view["A"][l, i], view["G"][rows, l]
    if view["P"] is not None:
        return g[:, None] * (view["P"][rows, l, i] @ A)
    return g[:, None] * A[view["sel"][rows, l, i].astype(np.int64)]


# ---- the measures -----------------------------------------------------------

def place_layer(view: dict, Xw: Float[np.ndarray, "n d"], l: int,
                var: float, min_mass_rows: int, band=BAND) -> dict:
    """The per-entry measures for every head of layer l, summarized over
    the layer's used entries. Each entry's trimmed extreme and its
    centroid are returned too, in atom coordinates (each read about the
    head's centre c and placed about its mean atom abar), for the forcing
    tests; entries with fewer than `min_mass_rows` rows read NaN.

    Atoms and centroids are compared about their own means: the head's
    mass-weighted mean atom abar and its one-atom least-squares write c.
    A vector added to every atom of a head adds g times it to every row,
    which any head, or a share of each, could carry (the mean-atom gauge).
    So a residual component proportional to the gain moves each head's own
    least-squares atoms by the whole of it, and refitting the layer jointly
    removes it once. Its size per head is reported as `common`, and the
    uncentred distance as `off_raw`. Per-head prices are not additive
    across heads for the same reason."""
    A = view["A"][l]
    h, k, d = A.shape
    resid = Xw - view["Y"]
    g = view["G"][:, l].astype(np.float64)
    keep = g > 0
    rows = {key: [] for key in ("off", "off_raw", "beyond", "reach",
                                "atom_over_mu", "price_atom",
                                "price_extreme")}
    E = np.full_like(A, np.nan)
    M = np.full_like(A, np.nan)
    head_price, common, common_price = [], [], []
    for i in range(h):
        sel = view["sel"][:, l, i].astype(np.int64)
        t = resid + head_write(view, l, i)
        mass, mu, spread = centroids(t[keep], g[keep], sel[keep], k)
        z = t[keep] / g[keep, None]
        w = g[keep] ** 2
        live = mass > 0
        c = (mass[live, None] * mu[live]).sum(0) / mass[live].sum()
        abar = (mass[live, None] * A[i, live]).sum(0) / mass[live].sum()
        rms = np.sqrt((mass[live] * spread[live] ** 2).sum()
                      / mass[live].sum())
        common.append(np.linalg.norm(abar - c) / rms)
        common_price.append(mass[live].sum()
                            * ((abar - c) ** 2).sum() / var)
        counts = np.bincount(sel[keep], minlength=k)
        hp = 0.0
        for j in range(k):
            if counts[j] < min_mass_rows:
                continue
            m = sel[keep] == j
            at, mt = A[i, j] - abar, mu[j] - c
            beyond, reach, e = ray(z[m], w[m], c, c + at, band)
            E[i, j] = abar + (e - c)
            M[i, j] = abar + mt
            rows["off"].append(np.linalg.norm(at - mt) / spread[j])
            rows["off_raw"].append(np.linalg.norm(A[i, j] - mu[j])
                                   / spread[j])
            rows["beyond"].append(beyond)
            rows["reach"].append(reach)
            rows["atom_over_mu"].append(np.linalg.norm(at)
                                        / np.linalg.norm(mt))
            pa = float(price(mass[j], mt, at))
            pe = float(price(mass[j], mu[j], e)) if np.isfinite(e).all() \
                else np.nan
            rows["price_atom"].append(pa / var)
            rows["price_extreme"].append(pe / var)
            if np.isfinite(pe):
                hp += (pe - pa) / var
        head_price.append(hp)
    out = {key: quartiles(v) for key, v in rows.items()}
    out["extreme_dfvu_per_head"] = quartiles(head_price)
    out["common"] = quartiles(common)
    out["common_price_per_head"] = quartiles(common_price)
    return out, E, M


def forcing(view: dict, E_all: Dict[int, np.ndarray], Xw: np.ndarray,
            queries: int, rng: np.random.Generator) -> dict:
    """Nearest-neighbour distances of real, reconstructed and forced rows
    to the held-out rows past the queries, as ratios to the real rows'
    median; and each prefix's norm against the rows'."""
    Q, R = slice(0, queries), slice(queries, None)
    ref = Xw[R]
    soft = view["P"] is not None
    real = nn_dist(Xw[Q], ref)
    base = float(np.median(real))
    res = {"queries": queries, "reference_rows": int(len(ref)),
           "real_median_nn": base,
           "real": quartiles(real / base),
           "recon": quartiles(nn_dist(view["Y"][Q], ref) / base)}
    if soft:
        res["argmax_decode"] = quartiles(
            nn_dist(view["Yhard"][Q], ref) / base)
    mean = Xw[R].mean(0)
    rnorm = np.linalg.norm(Xw[R] - mean, axis=1)
    res["norm_real"] = quartiles(rnorm / np.median(rnorm))
    res["norm_recon"] = quartiles(
        np.linalg.norm(view["Y"][Q] - mean, axis=1) / np.median(rnorm))
    if soft:
        res["norm_argmax_decode"] = quartiles(
            np.linalg.norm(view["Yhard"][Q] - mean, axis=1)
            / np.median(rnorm))
    L, h, k, _ = view["A"].shape
    n = queries
    for l in range(L):
        heads = rng.integers(0, h, n)
        cur = view["sel"][Q, l][np.arange(n), heads].astype(np.int64)
        other = (cur + rng.integers(1, k, n)) % k
        g = view["G"][Q, l][:, None]
        A = view["A"][l]
        own = np.stack([head_write(view, l, int(i), slice(r, r + 1))[0]
                        for r, i in enumerate(heads)])
        Yq = view["Y"][Q]
        forced = Yq - own + g * A[heads, other]
        ext = E_all[l][heads, other]
        ok = np.isfinite(ext).all(1)
        forced_e = Yq[ok] - own[ok] + g[ok] * ext[ok]
        entry = {"atom": quartiles(nn_dist(forced, ref) / base),
                 "extreme": quartiles(nn_dist(forced_e, ref) / base),
                 "extreme_rows": int(ok.sum()),
                 "norm_atom": quartiles(np.linalg.norm(forced - mean, axis=1)
                                        / np.median(rnorm))}
        if soft:
            vert = Yq - own + g * A[heads, cur]
            entry["own_vertex"] = quartiles(nn_dist(vert, ref) / base)
        res[f"layer {l}"] = entry
    return res


def shaped_random(ref: Float[np.ndarray, "N d"], m: int,
                  rng: np.random.Generator, mix: int = 512
                  ) -> Float[np.ndarray, "m d"]:
    """`m` random directions with the reference rows' covariance: Gaussian
    combinations of `mix` of the rows at a time. A direction the data
    spreads along has a high best cosine to some row whatever it means, so
    this is the null an atom is read against, not an isotropic one."""
    idx = rng.integers(0, len(ref), (m, mix))
    return np.einsum("mj,mjd->md", rng.normal(size=(m, mix)), ref[idx])


def ood(view: dict, Xw: np.ndarray, sae: Optional[str],
        w_sqrt: np.ndarray, ref_rows: int, rng: np.random.Generator) -> dict:
    """1 - best cosine of each atom's offset from its head's centre to the
    layer input's centred rows, per layer, beside random directions with
    the rows' covariance and (layer 0) an SAE's decoder directions."""
    L, h, k, d = view["A"].shape
    res = {}
    for l in range(L):
        Yin = view["Yin"][:ref_rows, l]
        ref = Yin - Yin.mean(0)
        counts = np.stack([np.bincount(view["sel"][:, l, i].astype(np.int64),
                                       minlength=k) for i in range(h)])
        use = counts / counts.sum(1, keepdims=True)
        A = view["A"][l]
        centred = (A - (use[..., None] * A).sum(1, keepdims=True))
        live = counts.reshape(-1) > 0
        atoms = centred.reshape(-1, d)[live]
        s_atoms = 1.0 - best_cos(atoms, ref)
        s_rand = 1.0 - best_cos(shaped_random(ref, 1024, rng), ref)
        res[f"layer {l}"] = {
            "atoms": quartiles(s_atoms), "atoms_mean": float(s_atoms.mean()),
            "random": quartiles(s_rand), "random_mean": float(s_rand.mean())}
    if sae:
        W = np.load(Path(sae) / "params.npz")["W_dec"] * w_sqrt
        ref = Xw[:ref_rows] - Xw[:ref_rows].mean(0)
        s = 1.0 - best_cos(W, ref)
        res["sae layer 0"] = {"atoms": quartiles(s),
                              "atoms_mean": float(np.mean(s))}
    return res


def decomposition_error(view: dict, rows: int = 256) -> float:
    """Max relative error of sum over layers and heads of gain x mixture
    of decoded atoms against the reconstruction less decode(0): the check
    that gain x atom decomposes this model."""
    L, h, k, d = view["A"].shape
    tot = np.zeros((rows, d))
    for l in range(L):
        for i in range(h):
            tot += head_write(view, l, i, slice(0, rows))
    zero = view["Y"][:rows] - tot
    # decode(0) is one vector; every row must leave the same remainder
    return float(np.abs(zero - zero.mean(0)).max()
                 / np.abs(view["Y"][:rows]).max())


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--temperature", type=float, default=None,
                   help="default: the run's final temperature (log.jsonl)")
    p.add_argument("--eval-cache",
                   default="data/sonar_embeddings/mc4_fresh.npy")
    p.add_argument("--rows", type=int, default=32768,
                   help="held-out rows from the start of the eval cache")
    p.add_argument("--queries", type=int, default=1024,
                   help="rows forced and scored by nearest neighbour; the "
                        "rest are the reference set")
    p.add_argument("--ood-rows", type=int, default=16384,
                   help="reference rows for the best-cosine score")
    p.add_argument("--min-rows", type=int, default=32,
                   help="cells with fewer rows are skipped")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--sae", default=None,
                   help="an sae.py run dir whose decoder directions are "
                        "scored at layer 0")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="directory for the JSON")
    cfg = p.parse_args()

    T = cfg.temperature or run_temperature(cfg.ckpt)
    w_sqrt = np.sqrt(np.load(cfg.mse_weights).astype(np.float64)
                     ).astype(np.float32)
    X = np.asarray(np.load(cfg.eval_cache, mmap_mode="r")[:cfg.rows],
                   np.float32)
    rng = np.random.default_rng(cfg.seed)
    from pareto import load_onto
    model, _, _ = load_onto(cfg.ckpt, cfg.step)
    # under `ste` the forward is the argmax, so there is no soft mixture
    soft = model.select.lower() not in ("ste", "argmax")
    del model
    view = model_view(cfg.ckpt, cfg.step, T, X, w_sqrt, cfg.batch, soft)
    Xw = X * w_sqrt
    var = float(((Xw - Xw.mean(0)) ** 2).sum())
    err = decomposition_error(view)
    print(f"{Path(cfg.ckpt).name} step {view['step']} T={T:g} "
          f"({'soft' if soft else 'hard'} code), {len(X)} rows; "
          f"decomposition error {err:.2e}", flush=True)
    if err > 1e-3:
        raise RuntimeError(f"gain x atom does not decompose the forward "
                           f"(relative error {err:.2e})")
    fvu = float(((Xw - view["Y"]) ** 2).sum() / var)
    fvu_hard = float(((Xw - view["Yhard"]) ** 2).sum() / var)
    res = {"ckpt": cfg.ckpt, "step": int(view["step"]), "temperature": T,
           "soft": soft, "rows": len(X), "fvu": fvu, "fvu_argmax": fvu_hard,
           "band": list(BAND), "decomposition_error": err, "layers": {}}
    print(f"  FVU_w {fvu:.4f}, end-to-end argmax {fvu_hard:.4f}")
    E_all = {}
    for l in range(view["A"].shape[0]):
        res["layers"][l], E_all[l], _ = place_layer(view, Xw, l, var,
                                                    cfg.min_rows)
        r = res["layers"][l]
        med = lambda key: r[key].get("median", np.nan)  # noqa: E731
        print(f"  layer {l}: off {med('off'):.3f} (raw {med('off_raw'):.3f},"
              f" common {med('common'):.3f})  beyond {med('beyond'):.3f}  "
              f"reach {med('reach'):.3f}  |a|/|mu| {med('atom_over_mu'):.3f}"
              f"  extreme dFVU/head {med('extreme_dfvu_per_head'):.2e}",
              flush=True)
    res["forcing"] = forcing(view, E_all, Xw, cfg.queries, rng)
    f = res["forcing"]
    print(f"  nn / real median: recon {f['recon']['median']:.3f}"
          + (f", argmax decode {f['argmax_decode']['median']:.3f}"
             if soft else ""), flush=True)
    for l in range(view["A"].shape[0]):
        e = f[f"layer {l}"]
        print(f"    layer {l} forced: atom {e['atom']['median']:.3f}  "
              f"extreme {e['extreme'].get('median', np.nan):.3f}"
              + (f"  own vertex {e['own_vertex']['median']:.3f}"
                 if soft else ""), flush=True)
    res["ood"] = ood(view, Xw, cfg.sae, w_sqrt, cfg.ood_rows, rng)
    for key, v in res["ood"].items():
        line = f"  ood {key}: atoms {v['atoms_mean']:.3f}"
        if "random_mean" in v:
            line += f"  random {v['random_mean']:.3f}"
        print(line, flush=True)
    if cfg.out:
        out = Path(cfg.out)
        out.mkdir(parents=True, exist_ok=True)
        name = f"{Path(cfg.ckpt).name}_{view['step']}.json"
        (out / name).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

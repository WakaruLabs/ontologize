"""The planted-simplex toy: where each selection rule puts a head's
vertices when the data's extremes are known.

Rows are convex mixtures of k planted atoms plus noise,
x = w A + sigma eps with w ~ Dirichlet(alpha 1_k). A small alpha puts most
rows near one atom, so the planted atoms are the corners of the data's
hull: the in-distribution extremes the sparsemax-argmax-sparsemax
schedule aims for. A cell's centroid sits inside its corner, pulled in by
the mixing, and a vertex past the corner is outside the data.

Each arm trains one head of k entries on the same rows, with the stock
training step and schedules, in `dictenc-toy`'s one-layer model with
gain-shape off:

  ste                 argmax forward, straight-through backward, under
  softmax             the stock temperature anneal
  sparsemax
  sparsemax_fixed     at the starting temperature throughout
  sparsemax_setpoint  the temperature set by feedback (the package's
                      DualLoop) to hold the share of decided rows, those
                      whose support is one entry, at --decided
  schedule_short      sparsemax, then ste, then sparsemax again, at the
  schedule_long       starting temperature throughout as in the real
                      recipe, so they differ from sparsemax_fixed by the
                      ste phase alone; --short and --long give where the
                      second and third phases start, as shares of the steps

The model can be taken toward the SONAR models' setting one knob at a
time: --h heads per layer, --l layers (with --deepsup and --resid-norm as
the stack uses them), --no-const0 (no constant coordinate at layer 0, so
the quadratic classifier is even in its input) and --unit (rows scaled
to unit norm). With more than one head or layer a head's atom is one term
of a sum, not a point of the data's space, so only `place.py`'s measures
are read, per layer.

`voronoi_ste` and `voronoi_softmax` are a different head: its logits are
tied to its atoms, -beta |x - a_j|^2 / 2, so its cells are the atoms'
Voronoi cells, and `ste` makes the Voronoi STE. It is trained on
reconstruction alone under the same anneal, with a learned beta and none
of the stock arms' training noises, from atoms near the rows' mean or at
k training rows (--voronoi-init).

Beside them is `archetypal`: the archetypes of the training rows
(Cutler & Breiman 1994). Rows are convex mixtures of archetypes, and
archetypes convex mixtures of rows. It is not a head, but the reference
for what a method held to the data's hull recovers here.

A head's entries are matched to planted atoms through its cells: the
contingency of each row's entry with its largest planted weight. A
matched entry's points are read against its atom a and the planted
atoms' mean m:

  lam   position along the ray from m through a: 1 at the atom, below 1
        inside, above 1 past it
  err   distance from a, over |a - m|

for the entry's vertex (its decoded atom), its cell's centroid and its
cell's trimmed extreme (`place.py`). Beside them are `place.py`'s own
measures of the head (off, beyond, reach, atom over centroid), the
held-out FVU of the soft and the argmax decode with the noise floor, the
share of rows whose entry is matched to their largest weight (acc), and
the code's effective support (entries carrying at least 1% of a row) and
decided share (rows with one entry at 99% or more).

  uv run python experiments/vertices/simplex.py --seeds 5

CPU by default. Writes <out>/summary.json and <out>/arms.csv.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import csv  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from jaxtyping import Float, Int  # noqa: E402
from scipy.optimize import linear_sum_assignment, nnls  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

ARMS = ("ste", "softmax", "sparsemax", "sparsemax_fixed",
        "sparsemax_setpoint", "schedule_short", "schedule_long",
        "voronoi_ste", "voronoi_softmax")
POINTS = ("vertex", "centroid", "extreme")
COLUMNS = ("acc", "fvu", "fvu_hard", "lam_vertex", "err_vertex",
           "lam_centroid", "lam_extreme", "atom_over_mu", "aom_max",
           "beyond", "reach", "support", "decided", "T")


# ---- the planted simplex ----------------------------------------------------

def mixtures(rng: np.random.Generator, A: Float[np.ndarray, "k d"], n: int,
             alpha: float, sigma: float
             ) -> Tuple[Float[np.ndarray, "n d"], Float[np.ndarray, "n k"],
                        Float[np.ndarray, "n d"]]:
    """n rows x = w A + sigma eps, w ~ Dirichlet(alpha 1_k): the rows, their
    weights and the rows without noise."""
    W = rng.dirichlet(np.full(len(A), alpha), n)
    clean = W @ A
    X = clean + sigma * rng.normal(size=clean.shape)
    return X.astype(np.float32), W, clean.astype(np.float32)


def ray_position(V: Float[np.ndarray, "m d"], A: Float[np.ndarray, "m d"],
                 m: Float[np.ndarray, "d"]
                 ) -> Tuple[Float[np.ndarray, "m"], Float[np.ndarray, "m"]]:
    """Each point v's position along the ray from m through its atom a (1 at
    a, 0 at m), and its distance from a over |a - m|."""
    u = A - m
    L = np.linalg.norm(u, axis=-1)
    lam = ((V - m) * u).sum(-1) / (L * L)
    return lam, np.linalg.norm(V - A, axis=-1) / L


def match_cells(sel: Int[np.ndarray, "n"], dom: Int[np.ndarray, "n"],
                k_model: int, k_true: int
                ) -> Tuple[Int[np.ndarray, "m"], Int[np.ndarray, "m"], float]:
    """Entries matched one to one with planted atoms, by Hungarian on the
    contingency of each row's entry with its largest planted weight, and the
    share of rows the matching agrees on."""
    C = np.zeros((k_model, k_true), np.int64)
    np.add.at(C, (sel, dom), 1)
    r, c = linear_sum_assignment(-C)
    return r, c, float(C[r, c].sum() / C.sum())


# ---- archetypal analysis ----------------------------------------------------

def simplex_project(V: Float[np.ndarray, "n k"]) -> Float[np.ndarray, "n k"]:
    """Each row's Euclidean projection onto the probability simplex."""
    k = V.shape[-1]
    z = -np.sort(-V, axis=-1)
    cs = np.cumsum(z, axis=-1)
    r = np.arange(1, k + 1)
    n = (1 + r * z > cs).sum(-1, keepdims=True)
    tau = (np.take_along_axis(cs, n - 1, axis=-1) - 1) / n
    return np.maximum(V - tau, 0.0)


def mixing(X: Float[np.ndarray, "n d"], Z: Float[np.ndarray, "k d"],
           iters: int = 200) -> Float[np.ndarray, "n k"]:
    """Each row's convex weights over the archetypes Z, minimizing
    |x - s Z|^2: accelerated projected gradient from the uniform weights."""
    G = Z @ Z.T
    L = max(float(np.linalg.eigvalsh(G).max()), 1e-12)
    XZ = X @ Z.T
    S = np.full((len(X), len(Z)), 1.0 / len(Z))
    Y, t = S, 1.0
    for _ in range(iters):
        S_new = simplex_project(Y - (Y @ G - XZ) / L)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        Y = S_new + ((t - 1.0) / t_new) * (S_new - S)
        S, t = S_new, t_new
    return S


def hull_point(z: Float[np.ndarray, "d"], X: Float[np.ndarray, "n d"]
               ) -> Float[np.ndarray, "d"]:
    """The point of the convex hull of X's rows nearest z. Nonnegative least
    squares, with sum-to-one as a heavily weighted extra equation (Cutler &
    Breiman's device), then normalized."""
    M = 1e3 * max(1.0, float(np.abs(X).max()))
    Aug = np.vstack([X.T, np.full(len(X), M)])
    c, _ = nnls(Aug, np.concatenate([z, [M]]), maxiter=50 * len(X))
    return (c / c.sum()) @ X


def furthest_sum(X: Float[np.ndarray, "n d"], k: int,
                 rng: np.random.Generator) -> Int[np.ndarray, "k"]:
    """FurthestSum seeds (Mørup & Hansen 2012): from a random row, add the
    row whose summed distance to those chosen is largest, k times, then
    drop the random start."""
    chosen = [int(rng.integers(len(X)))]
    dsum = np.linalg.norm(X - X[chosen[0]], axis=1)
    for _ in range(k):
        d = dsum.copy()
        d[chosen] = -np.inf
        j = int(np.argmax(d))
        chosen.append(j)
        dsum += np.linalg.norm(X - X[j], axis=1)
    return np.array(chosen[1:])


def archetypes(X: Float[np.ndarray, "n d"], k: int, rng: np.random.Generator,
               iters: int = 100, restarts: int = 3, tol: float = 1e-7
               ) -> Float[np.ndarray, "k d"]:
    """Cutler & Breiman's alternating least squares. Given the archetypes,
    each row's convex weights; given the weights, each archetype in turn at
    its least-squares position, projected onto the rows' hull. The error is
    isotropic in one archetype, so the projection solves that subproblem.
    From FurthestSum seeds; the lowest error over the restarts wins."""
    X = np.asarray(X, np.float64)
    best = (np.inf, None)
    for _ in range(restarts):
        Z = X[furthest_sum(X, k, rng)].copy()
        prev = np.inf
        for _ in range(iters):
            S = mixing(X, Z)
            for j in range(k):
                w = S[:, j]
                ww = float(w @ w)
                if ww < 1e-12:
                    continue
                R = X - S @ Z + np.outer(w, Z[j])
                Z[j] = hull_point((w @ R) / ww, X)
            rss = float(((X - S @ Z) ** 2).sum())
            if prev - rss < tol * prev:
                break
            prev = rss
        rss = float(((X - mixing(X, Z) @ Z) ** 2).sum())
        if rss < best[0]:
            best = (rss, Z.copy())
    return best[1]


# ---- the arms ---------------------------------------------------------------

def phases(arm: str, short: Tuple[float, float], long: Tuple[float, float]
           ) -> List[Tuple[float, str]]:
    """Each phase of an arm: where it starts, as a share of the steps, and
    its selection rule."""
    if arm in ("ste", "softmax"):
        return [(0.0, arm)]
    if arm.startswith("sparsemax"):
        return [(0.0, "sparsemax")]
    f1, f2 = {"schedule_short": short, "schedule_long": long}[arm]
    return [(0.0, "sparsemax"), (f1, "ste"), (f2, "sparsemax")]


def select_at(ph: List[Tuple[float, str]], step: int, steps: int) -> str:
    """The rule of the last phase begun by `step`."""
    rule = ph[0][1]
    for start, r in ph:
        if step >= round(start * steps):
            rule = r
    return rule


def with_select(state, rule: str):
    """The training state with its model's selection rule changed: the same
    parameters and optimizer state, and the forward `state_init` builds
    without ghost gradients."""
    from ontologize.ontologizer import Ontologizer
    model = state.model.clone(select=rule)
    f_stats = functools.partial(model.apply, method=Ontologizer.withStats)

    def f(params, X, *args, **kwargs):
        Y, stats, rng_next = f_stats(params, X, *args, **kwargs)
        return Y, None, stats, rng_next

    return state.replace(model=model, apply_fn=f)


def build(cfg, arm: str, seed: int):
    from ontologize.training.config import Hyperparams
    anneal = arm in ("ste", "softmax", "sparsemax", "sparsemax_setpoint")
    hyper = Hyperparams(
        cfg.d, cfg.d, cfg.b, epochs=1, lr=cfg.lr, wd=0.0,
        temperature=cfg.temperature,
        noise_in="batchnorm", noise_K="normal", noise_F="featvar",
        sd_in=0.0, sd_K=cfg.sd_k, sd_F=cfg.sd_f,
        p_drop=cfg.p_drop, p_revive=0.0,
        s_L1F=0.0, s_bcossim=0.0, s_hcossim=0.0, s_H=0.0, s_kcossim=0.0,
        s_Hm=0.0, seed=seed, ghost=False,
        temperature_end=cfg.temperature_end if anneal else None,
        anneal_steps=cfg.anneal_steps if anneal else 0,
        p_drop_start=0.0 if anneal else None,
        sd_K_end=0.2 * cfg.temperature_end if anneal else None)
    rule = phases(arm, cfg.short, cfg.long)[0][1]
    model = hyper.ontologizer(
        cfg.d, cfg.d, cfg.e_dec, cfg.k, cfg.h, cfg.l, n=cfg.n, gate="none",
        select=rule, forward="resid", deepsup=cfg.deepsup,
        resid_norm=cfg.resid_norm, resid_const=True,
        const0=not cfg.no_const0, resid_gain=False,
        dtype_str="float32", dtype_p_str="float32")
    return hyper, hyper.init(model, save_each=cfg.steps)


def decided_fn(model):
    """Jitted (params, X, T) -> the share of rows whose code is one entry."""
    import jax

    def probe(module, X, T):
        E, _ = module.encode(X, 0.0, None)
        de = module.dictencs[0]
        U, _ = de.gainshape_in(module.constinput(E))
        P = de.dict.cluster(de.classifier(U), T)
        return ((P > 0).sum(-1) == 1).mean()

    return jax.jit(lambda p, X, T: model.apply(p, X, T, method=probe))


def train(cfg, arm: str, X_tr: Float[np.ndarray, "n d"], seed: int):
    """The trained state and the temperature its last step used."""
    import jax
    import jax.numpy as jnp
    from ontologize.training.ontostate import DualLoop, schedules, update
    hyper, state = build(cfg, arm, seed)
    ph = phases(arm, cfg.short, cfg.long)
    rng = hyper.rng()
    order = np.random.default_rng(seed)
    n_batches = len(X_tr) // cfg.b
    perm = order.permutation(len(X_tr))
    # the hardness 1/T, raised while more rows than the setpoint allows are
    # undecided; never softer than the starting temperature
    loop = DualLoop(1.0 / cfg.temperature, 1.0 - cfg.decided,
                    cfg.decided_eta, cfg.decided_ema, 1.0 / cfg.temperature,
                    1.0 / cfg.tmin, cfg.anneal_steps, cfg.decided_warmup) \
        if arm == "sparsemax_setpoint" else None
    decided = decided_fn(state.model) if loop else None
    T_ctl, share = cfg.temperature, float("nan")
    t0 = time.time()
    for step in range(cfg.steps):
        rule = select_at(ph, step, cfg.steps)
        if rule != state.model.select:
            state = with_select(state, rule)
        if step and step % n_batches == 0:
            perm = order.permutation(len(X_tr))
        i = (step % n_batches) * cfg.b
        Xb = jnp.asarray(X_tr[perm[i:i + cfg.b]])
        T, pd, sk, _ = schedules(
            step, hyper.anneal_steps, hyper.temperature,
            hyper.temperature_end, hyper.p_drop, hyper.p_drop_start,
            hyper.sd_K, hyper.sd_K_end)
        if loop:
            # the logit noise keeps the ratio to the temperature that the
            # stock schedule gives it at this step
            sk, T = sk * T_ctl / T, T_ctl
        rng, r = jax.random.split(rng)
        state, L, _ = update(
            state, hyper.loss, r, Xb, Xb, temperature=T, p_drop=pd,
            sd_K=sk, sd_in=0.0, sd_F=hyper.sd_F, grad_clip=hyper.grad_clip,
            p_revive=0.0, revive_frac=hyper.revive_frac)
        if loop:
            share = float(decided(state.params, Xb, T))
            if step + 1 >= cfg.decided_warmup:
                loop.step(step + 1, max(1.0 - share, 1e-4))
                T_ctl = 1.0 / loop.s
        if step % 1000 == 0 or step == cfg.steps - 1:
            row = state.last_stats()
            print(f"    {arm} step {step:5d}  MSE {float(row[1]):.4f}  "
                  f"T {T:.4f}  {rule}"
                  + (f"  decided {share:.3f}" if loop else "")
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
    return state, float(T)


# ---- a Voronoi head ---------------------------------------------------------

def voronoi_logits(X, A, beta):
    """A Voronoi head's logits, -beta |x - a_j|^2 / 2: the classifier is tied
    to the atoms, so its cells are the atoms' Voronoi cells. Works on numpy
    and JAX arrays alike."""
    d2 = (X * X).sum(-1)[:, None] - 2.0 * X @ A.T + (A * A).sum(-1)[None, :]
    return -0.5 * beta * d2


def voronoi_init(X: Float[np.ndarray, "n d"], k: int, how: str,
                 rng: np.random.Generator) -> Float[np.ndarray, "k d"]:
    """The head's starting atoms: `mean`, the rows' mean plus Gaussian noise
    of a tenth of each coordinate's SD (near the stock head's start, whose
    decoded atoms begin close together), or `rows`, k distinct training
    rows (the usual quantizer start)."""
    if how == "rows":
        return X[rng.choice(len(X), k, replace=False)].astype(np.float32)
    return (X.mean(0) + 0.1 * X.std(0) * rng.normal(size=(k, X.shape[1]))
            ).astype(np.float32)


def train_voronoi(cfg, rule: str, X_tr: Float[np.ndarray, "n d"], seed: int,
                  A0: Optional[Float[np.ndarray, "k d"]] = None):
    """One Voronoi head trained on reconstruction alone, x_hat = p A with
    p = rule(logits / T), under the stock temperature anneal, by Adam at
    the stock arms' learning rate and batch. `ste` is the Voronoi STE: the
    forward writes the nearest atom. The learned scale beta lets the head
    soften or sharpen its own code, as the stock classifier's logit scale
    does; the stock arms' training noises are left out. It starts from
    `voronoi_init`'s atoms unless `A0` gives them. Returns the atoms, beta
    and the last temperature."""
    import jax
    import jax.numpy as jnp
    import optax
    from ontologize.fns.keys import get_activation
    from ontologize.training.ontostate import schedules
    fn = get_activation(rule)
    rng = np.random.default_rng(seed)
    if A0 is None:
        A0 = voronoi_init(X_tr, cfg.k, cfg.voronoi_init, rng)
    params = {"A": jnp.asarray(A0, jnp.float32),
              "log_beta": jnp.asarray(np.log(cfg.voronoi_beta), jnp.float32)}
    opt = optax.adam(cfg.lr)
    ost = opt.init(params)

    def loss(p, X, T):
        P = fn(voronoi_logits(X, p["A"], jnp.exp(p["log_beta"])) / T)
        return ((X - P @ p["A"]) ** 2).mean()

    @jax.jit
    def step(p, s, X, T):
        L, g = jax.value_and_grad(loss)(p, X, T)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, L

    n_batches = len(X_tr) // cfg.b
    perm = rng.permutation(len(X_tr))
    t0 = time.time()
    for i in range(cfg.steps):
        if i and i % n_batches == 0:
            perm = rng.permutation(len(X_tr))
        j = (i % n_batches) * cfg.b
        T = schedules(i, cfg.anneal_steps, cfg.temperature,
                      cfg.temperature_end, 0.0, None, 0.0, None)[0]
        params, ost, L = step(params, ost, jnp.asarray(X_tr[perm[j:j + cfg.b]]),
                              T)
        if i % 1000 == 0 or i == cfg.steps - 1:
            print(f"    voronoi_{rule} step {i:5d}  MSE {float(L):.4f}  "
                  f"T {T:.4f}  beta {float(jnp.exp(params['log_beta'])):.3g}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    return (np.asarray(params["A"]), float(np.exp(params["log_beta"])),
            float(T))


def evaluate_voronoi(cfg, rule: str, A: Float[np.ndarray, "k d"], beta: float,
                     T: float, X: Float[np.ndarray, "n d"],
                     W: Float[np.ndarray, "n k"],
                     clean: Float[np.ndarray, "n d"],
                     A_ref: Float[np.ndarray, "k d"]) -> Dict[str, float]:
    """`evaluate`'s scores for a Voronoi head. Its atoms are its vertices,
    and its write is the whole reconstruction, so `place.py` reads it
    through a one-head view with gain 1."""
    import jax.numpy as jnp
    from ontologize.fns.keys import get_activation
    from place import place_layer
    fn = get_activation(rule)
    P = np.asarray(fn(jnp.asarray(voronoi_logits(X, A, beta) / T)),
                   np.float64)
    sel = P.argmax(1)
    soft = rule not in ("ste", "argmax")
    view = dict(A=A[None, None], sel=sel[:, None, None],
                P=P[:, None, None] if soft else None,
                G=np.ones((len(X), 1)), Y=P @ A)
    var = float(((X - X.mean(0)) ** 2).sum())
    r, E, M = place_layer(view, X, 0, var, cfg.min_rows)
    ents, atoms, acc = match_cells(sel, W.argmax(1), len(A), len(A_ref))
    m = A_ref.mean(0)
    out = {"acc": acc, "T": T, "beta": beta,
           "fvu": float(((X - view["Y"]) ** 2).sum() / var),
           "fvu_hard": float(((X - A[sel]) ** 2).sum() / var),
           "fvu_oracle": float(((X - clean) ** 2).sum() / var)}
    for name, pts in zip(POINTS, (A, M[0], E[0])):
        lam, err = ray_position(pts[ents], A_ref[atoms], m)
        out[f"lam_{name}"] = float(np.nanmean(lam))
        out[f"err_{name}"] = float(np.nanmean(err))
    Pc = P if soft else np.eye(len(A))[sel]
    out["support"] = float((Pc >= 0.01).sum(-1).mean())
    out["decided"] = float((Pc.max(-1) >= 0.99).mean())
    for key in ("off", "beyond", "reach", "atom_over_mu", "common"):
        out[key] = float(r[key].get("median", np.nan))
    out["aom_max"] = out["atom_over_mu"]
    return out


# ---- scoring ----------------------------------------------------------------

def evaluate(cfg, state, T: float, X: Float[np.ndarray, "n d"],
             W: Float[np.ndarray, "n k"], clean: Float[np.ndarray, "n d"],
             A_ref: Float[np.ndarray, "k d"]) -> Dict[str, float]:
    """One trained arm's scores on the held-out rows. `place.py`'s measures
    are read per layer (`L<l>_<measure>`); the unprefixed ones are layer
    0's, and `aom_max` is the largest atom over centroid of any layer. The
    readout against the planted corners (acc, lam, err) needs one head at
    one layer, whose vertices are points of the data's own space; with more
    heads or layers a head's atom is one term of a sum, and it is left
    out. Support and decided share are over layer 0's heads."""
    from ontologize.ontologizer import Ontologizer
    from place import place_layer, view_of
    model, params = state.model, state.params
    soft = model.select.lower() not in ("ste", "argmax")
    view = view_of(model, params["params"], T, X,
                   np.ones(X.shape[1], np.float32), 1024, soft)
    L, h, k, d = view["A"].shape
    V = np.asarray(model.apply(params, method=Ontologizer.decodeEntries)[0]
                   ).reshape(L, h, k, d)
    b = (V - view["A"]).reshape(-1, d).mean(0)   # decode(0), every entry's
    if np.abs(V - view["A"] - b).max() > 1e-4 * max(1.0, np.abs(V).max()):
        raise RuntimeError("decoded entries are not decode(0) plus atoms")
    var = float(((X - X.mean(0)) ** 2).sum())
    out = {"T": T,
           "fvu": float(((X - view["Y"]) ** 2).sum() / var),
           "fvu_hard": float(((X - view["Yhard"]) ** 2).sum() / var),
           "fvu_oracle": float(((X - clean) ** 2).sum() / var)}
    P = view["P"][:, 0] if soft else \
        np.eye(k)[view["sel"][:, 0].astype(np.int64)]
    out["support"] = float((P >= 0.01).sum(-1).mean())
    out["decided"] = float((P.max(-1) >= 0.99).mean())
    for l in range(L):
        r, E, M = place_layer(view, X, l, var, cfg.min_rows)
        for key in ("off", "beyond", "reach", "atom_over_mu", "common"):
            out[f"L{l}_{key}"] = float(r[key].get("median", np.nan))
        if l == 0:
            E0, M0 = E, M
    for key in ("off", "beyond", "reach", "atom_over_mu"):
        out[key] = out[f"L0_{key}"]
    out["aom_max"] = float(np.nanmax([out[f"L{l}_atom_over_mu"]
                                      for l in range(L)]))
    if L == 1 and h == 1:
        sel = view["sel"][:, 0, 0].astype(np.int64)
        ents, atoms, out["acc"] = match_cells(sel, W.argmax(1), k,
                                              len(A_ref))
        m = A_ref.mean(0)
        for name, pts in zip(POINTS, (V[0, 0], b + M0[0], b + E0[0])):
            lam, err = ray_position(pts[ents], A_ref[atoms], m)
            out[f"lam_{name}"] = float(np.nanmean(lam))
            out[f"err_{name}"] = float(np.nanmean(err))
    return out


def evaluate_archetypes(Z: Float[np.ndarray, "k d"],
                        X: Float[np.ndarray, "n d"],
                        W: Float[np.ndarray, "n k"],
                        A_true: Float[np.ndarray, "k d"]) -> Dict[str, float]:
    S = mixing(np.asarray(X, np.float64), Z)
    ents, atoms, acc = match_cells(S.argmax(1), W.argmax(1), len(Z),
                                   len(A_true))
    lam, err = ray_position(Z[ents], A_true[atoms], A_true.mean(0))
    var = float(((X - X.mean(0)) ** 2).sum())
    return {"acc": acc, "fvu": float(((X - S @ Z) ** 2).sum() / var),
            "lam_vertex": float(lam.mean()), "err_vertex": float(err.mean()),
            "support": float((S >= 0.01).sum(-1).mean()),
            "decided": float((S.max(-1) >= 0.99).mean())}


def print_header() -> None:
    print(f"  {'':20s}" + "".join(f"{c:>13s}" for c in COLUMNS), flush=True)


def print_row(name: str, res: Dict[str, float]) -> None:
    print(f"  {name:20s}" + "".join(f"{res.get(c, np.nan):>13.3f}"
                                    for c in COLUMNS), flush=True)


def replicate(cfg, seed: int, data_seed: int) -> dict:
    data = np.random.default_rng(data_seed)
    A_true = data.normal(0.0, 1.0, (cfg.k_true, cfg.d)).astype(np.float32)
    X_tr, _, _ = mixtures(data, A_true, cfg.n_train, cfg.alpha, cfg.sigma)
    X_ev, W_ev, C_ev = mixtures(data, A_true, cfg.n_eval, cfg.alpha,
                                cfg.sigma)
    A_ref = A_true
    if cfg.unit:
        n_ev = np.linalg.norm(X_ev, axis=1, keepdims=True)
        X_tr = X_tr / np.linalg.norm(X_tr, axis=1, keepdims=True)
        X_ev, C_ev = X_ev / n_ev, C_ev / n_ev
        A_ref = A_true / np.linalg.norm(A_true, axis=1, keepdims=True)
    # where a perfect hard code's atoms would sit: the planted cells'
    # centroids
    dom = W_ev.argmax(1)
    cen = np.stack([X_ev[dom == j].mean(0) for j in range(cfg.k_true)])
    lam_c, _ = ray_position(cen, A_ref, A_ref.mean(0))
    # the planted code itself, read as the heads' codes are
    res = {"seed": seed, "data_seed": data_seed,
           "planted_centroid_lam": float(lam_c.mean()),
           "planted_support": float((W_ev >= 0.01).sum(-1).mean()),
           "planted_decided": float((W_ev.max(-1) >= 0.99).mean()),
           "arms": {}}
    print(f"seed {seed}, data seed {data_seed}: the planted cells' centroids "
          f"sit at lam {lam_c.mean():.3f}; the planted code's support "
          f"{res['planted_support']:.2f}, decided "
          f"{res['planted_decided']:.3f}", flush=True)
    for arm in cfg.arms:
        if arm.startswith("voronoi_"):
            rule = arm[len("voronoi_"):]
            A, beta, T = train_voronoi(cfg, rule, X_tr, seed)
            res["arms"][arm] = evaluate_voronoi(cfg, rule, A, beta, T, X_ev,
                                                W_ev, C_ev, A_ref)
            continue
        state, T = train(cfg, arm, X_tr, seed)
        res["arms"][arm] = evaluate(cfg, state, T, X_ev, W_ev, C_ev, A_ref)
    sub = data.choice(len(X_tr), cfg.aa_rows, replace=False)
    Z = archetypes(X_tr[sub], cfg.k_true, np.random.default_rng(seed),
                   cfg.aa_iters, cfg.aa_restarts)
    res["arms"]["archetypal"] = evaluate_archetypes(Z, X_ev, W_ev, A_ref)
    print_header()
    for arm, r in res["arms"].items():
        print_row(arm, r)
    print(f"  noise floor FVU "
          f"{next(iter(res['arms'].values())).get('fvu_oracle', np.nan):.4f}",
          flush=True)
    return res


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("planted simplex")
    g.add_argument("--k-true", type=int, default=8)
    g.add_argument("--d", type=int, default=16)
    g.add_argument("--alpha", type=float, default=0.2,
                   help="Dirichlet concentration of the mixing weights; "
                        "below 1 puts most rows near a corner")
    g.add_argument("--sigma", type=float, default=0.05,
                   help="noise SD per coordinate; atoms are N(0, 1) per "
                        "coordinate")
    g.add_argument("--n-train", type=int, default=32768)
    g.add_argument("--n-eval", type=int, default=8192)
    g.add_argument("--data-seed", type=int, default=0)
    m = p.add_argument_group("model")
    m.add_argument("--k", type=int, default=None, help="entries (k_true)")
    m.add_argument("--h", type=int, default=1, help="heads per layer")
    m.add_argument("--l", type=int, default=1, help="layers")
    m.add_argument("--e-dec", type=int, default=None,
                   help="dictionary width (default: 32, or h k if larger)")
    m.add_argument("--n", type=int, default=2, help="classifier order")
    m.add_argument("--deepsup", action="store_true",
                   help="every layer prefix reconstructs the row too")
    m.add_argument("--resid-norm", action="store_true",
                   help="unit-normalize the residual that each layer after "
                        "the first reads")
    m.add_argument("--no-const0", action="store_true",
                   help="leave the constant coordinate out of layer 0's "
                        "classifier input, which makes the quadratic "
                        "classifier even in its input")
    m.add_argument("--unit", action="store_true",
                   help="scale rows to unit norm, as SONAR's cached rows are; "
                        "the planted corners become the atoms' directions")
    t = p.add_argument_group("training")
    t.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--anneal-steps", type=int, default=3000)
    t.add_argument("--b", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument("--temperature-end", type=float, default=0.03)
    t.add_argument("--p-drop", type=float, default=0.1)
    t.add_argument("--sd-k", type=float, default=0.02)
    t.add_argument("--sd-f", type=float, default=0.1)
    t.add_argument("--short", type=float, nargs=2, default=(0.75, 0.9),
                   help="where schedule_short's argmax and last sparsemax "
                        "phases start, as shares of the steps")
    t.add_argument("--long", type=float, nargs=2, default=(0.6, 0.75))
    t.add_argument("--decided", type=float, default=0.5,
                   help="sparsemax_setpoint's share of decided rows")
    t.add_argument("--decided-eta", type=float, default=1e-2)
    t.add_argument("--decided-ema", type=float, default=0.99)
    t.add_argument("--decided-warmup", type=int, default=500)
    t.add_argument("--tmin", type=float, default=1e-3,
                   help="the lowest temperature the setpoint may reach")
    t.add_argument("--voronoi-init", default="mean", choices=("mean", "rows"),
                   help="the Voronoi arms' starting atoms: near the rows' "
                        "mean, or k training rows")
    t.add_argument("--voronoi-beta", type=float, default=1.0,
                   help="the Voronoi arms' starting logit scale")
    t.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-rows", type=int, default=32,
                   help="cells with fewer rows get no centroid or extreme")
    p.add_argument("--aa-rows", type=int, default=2048,
                   help="training rows the archetypes are fitted on")
    p.add_argument("--aa-iters", type=int, default=100)
    p.add_argument("--aa-restarts", type=int, default=3)
    p.add_argument("--seeds", type=int, default=1,
                   help="replicates; the model seed and the data seed "
                        "advance together")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default=None,
                   help="default data/out/vertices/simplex/<name>")
    cfg = p.parse_args()
    cfg.k = cfg.k or cfg.k_true
    cfg.e_dec = cfg.e_dec or max(32, cfg.h * cfg.k)
    if cfg.name is None:
        cfg.name = (f"k{cfg.k_true}_d{cfg.d}_a{cfg.alpha:g}_s{cfg.sigma:g}"
                    + (f"_m{cfg.k}" if cfg.k != cfg.k_true else "")
                    + (f"_h{cfg.h}" if cfg.h != 1 else "")
                    + (f"_l{cfg.l}" if cfg.l != 1 else "")
                    + ("_deepsup" if cfg.deepsup else "")
                    + ("_rnorm" if cfg.resid_norm else "")
                    + ("_noconst0" if cfg.no_const0 else "")
                    + ("_unit" if cfg.unit else "")
                    + (f"_x{cfg.seeds}" if cfg.seeds > 1 else ""))
    return cfg


def main() -> None:
    cfg = parse_args()
    out = Path(cfg.out) if cfg.out else \
        ROOT / "data/out/vertices/simplex" / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"{cfg.name}: {cfg.k_true} atoms in d={cfg.d}, alpha={cfg.alpha:g}, "
          f"sigma={cfg.sigma:g}; one head of {cfg.k}; {cfg.steps} steps x "
          f"{cfg.seeds} seed(s)", flush=True)
    results = [replicate(cfg, cfg.seed + i, cfg.data_seed + i)
               for i in range(cfg.seeds)]
    names = list(results[0]["arms"])
    tally = {}
    for arm in names:
        tally[arm] = {}
        for c in sorted({c for r in results for c in r["arms"][arm]}):
            v = np.array([r["arms"][arm].get(c, np.nan) for r in results])
            if np.isfinite(v).any():
                tally[arm][c] = {"mean": float(np.nanmean(v)),
                                 "median": float(np.nanmedian(v)),
                                 "sd": float(np.nanstd(v))}
        # seeds whose cells match the planted ones; a seed that merges two
        # corners into one entry leaves a corner unmatched
        tally[arm]["clean"] = int(sum(r["arms"][arm].get("acc", np.nan) >= 0.9
                                      for r in results))
    planted = {key: float(np.mean([r[key] for r in results]))
               for key in ("planted_centroid_lam", "planted_support",
                           "planted_decided")}
    if cfg.seeds > 1:
        print(f"\nover {cfg.seeds} seeds; the planted cells' centroids sit at "
              f"lam {planted['planted_centroid_lam']:.3f}; the planted code's "
              f"support {planted['planted_support']:.2f}, decided "
              f"{planted['planted_decided']:.3f}")
        for stat in ("mean", "median"):
            print(f"{stat}s:")
            print_header()
            for arm in names:
                print_row(arm, {c: v[stat] for c, v in tally[arm].items()
                                if isinstance(v, dict)})
        print("seeds with acc >= 0.9: " + ", ".join(
            f"{arm} {tally[arm]['clean']}" for arm in names))
    with open(out / "arms.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        keys = sorted({k for r in results for a in r["arms"].values()
                       for k in a})
        w.writerow(["seed", "arm"] + keys)
        for r in results:
            for arm, a in r["arms"].items():
                w.writerow([r["seed"], arm] + [a.get(k, "") for k in keys])
    cfg_json = {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in vars(cfg).items()}
    (out / "summary.json").write_text(json.dumps(
        {"config": cfg_json, "planted": planted, "seeds": results,
         "tally": tally}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

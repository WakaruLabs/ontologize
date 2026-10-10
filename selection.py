"""Selection surfaces with their trivial baselines drawn.

Two figures over a set of runs (--runs, globs allowed), each run's
settings read from its restored spec and its log.jsonl env_config, plus
the derived keys below. Every score is on held-out cache tail rows (the
sae.py holdout, --eval-rows).

CURVE (default): the noise2self partition score against a sweep key --x.
Each head's assignment P_h (b, k) is scored as `ontologize.fns.pwak.pwak_l2`
scores it during training, but as an evaluation on the tail:

  G = wak((P_h P_h^T) . D),   score = l2(E, G^s E) / l2(E, mean E)

D is the heat-kernel affinity of the layer's input E (self-edges removed,
`pwak.affinity`, --tau), so each sample is predicted from its
co-classified neighbours' values, never its own. Its own input still sets
its weights, so the score is leave-one-out, not J-invariant, and any
input-dependent partition gains on it somewhat even on structureless
data. E is the input the layer's
classifier sees: X for layer 0 (unit-normalized when the run has
`resid_gain` and no encoder, as the classifier sees it), the residual
from `autointerp.onto_acts_fn(..., inputs=True)` above that. 0 means the
neighbours predict a sample exactly, 1 that the graph does no better than
the batch mean (dotted line). A sample alone in its cell has an all-zero
row and is predicted as 0, so a fragmented partition is penalized.

Per panel (one per layer, one figure per eval depth --s) the drawn
series are
  partition    mean over the scored heads, band = 10-90% of heads
  joint        the layer's joint gate sum_h P_h P_h^T: the training stat
  random       the same assignments with the sample rows permuted, i.e. a
               random partition with exactly matched cell sizes and
               softness (mean +- sd over --nulls draws; per head, averaged).
               It ignores the input, so it is a floor only for partitions
               that do too
  unpartitioned  one cell holding every sample: the plain heat-kernel
               smoother. Horizontal on layer 0, where every run shares the
               input; above it each run has its own residual, so it is a
               per-run baseline.
The star is the argmin of the partition series; the corner text gives the
same run's baselines beside it. The lower row redraws every series minus
its own run's unpartitioned score, so a curve that falls only because the
residual it scores got easier (upper layers of a better-reconstructing
run) reads flat there; below 0 the partition beats the plain smoother.
Per-head scores depend on cell size relative to --b (a hard head with k
cells leaves ~b/k neighbours per sample), so compare across k against the
size-matched random series, not against 1. All-singletons is not drawn: every
row of G is zero there, so it predicts 0 for every sample, a fixed
function of E rather than a property of any partition.

SPLIT SCORE (--jinv, with the curve): a J-invariant form of the score,
the bandwidth it selects, and the null a partition must beat there. Each
batch's layer input is centered, rotated by a random orthogonal Q fixed
per run, and cut in half: view A builds every gate and kernel, view B is
the value scored (`split_views`), so no row's scored coordinates set its
own weights, which Noise2Self's guarantee requires. The bandwidth is
chosen as DEWAKSS chooses its own, as the --taus value whose plain
smoother scores lowest under the split form (`select_bandwidth`; the
`selected` column). At every --taus, jinv.csv holds the full
(leave-one-out) and split forms of
  smoother       one cell: the unpartitioned series
  trained_head   the scored heads, mean (full form only: a trained head
                 reads every coordinate of E, so it has no split form)
  trained_joint  the layer's joint gate (full form only)
  lsh_head       random-projection heads of the run's k, one at a time
                 (mean over --lsh-heads)
  lsh_joint      the joint gate of those --lsh-heads
  lsh_kernel     the joint gate of --lsh-kernel heads drawn separately,
                 which estimates the family's collision kernel
A random-projection head (`lsh_assign`) is a locality-sensitive hash
(Har-Peled, Indyk & Motwani 2012): its collision probability depends only
on the angle between two rows and falls as the angle grows, so gating the
heat kernel by it sharpens the kernel the way a smaller tau does. A
partition shows more than locality only by beating these, against the
smoother at the selected bandwidth. jinv_isotropic.csv scores the
smoother on isotropic Gaussian rows, where any J-invariant predictor
scores at least (b/(b-1))^2 at s=1: the split form stays at or above it,
and the full form falls below it by its leak.

GRID (--grid ROW COL): held-out whitened FVU over a 2-D grid of settings.
Each run's FVU is pareto.py's: the "soft" row (the model as it runs, at
its temperature) or --point hard, read from an existing pareto.csv when
one is found (<run>/pareto/, then pareto_<run name>/ beside the run, then
<--pareto-dir>/pareto_<run name>/, by default data/out/<substrate>/pareto
of the run's own substrate; exact names only, since a pareto.csv does not
record its checkpoint and run names repeat across substrates), otherwise computed with
`pareto.onto_points` on the same tail rows and --mse-weights, which must
match the cache's width (pass the activation cache's own weights for
gpt2_l8). Cells are annotated with FVU and nominal capacity under
pareto.py's convention:
  argmax / ste / top1  0 coefficients, l*h*log2(k) index bits (its "hard")
  top<n>, n < k        l*h*n coefficients, l*h*log2 C(k, n) bits (its "dev m=n")
  softmax              l*h*k coefficients, 0 bits (its "soft")
Runs sharing a cell are averaged (n shown); cells no run covers stay
blank, never zero. The argmin cell is outlined. --sae-csv marks the SAE
rows of a pareto.csv on the colour bar, on the same scale: only its
capacity frontier (no SAE with fewer or equal coefficients does better),
labelled with each one's coefficients. Read CSVs record
nothing about the eval split they used, so the source of every cell is
printed and kept in the output CSV.

Derived keys usable as --x / --grid / --hue, besides any spec or
env_config field (l, h, k, select, e_dec, s_Hm, temperature_end, ...):
  name     the run directory name
  n_sel    entries selected per head: top<n> -> n, softmax -> k,
           argmax/ste -> 1
  bits, coeffs   the nominal capacity above

  uv run python selection.py --runs 'data/out/sonar/multilingual/sweep_top[0-9]*' \\
      data/out/sonar/multilingual/sweep_softmax* --exclude relu \\
      --x n_sel --hue s_Hm
  uv run python selection.py --runs ... --grid s_Hm n_sel \\
      --sae-csv data/out/sonar/pareto_unified/pareto.csv
  uv run python selection.py --runs ... --jinv --taus 0.2 0.1 0.05 0.03
  uv run python selection.py --replot --out <dir>          # either figure

Writes <out>/n2s.csv (long form: run, x, hue, layer, s, stat, value),
n2s.json (the keys, tau and row settings) and n2s_s<S>.png, or grid.csv
and grid.png; --replot redraws from them (pass --grid with any two keys
for the grid; the CSV records the real ones). --jinv adds jinv.csv (long
form: run, x, hue, layer, s, tau, form, series, value, selected) and
jinv_isotropic.csv (tau, s, full, split, floor), and prints each run's
gains over the smoother at the same tau; with --replot it reprints them.
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from assignmap import read_hyper, temperature_at
from ontologize.fns.pwak import affinity, wak

STATS = ("partition", "partition_p10", "partition_p90", "joint",
         "random", "random_sd", "random_joint", "random_joint_sd",
         "unpartitioned")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=[],
                   help="Ontologizer run dirs (globs expanded)")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="drop runs whose name contains any of these")
    p.add_argument("--x", default="n_sel", help="sweep key of the curve")
    p.add_argument("--hue", default=None,
                   help="key splitting runs into separate series")
    p.add_argument("--grid", nargs=2, default=None, metavar=("ROW", "COL"),
                   help="draw the FVU grid over these two keys instead")
    p.add_argument("--point", choices=["soft", "hard"], default="soft",
                   help="grid: which pareto.py point is the run's FVU")
    p.add_argument("--pareto-dir", default=None,
                   help="grid: where pareto_<run name>/pareto.csv may exist "
                   "(default: <data/out/SUBSTRATE>/pareto for each run, so a "
                   "namesake on another substrate is never read)")
    p.add_argument("--recompute", action="store_true",
                   help="grid: ignore existing pareto.csv files")
    p.add_argument("--sae-csv", default=None,
                   help="grid: pareto.csv whose sae rows mark the colour bar")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="held-out tail rows of the cache (sae.py's default)")
    p.add_argument("--rows", type=int, default=8192,
                   help="curve: tail rows scored, drawn once and shared by "
                   "every run (a multiple of --b)")
    p.add_argument("--b", type=int, default=2048,
                   help="curve: graph batch (the kernel is batch-local); "
                   "grid: pareto.py's eval batch is 2x this")
    p.add_argument("--s", type=int, nargs="+", default=[1],
                   help="curve: diffusion depths scored (one figure each)")
    p.add_argument("--tau", type=float, default=0.2,
                   help="heat-kernel temperature (pwak_tau's default)")
    p.add_argument("--nulls", type=int, default=3,
                   help="random-partition draws per batch")
    p.add_argument("--max-heads", type=int, default=32,
                   help="curve: heads scored per layer, evenly spaced")
    p.add_argument("--jinv", action="store_true",
                   help="curve: also write the split (J-invariant) score, "
                   "the bandwidth it selects and the random-projection "
                   "null to jinv.csv")
    p.add_argument("--taus", type=float, nargs="+",
                   default=[0.2, 0.1, 0.05, 0.03, 0.02, 0.01],
                   help="--jinv: heat-kernel temperatures scored and "
                   "selected among")
    p.add_argument("--lsh-heads", type=int, default=8,
                   help="--jinv: random-projection heads per batch, scored "
                   "one at a time and as one joint gate")
    p.add_argument("--lsh-kernel", type=int, default=64,
                   help="--jinv: random-projection heads in the joint gate "
                   "that estimates the collision kernel")
    p.add_argument("--replot", action="store_true",
                   help="redraw from <out>/n2s.csv or grid.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/out/sonar/selection")
    return p.parse_args(argv)


# ---------- settings ----------

def select_entries(select: str, k: int) -> int:
    """Entries a head passes on per sample under its `select` rule."""
    s = select.lower()
    if s == "softmax":
        return k
    m = re.fullmatch(r"top(\d+)", s)
    if m:
        return min(int(m.group(1)), k)
    return 1  # argmax / ste: one-hot


def capacity(l: int, h: int, k: int, select: str) -> tuple:
    """(coefficients, index bits) per sample under pareto.py's convention:
    one-hot codes are its "hard" point, top<n> its "dev m=n", softmax its
    "soft" point."""
    from pareto import log2_choose
    n = select_entries(select, k)
    if n >= k:
        return l * h * k, 0
    if n == 1:
        return 0, round(l * h * math.log2(k))
    return l * h * n, round(l * h * log2_choose(k, n))


def expand_runs(patterns: Sequence[str], exclude: Sequence[str] = ()):
    """Glob-expanded run dirs, de-duplicated in order, dropping names that
    contain an --exclude substring."""
    out = []
    for pat in patterns:
        for r in sorted(glob.glob(pat)) or [pat]:
            if r not in out and not any(e in Path(r).name for e in exclude):
                out.append(r)
    return out


def run_settings(ckpt) -> dict:
    """Spec fields over env_config hyperparameters, plus name, step,
    n_sel, coeffs and bits."""
    import orbax.checkpoint as ocp
    from ontologize.training.serialize import restore_spec
    manager = ocp.CheckpointManager(
        Path(ckpt).resolve(),
        checkpointers={"state": ocp.PyTreeCheckpointer(),
                       "spec": ocp.PyTreeCheckpointer()})
    step = manager.latest_step()
    spec = restore_spec(manager, step)
    st = dict(read_hyper(ckpt) or {})
    st.update({k: v for k, v in spec.items()
               if isinstance(v, (int, float, str, bool, type(None)))})
    st["name"], st["path"], st["step"] = Path(ckpt).name, str(ckpt), int(step)
    st["n_sel"] = select_entries(st["select"], st["k"])
    st["coeffs"], st["bits"] = capacity(st["l"], st["h"], st["k"],
                                        st["select"])
    return st


def run_temperature(st: dict) -> float:
    """The run's training temperature at its restored step."""
    return temperature_at(st, st["step"]) if "temperature" in st else 1.0


def sort_values(vals):
    """Distinct values, numeric order when every one parses as a number."""
    uniq = list(dict.fromkeys(vals))
    try:
        return sorted(uniq, key=float)
    except (TypeError, ValueError):
        return sorted(uniq, key=str)


# ---------- noise2self partition score ----------

def n2s_score(gate: Float[Array, "b b"], D: Float[Array, "b b"],
              E: Float[Array, "b d"], ss: Sequence[int]) -> Float[Array, "S"]:
    """`pwak.pwak_l2` at each depth in `ss` from one walk: with
    G = wak(gate . D), l2(E, G^s E) / l2(E, mean E). `D` carries the
    zeroed diagonal (`pwak.affinity`)."""
    G = wak(gate * D)
    spread = jnp.mean((E - E.mean(0, keepdims=True)) ** 2)
    out, Y = [], E
    for s in range(1, max(ss) + 1):
        Y = G @ Y
        if s in ss:
            out.append(jnp.mean((E - Y) ** 2))
    return jnp.stack(out) / (spread + jnp.finfo(E.dtype).eps)


def layer_scores(P: Float[Array, "b h k"], E: Float[Array, "b d"],
                 heads: Int[Array, "m"], perms: Int[Array, "r b"],
                 ss: Sequence[int], tau: float) -> dict:
    """One batch, one layer. Per head of `heads`, the joint gate, the
    one-cell gate, and both again under each row permutation in `perms`:
      head (m, S), joint (S,), unpartitioned (S,),
      random (r, m, S), random_joint (r, S)."""
    P, E = P.astype(jnp.float32), E.astype(jnp.float32)
    D = affinity(E, tau)
    ss = tuple(ss)

    def per_head(Q):
        return jax.lax.map(lambda p: n2s_score(p @ p.T, D, E, ss),
                           Q[:, heads].transpose(1, 0, 2))

    def joint(Q):
        return n2s_score(jnp.einsum("ihc,jhc->ij", Q, Q), D, E, ss)

    rnd = jax.lax.map(lambda pm: (per_head(P[pm]), joint(P[pm])), perms)
    return {"head": per_head(P), "joint": joint(P),
            "unpartitioned": n2s_score(jnp.ones_like(D), D, E, ss),
            "random": rnd[0], "random_joint": rnd[1]}


layer_scores_jit = jax.jit(layer_scores, static_argnames=("ss", "tau"))


# ---------- split score, bandwidth, random-projection null ----------

JINV_SERIES = ("smoother", "trained_head", "trained_joint", "lsh_head",
               "lsh_joint", "lsh_kernel")


def split_views(E: Float[Array, "b d"], Q: Float[Array, "d d"]
                ) -> tuple[Float[Array, "b a"], Float[Array, "b c"]]:
    """The split score's two views of a batch: E centered, rotated by Q and
    cut into its first d // 2 coordinates (A, which builds every gate and
    kernel) and the rest (B, the values scored). The rotation spreads every
    direction of E over both views."""
    Z = (E - E.mean(0, keepdims=True)) @ Q
    a = E.shape[1] // 2
    return Z[:, :a], Z[:, a:]


def lsh_assign(E: Float[Array, "b d"], W: Float[Array, "n d k"]
               ) -> Float[Array, "n b k"]:
    """One-hot assignments of n random-projection heads: each puts a row in
    the cell of its largest projection on k Gaussian directions, after
    centering on the batch mean."""
    Ec = E - E.mean(0, keepdims=True)
    return jax.nn.one_hot(jnp.einsum("bd,ndk->nbk", Ec, W).argmax(-1),
                          W.shape[-1], dtype=E.dtype)


def joint_gate(Q: Float[Array, "n b k"]) -> Float[Array, "b b"]:
    """How many of the n heads put rows i and j in one cell. `wak` removes
    any constant factor, so this scores as the mean co-assignment, which
    over many random-projection heads estimates their collision kernel."""
    return jnp.einsum("nik,njk->ij", Q, Q)


def jinv_layer(P: Float[Array, "b h k"], E: Float[Array, "b d"],
               heads: Int[Array, "m"], Q: Float[Array, "d d"],
               W: Float[Array, "n d k"], W_A: Float[Array, "n a k"],
               V: Float[Array, "c d k"], V_A: Float[Array, "c a k"],
               ss: Sequence[int], taus: Sequence[float]) -> dict:
    """One batch, one layer: `n2s_score`'s ratio at every bandwidth in
    `taus` for each JINV_SERIES, keyed "<series>/<form>", each (T, S).
    The full form scores E on the kernel of E; the split form scores view
    B on gates and the kernel of view A (`split_views`). W draws the
    random-projection heads scored singly and jointly, V those of the
    collision kernel; W_A and V_A are their view-A counterparts."""
    P, E = P.astype(jnp.float32), E.astype(jnp.float32)
    A, B = split_views(E, Q)
    ss = tuple(ss)
    ones = jnp.ones((E.shape[0], E.shape[0]), E.dtype)
    R, R_A = lsh_assign(E, W), lsh_assign(A, W_A)
    gates = {"trained_joint/full": jnp.einsum("ihc,jhc->ij", P, P),
             "lsh_joint/full": joint_gate(R),
             "lsh_joint/split": joint_gate(R_A),
             "lsh_kernel/full": joint_gate(lsh_assign(E, V)),
             "lsh_kernel/split": joint_gate(lsh_assign(A, V_A))}
    out = {}
    for tau in taus:
        D, D_A = affinity(E, tau), affinity(A, tau)
        kernel = {"full": (D, E), "split": (D_A, B)}

        def single(Qs, form):
            K, Y = kernel[form]
            return jax.lax.map(lambda p: n2s_score(p @ p.T, K, Y, ss),
                               Qs).mean(0)

        row = {"smoother/full": n2s_score(ones, D, E, ss),
               "smoother/split": n2s_score(ones, D_A, B, ss),
               "trained_head/full": single(
                   P[:, heads].transpose(1, 0, 2), "full"),
               "lsh_head/full": single(R, "full"),
               "lsh_head/split": single(R_A, "split")}
        for key, g in gates.items():
            K, Y = kernel[key.split("/")[1]]
            row[key] = n2s_score(g, K, Y, ss)
        for key, v in row.items():
            out.setdefault(key, []).append(v)
    return {key: jnp.stack(v) for key, v in out.items()}


jinv_layer_jit = jax.jit(jinv_layer, static_argnames=("ss", "taus"))


def select_bandwidth(taus: Sequence[float],
                     split: Float[np.ndarray, "T"]) -> float:
    """The bandwidth whose smoother scores lowest under the split form:
    DEWAKSS's choice, valid because that score is J-invariant."""
    return taus[int(np.argmin(split))]


def isotropic_scores(taus: Sequence[float], b: int, d: int, nb: int,
                     ss: Sequence[int], seed: int = 0) -> dict:
    """The smoother's full and split scores on structureless rows
    (isotropic Gaussian, unit-normalized), each (T, S), averaged over nb
    batches of b. Any J-invariant predictor scores at least (b/(b-1))^2
    there at s=1, so the full form's shortfall below it is its leak."""
    rng = np.random.default_rng(seed)
    acc = {"full": [], "split": []}

    @jax.jit
    def one(E, Q):
        A, B = split_views(E, Q)
        ones = jnp.ones((b, b), E.dtype)
        return (jnp.stack([n2s_score(ones, affinity(E, t), E, tuple(ss))
                           for t in taus]),
                jnp.stack([n2s_score(ones, affinity(A, t), B, tuple(ss))
                           for t in taus]))

    for _ in range(nb):
        E = rng.standard_normal((b, d))
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
        f, s = one(jnp.asarray(E, jnp.float32), jnp.asarray(Q, jnp.float32))
        acc["full"].append(np.asarray(f))
        acc["split"].append(np.asarray(s))
    return {form: np.mean(v, 0) for form, v in acc.items()}


def head_subset(h: int, m: int) -> np.ndarray:
    """At most m head indices, evenly spaced over [0, h)."""
    return np.unique(np.linspace(0, h - 1, min(m, h)).round().astype(int))


def summarize(acc: dict) -> dict:
    """Per-batch `layer_scores` outputs (lists) -> one row of STATS per
    depth: head means/percentiles over heads, null mean and sd over every
    draw of every batch."""
    head = np.mean(acc["head"], 0)               # (m, S)
    rnd = np.concatenate(acc["random"], 0)       # (B*r, m, S)
    rj = np.concatenate([np.asarray(r) for r in acc["random_joint"]], 0)
    return {"partition": head.mean(0),
            "partition_p10": np.percentile(head, 10, 0),
            "partition_p90": np.percentile(head, 90, 0),
            "joint": np.mean(acc["joint"], 0),
            "random": rnd.mean(1).mean(0), "random_sd": rnd.mean(1).std(0),
            "random_joint": rj.mean(0), "random_joint_sd": rj.std(0),
            "unpartitioned": np.mean(acc["unpartitioned"], 0)}


def score_run(ckpt, st: dict, X_all: Float[np.ndarray, "n d"], cfg):
    """({layer: summarize(...)}, {layer: {key: (T, S)}} or None) for one
    run on the shared tail rows; the second is `jinv_layer` averaged over
    batches, under --jinv. Its draws come from their own generator, so
    --jinv leaves the first unchanged."""
    from autointerp import onto_acts_fn
    T = run_temperature(st)
    acts, _, _, meta, layer_inputs = onto_acts_fn(ckpt, 0, T, inputs=True)
    l, h, k = meta["l"], meta["h"], meta["k"]
    heads = jnp.asarray(head_subset(h, cfg.max_heads))
    rng = np.random.default_rng(cfg.seed)
    jrng = np.random.default_rng([cfg.seed, 1])
    rotations = {}  # input width -> Q, one per run

    def gauss(n, w):
        return jnp.asarray(jrng.standard_normal((n, w, k)), jnp.float32)

    unit0 = st.get("resid_gain") and not st.get("encoded")
    accs = [{key: [] for key in ("head", "joint", "unpartitioned", "random",
                                 "random_joint")} for _ in range(l)]
    jaccs = [{} for _ in range(l)]
    for i in range(0, len(X_all), cfg.b):
        X = jnp.asarray(X_all[i:i + cfg.b])
        P = acts(X).reshape(len(X), l, h, k)
        Us = layer_inputs(X)
        E0 = X / jnp.linalg.norm(X, axis=-1, keepdims=True) if unit0 else X
        perms = jnp.asarray(np.stack([rng.permutation(len(X))
                                      for _ in range(cfg.nulls)]))
        for L in range(l):
            E = E0 if L == 0 else Us[L - 1]
            r = layer_scores_jit(P[:, L], E, heads, perms, tuple(cfg.s),
                                 cfg.tau)
            for key, v in r.items():
                accs[L][key].append(np.asarray(v))
            if cfg.jinv:
                d_in = E.shape[1]
                if d_in not in rotations:
                    rotations[d_in] = jnp.asarray(np.linalg.qr(
                        jrng.standard_normal((d_in, d_in)))[0], jnp.float32)
                a = d_in // 2
                r = jinv_layer_jit(
                    P[:, L], E, heads, rotations[d_in],
                    gauss(cfg.lsh_heads, d_in), gauss(cfg.lsh_heads, a),
                    gauss(cfg.lsh_kernel, d_in), gauss(cfg.lsh_kernel, a),
                    tuple(cfg.s), tuple(cfg.taus))
                for key, v in r.items():
                    jaccs[L].setdefault(key, []).append(np.asarray(v))
    print(f"  {st['name']} step {st['step']} T={T:g}: {l}x{h}x{k}, "
          f"{len(heads)} heads/layer scored")
    jinv = ({L: {key: np.mean(v, 0) for key, v in a.items()}
             for L, a in enumerate(jaccs)} if cfg.jinv else None)
    return {L: summarize(a) for L, a in enumerate(accs)}, jinv


def n2s_rows(name: str, x, hue, scores: dict, ss: Sequence[int]):
    """Long-form CSV rows of one run's `score_run` output."""
    return [(name, x, hue, L, s, stat, float(row[stat][j]))
            for L, row in scores.items() for j, s in enumerate(ss)
            for stat in STATS]


def read_rows(path) -> list:
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)
        return [(r[0], r[1], r[2], int(r[3]), int(r[4]), r[5], float(r[6]))
                for r in rd]


JINV_COLS = ("run", "x", "hue", "layer", "s", "tau", "form", "series",
             "value", "selected")


def jinv_rows(name: str, x, hue, jinv: dict, ss: Sequence[int],
              taus: Sequence[float]) -> list:
    """Long-form jinv.csv rows of one run's `score_run` split scores;
    `selected` marks, per layer and depth, the bandwidth
    `select_bandwidth` picks."""
    rows = []
    for L, d in jinv.items():
        for j, s in enumerate(ss):
            best = select_bandwidth(taus, d["smoother/split"][:, j])
            for key, v in d.items():
                series, form = key.split("/")
                rows += [(name, x, hue, L, s, tau, form, series,
                          float(v[t, j]), int(tau == best))
                         for t, tau in enumerate(taus)]
    return rows


def read_jinv_rows(path) -> list:
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)
        return [(r[0], r[1], r[2], int(r[3]), int(r[4]), float(r[5]), r[6],
                 r[7], float(r[8]), int(r[9])) for r in rd]


def print_jinv(rows, tau_ref: float):
    """Per run, layer and depth: the smoother's full/split scores and its
    gain over the smoother at tau_ref, then every other series' gain over
    the smoother at the same tau and form (positive: the series predicts
    better). * marks the selected bandwidth."""
    val = {(r[0], r[3], r[4], r[5], r[6], r[7]): r[8] for r in rows}
    sel = {(r[0], r[3], r[4]): r[5] for r in rows if r[9]}
    taus = sort_values([r[5] for r in rows])[::-1]
    cols = [(srs, f) for srs in JINV_SERIES[1:] for f in ("full", "split")
            if any(r[7] == srs and r[6] == f for r in rows)]
    for (run, L, s), best in sel.items():
        print(f"\n  {run} layer {L} s={s}: split score selects tau={best:g}")
        print(f"  {'tau':>7} {'smoother f/s':>15} {'vs ' + str(tau_ref):>15} "
              + " ".join(f"{a[:13] + ' ' + b[0]:>15}" for a, b in cols))
        for tau in taus:
            sm = [val.get((run, L, s, tau, f, "smoother")) for f in
                  ("full", "split")]
            ref = [val.get((run, L, s, tau_ref, f, "smoother")) for f in
                   ("full", "split")]
            gain = [f"{r_ - v:+.4f}" if None not in (r_, v) else "--"
                    for r_, v in zip(ref, sm)]
            cells = []
            for srs, f in cols:
                v = val.get((run, L, s, tau, f, srs))
                base = sm[0] if f == "full" else sm[1]
                cells.append(f"{base - v:+.4f}" if v is not None else "--")
            mark = "*" if tau == best else " "
            print(f"  {tau:>6g}{mark} {sm[0]:>7.4f}/{sm[1]:<7.4f} "
                  f"{'/'.join(gain):>15} "
                  + " ".join(f"{c:>15}" for c in cells))


# ---------- curve figure ----------

COLORS = ("#2a6fdb", "#e0731f", "#1f9e6e", "#b8357a", "#6b4fd0", "#8a6d1f")


def draw_curve(rows, s: int, xkey: str, hkey: Optional[str], title: str,
               path):
    """One panel per layer at depth s; see the module docstring for the
    series. Every baseline is drawn on every panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = [r for r in rows if r[4] == s]
    xs = sort_values([r[1] for r in rows])
    hues = sort_values([r[2] for r in rows])
    layers = sorted({r[3] for r in rows})
    xpos = {x: i for i, x in enumerate(xs)}
    val = {(r[0], r[3], r[5]): r[6] for r in rows}
    runs = {}  # name -> (x, hue)
    for r in rows:
        runs[r[0]] = (r[1], r[2])

    fig, axes = plt.subplots(
        2, len(layers), sharey="row", sharex=True, squeeze=False,
        figsize=(2.2 + 3.0 * len(layers), 7.0),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})
    for ax, dax, L in zip(axes[0], axes[1], layers):
        best = (np.inf, None)
        for c, hu in enumerate(hues):
            col = COLORS[c % len(COLORS)]
            names = sorted([n for n, (x, h_) in runs.items()
                            if h_ == hu and (n, L, "partition") in val],
                           key=lambda n: xpos[runs[n][0]])
            if not names:
                continue
            xi = np.array([xpos[runs[n][0]] for n in names], float)
            # small hue offset so overlapping error bars stay readable
            xi = xi + (c - (len(hues) - 1) / 2) * 0.08
            g = lambda st: np.array([val[(n, L, st)] for n in names])
            part, rnd, sd = g("partition"), g("random"), g("random_sd")
            unp = g("unpartitioned")
            for a, ref in ((ax, 0.0), (dax, unp)):
                a.fill_between(xi, g("partition_p10") - ref,
                               g("partition_p90") - ref,
                               color=col, alpha=0.15, lw=0)
                a.plot(xi, part - ref, "-o", color=col, lw=2, ms=6, zorder=3)
                a.plot(xi, g("joint") - ref, "-.", marker="D", mfc="none",
                       color=col, lw=1.2, ms=5)
                a.errorbar(xi, rnd - ref, yerr=sd, fmt=":s", color=col,
                           lw=1.2, ms=4, mfc="white", capsize=3)
            ax.plot(xi, unp, "--", color="black", lw=1.2, marker="_", ms=12,
                    zorder=2)
            j = int(np.argmin(part))
            if part[j] < best[0]:
                best = (part[j], (xi[j], unp[j], rnd[j]))
        ax.axhline(1.0, color="0.6", ls=":", lw=1)
        dax.axhline(0.0, color="black", ls="--", lw=1.2)
        if best[1] is not None:
            ax.plot(best[1][0], best[0], marker="*", ms=16, color="gold",
                    mec="black", zorder=5)
            ax.text(0.03, 0.03, f"argmin partition {best[0]:.3f}\n"
                    f"same run: unpartitioned {best[1][1]:.3f}\n"
                    f"               random {best[1][2]:.3f}",
                    transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox=dict(fc="white", ec="0.7", alpha=0.9))
        dax.set_xticks(range(len(xs)), [str(x) for x in xs], fontsize=8)
        dax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_title(f"layer {L}", fontsize=10)
        dax.set_xlabel(xkey)
        for a in (ax, dax):
            a.grid(True, axis="y", alpha=0.3)
    axes[0][0].set_ylabel(f"noise2self error / batch variance (s={s})")
    axes[1][0].set_ylabel("minus the same run's\nunpartitioned score\n"
                          "(< 0: partition helps)")
    handles = [
        Line2D([], [], color="0.2", ls="-", marker="o", lw=2,
               label="partition (head mean; band 10-90%)"),
        Line2D([], [], color="0.2", ls="-.", marker="D", mfc="none",
               label="joint gate (training stat)"),
        Line2D([], [], color="0.2", ls=":", marker="s", mfc="white",
               label="random partition, matched sizes (±sd)"),
        Line2D([], [], color="black", ls="--", label="unpartitioned"),
        Line2D([], [], color="0.6", ls=":", label="batch mean (=1)"),
        Line2D([], [], color="gold", marker="*", ls="", mec="black", ms=12,
               label="argmin partition")]
    if hkey:
        handles += [Line2D([], [], color=COLORS[c % len(COLORS)], lw=3,
                           label=f"{hkey}={hu}") for c, hu in enumerate(hues)]
    fig.legend(handles=handles, loc="lower center", fontsize=8,
               ncol=min(len(handles), 5), bbox_to_anchor=(0.5, -0.02),
               frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, wspace=0.08,
                        bottom=0.14 if len(handles) <= 5 else 0.17)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- grid ----------

def pareto_fvu(path, point: str = "soft") -> Optional[float]:
    """The soft (`onto m=<k> (soft)`) or hard (`onto hard (argmax)`) FVU
    row of a pareto.csv, None when the file or row is absent."""
    if not Path(path).exists():
        return None
    pat = r"onto m=\d+ \(soft\)" if point == "soft" else r"onto hard.*"
    with open(path) as f:
        for row in csv.DictReader(f):
            if re.fullmatch(pat, row["label"]):
                return float(row["fvu_w"])
    return None


def sae_fvus(path) -> list:
    """(label, coeffs, fvu) of every sae row of a pareto.csv."""
    with open(path) as f:
        return [(r["label"], int(r["coeffs"]), float(r["fvu_w"]))
                for r in csv.DictReader(f) if r["label"].startswith("sae")]


def sae_frontier(sae: Sequence) -> list:
    """The SAE rows no other row beats at fewer or equal coefficients,
    ordered by coefficients."""
    out, best = [], np.inf
    for row in sorted(sae, key=lambda r: (r[1], r[2])):
        if row[2] < best:
            out.append(row)
            best = row[2]
    return out


def spread_labels(y: Float[np.ndarray, "n"], gap: float
                  ) -> Float[np.ndarray, "n"]:
    """Label positions at least `gap` apart, each as near its y as the
    others allow (one upward sweep then one downward, in y order)."""
    order = np.argsort(y)
    z = np.asarray(y, float)[order].copy()
    for i in range(1, len(z)):
        z[i] = max(z[i], z[i - 1] + gap)
    shift = (z - np.asarray(y, float)[order]).mean()
    z -= shift
    for i in range(1, len(z)):
        z[i] = max(z[i], z[i - 1] + gap)
    out = np.empty_like(z)
    out[order] = z
    return out


def assemble_grid(records: Sequence[dict], rkey: str, ckey: str):
    """(row values, col values, mean value (R, C), count (R, C), names).
    Cells no record covers are NaN with count 0."""
    rv = sort_values([r[rkey] for r in records])
    cv = sort_values([r[ckey] for r in records])
    V = np.zeros((len(rv), len(cv)))
    n = np.zeros((len(rv), len(cv)), int)
    names = [[[] for _ in cv] for _ in rv]
    for r in records:
        i, j = rv.index(r[rkey]), cv.index(r[ckey])
        V[i, j] += r["value"]
        n[i, j] += 1
        names[i][j].append(r["name"])
    with np.errstate(invalid="ignore"):
        V = np.where(n > 0, V / np.maximum(n, 1), np.nan)
    return rv, cv, V, n, names


def draw_grid(records, rkey: str, ckey: str, title: str, path,
              sae: Sequence = ()):
    """FVU heatmap on a log colour scale; missing cells hatched blank;
    argmin outlined; SAE FVUs as ticks on the colour bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    rv, cv, V, n, _ = assemble_grid(records, rkey, ckey)
    cap = {}
    for r in records:
        cap.setdefault((r[rkey], r[ckey]), (r["coeffs"], r["bits"]))
    fin = V[np.isfinite(V)]
    # the scale is the grid's own, so SAEs far off it do not wash it out
    lo, hi = fin.min() / 1.5, fin.max() * 1.5
    if hi / lo < 10:
        mid = np.sqrt(lo * hi)
        lo, hi = mid / np.sqrt(10), mid * np.sqrt(10)
    norm = LogNorm(lo, hi)
    cmap = plt.get_cmap("viridis").with_extremes(bad="white")

    fig, ax = plt.subplots(figsize=(2 + 1.35 * len(cv), 1.6 + 1.0 * len(rv)))
    im = ax.imshow(np.ma.masked_invalid(V), cmap=cmap, norm=norm,
                   aspect="auto")
    for i in range(len(rv)):
        for j in range(len(cv)):
            if n[i, j] == 0:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       hatch="///", ec="0.8", lw=0))
                ax.text(j, i, "no run", ha="center", va="center",
                        fontsize=8, color="0.5")
                continue
            c, b = cap[(rv[i], cv[j])]
            light = norm(V[i, j]) > 0.55  # viridis is light at the top
            txt = f"{V[i, j]:.3g}\n{c} c / {b} b"
            if n[i, j] > 1:
                txt += f"\n(n={n[i, j]})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if light else "white")
    i, j = np.unravel_index(np.nanargmin(V), V.shape)
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                           ec="red", lw=3))
    ax.set_xticks(range(len(cv)), [str(c) for c in cv])
    ax.set_yticks(range(len(rv)), [str(r) for r in rv])
    ax.set_xlabel(ckey)
    ax.set_ylabel(rkey)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.11)
    from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
    cb.ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.ax.set_title("FVU (whitened)", fontsize=8, loc="left")
    cb.ax.yaxis.set_ticks_position("left")
    front = sae_frontier(sae)
    if front:
        # out-of-range SAEs sit at the bar's end, marked < or >
        span = np.log10(norm.vmax / norm.vmin)
        clip = [min(max(f, norm.vmin), norm.vmax) for _, _, f in front]
        ys = spread_labels(np.log10(clip), 0.07 * span)
        for (label, c, f), fc, y in zip(front, clip, ys):
            mark = "<" if f < norm.vmin else ">" if f > norm.vmax else ""
            cb.ax.axhline(fc, color="red", lw=1.5)
            cb.ax.annotate(f"SAE {label[4:]} ({c} c) {mark}{f:.3g}",
                           (1.0, fc),
                           xycoords=cb.ax.get_yaxis_transform(),
                           xytext=(1.4, 10 ** y),
                           textcoords=cb.ax.get_yaxis_transform(),
                           fontsize=7, va="center",
                           arrowprops=dict(arrowstyle="-", color="red",
                                           lw=0.6))
    ax.set_title(title + "\n(cell: FVU, nominal coefficients / index bits;"
                 " red box = argmin)", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def substrate_root(run) -> Path:
    """The run's directory one level under `out/` (data/out/sonar,
    data/out/gpt2_l8), which every pareto.csv for its substrate sits
    under; the run's parent when it is not under an `out/` directory."""
    parts = Path(run).parts
    if "out" in parts[:-1]:
        i = len(parts) - 1 - parts[::-1].index("out")
        if i + 1 < len(parts) - 1:
            return Path(*parts[:i + 2])
    return Path(run).parent


def pareto_candidates(run, name: str, pareto_dir=None) -> list:
    """Where a run's pareto.csv may be, in search order: <run>/pareto/,
    then pareto_<name>/ beside the run (the gpt2_l8 layout), then
    pareto_<name>/ under --pareto-dir, by default the substrate root's
    pareto/ (the sonar layout). Only exact name matches, and by default
    only within the run's own substrate: a pareto.csv does not record its
    checkpoint, and run names repeat across substrates (ste_h76 is both a
    SONAR and a GPT-2 run), so a folder named for another run, or for a
    namesake on another substrate, is never taken for this one."""
    if pareto_dir is None:
        pareto_dir = substrate_root(run) / "pareto"
    cands = [Path(run) / "pareto" / "pareto.csv",
             Path(run).parent / f"pareto_{name}" / "pareto.csv",
             Path(pareto_dir) / f"pareto_{name}" / "pareto.csv"]
    seen, out = set(), []
    for c in cands:
        key = c.resolve()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def grid_records(cfg, runs) -> list:
    """One record per run: settings + FVU + its source."""
    recs, todo = [], []
    for run in runs:
        st = run_settings(run)
        f, src = None, None
        if not cfg.recompute:
            for cand in pareto_candidates(run, st["name"], cfg.pareto_dir):
                f = pareto_fvu(cand, cfg.point)
                if f is not None:
                    src = str(cand)
                    break
        st["value"], st["source"] = f, src
        (recs if f is not None else todo).append(st)
    if todo:
        import pareto
        from types import SimpleNamespace
        mm = np.load(cfg.cache, mmap_mode="r")
        X_eval = np.asarray(mm[-cfg.eval_rows:], dtype=np.float32)
        w = np.load(cfg.mse_weights) if cfg.mse_weights \
            else np.ones(mm.shape[1])
        assert w.shape == (mm.shape[1],), (
            f"--mse-weights {cfg.mse_weights} is {w.shape[0]} wide but the "
            f"cache {cfg.cache} is {mm.shape[1]}: pass the weights that "
            f"match the cache (e.g. the activation cache's .mse_weights.npy)")
        base_w = (X_eval.var(0) * w).mean()
        for st in todo:
            ns = SimpleNamespace(
                ckpt=st["path"], step=0, temperature=run_temperature(st), ms=[], code="dev",
                origin="uniform", b=2 * cfg.b, cache=cfg.cache)
            pts = pareto.onto_points(ns, X_eval, w, base_w)
            lab = "(soft)" if cfg.point == "soft" else "hard"
            st["value"] = next(p[3] for p in pts if lab in p[0])
            st["source"] = "computed"
            recs.append(st)
    for st in recs:
        print(f"  {st['name']:<28} {cfg.grid[0]}={st.get(cfg.grid[0])} "
              f"{cfg.grid[1]}={st.get(cfg.grid[1])} FVU {st['value']:.4f} "
              f"[{st['source']}]")
    return recs


GRID_COLS = ("name", "row", "col", "value", "coeffs", "bits", "source")


def write_grid_csv(path, recs, rkey, ckey):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(GRID_COLS + (rkey, ckey))
        for r in recs:
            wr.writerow([r["name"], r[rkey], r[ckey], r["value"],
                         r["coeffs"], r["bits"], r["source"],
                         rkey, ckey])


def read_grid_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    rkey, ckey = rows[0][-2:]
    recs = [{"name": r[0], rkey: r[1], ckey: r[2], "value": float(r[3]),
             "coeffs": int(r[4]), "bits": int(r[5]), "source": r[6]}
            for r in rows[1:]]
    return recs, rkey, ckey


# ---------- main ----------

def main():
    cfg = parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    sae = sae_fvus(cfg.sae_csv) if cfg.sae_csv else []

    if cfg.grid:
        if cfg.replot:
            recs, rkey, ckey = read_grid_csv(out / "grid.csv")
        else:
            rkey, ckey = cfg.grid
            recs = grid_records(cfg, expand_runs(cfg.runs, cfg.exclude))
            write_grid_csv(out / "grid.csv", recs, rkey, ckey)
        draw_grid(recs, rkey, ckey, f"{out.name}: held-out FVU ({cfg.point})",
                  out / "grid.png", sae)
        print(f"-> {out / 'grid.png'}")
        return

    if cfg.replot:
        rows = read_rows(out / "n2s.csv")
        meta = json.loads((out / "n2s.json").read_text())
        cfg.x, cfg.hue, cfg.tau = meta["x"], meta["hue"], meta["tau"]
        if cfg.jinv:
            print_jinv(read_jinv_rows(out / "jinv.csv"), cfg.tau)
    else:
        runs = expand_runs(cfg.runs, cfg.exclude)
        assert cfg.rows % cfg.b == 0, "--rows must be a multiple of --b"
        mm = np.load(cfg.cache, mmap_mode="r")
        n = mm.shape[0]
        rng = np.random.default_rng(cfg.seed)
        idx = np.sort(rng.choice(np.arange(n - cfg.eval_rows, n), cfg.rows,
                                 replace=False))
        # shuffled so each graph batch spans the whole tail
        X_all = np.asarray(mm[idx], dtype=np.float32)[
            rng.permutation(len(idx))]
        print(f"{len(runs)} runs on {cfg.rows} tail rows, batches of {cfg.b}")
        rows, jrows = [], []
        for run in runs:
            st = run_settings(run)
            sc, jinv = score_run(run, st, X_all, cfg)
            x, hue = st.get(cfg.x), st.get(cfg.hue) if cfg.hue else ""
            rows += n2s_rows(st["name"], x, hue, sc, cfg.s)
            if jinv is not None:
                jrows += jinv_rows(st["name"], x, hue, jinv, cfg.s,
                                   cfg.taus)
            jax.clear_caches()
        with open(out / "n2s.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["run", "x", "hue", "layer", "s", "stat", "value"])
            wr.writerows(rows)
        meta = {"x": cfg.x, "hue": cfg.hue, "tau": cfg.tau, "rows": cfg.rows,
                "b": cfg.b, "nulls": cfg.nulls, "max_heads": cfg.max_heads,
                "cache": cfg.cache, "eval_rows": cfg.eval_rows, "runs": runs}
        if cfg.jinv:
            meta["jinv"] = {"taus": cfg.taus, "lsh_heads": cfg.lsh_heads,
                            "lsh_kernel": cfg.lsh_kernel}
        (out / "n2s.json").write_text(json.dumps(meta, indent=1))
        print(f"-> {out / 'n2s.csv'}")
        if cfg.jinv:
            with open(out / "jinv.csv", "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(JINV_COLS)
                wr.writerows(jrows)
            iso = isotropic_scores(cfg.taus, cfg.b, X_all.shape[1],
                                   cfg.rows // cfg.b, cfg.s, cfg.seed)
            floor = (cfg.b / (cfg.b - 1)) ** 2
            with open(out / "jinv_isotropic.csv", "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["tau", "s", "full", "split", "floor"])
                for t, tau in enumerate(cfg.taus):
                    for j, s in enumerate(cfg.s):
                        wr.writerow([tau, s, iso["full"][t, j],
                                     iso["split"][t, j], floor])
            print_jinv(jrows, cfg.tau)
            print(f"\n  isotropic rows (b={cfg.b}, d={X_all.shape[1]}), "
                  f"smoother full/split, floor {floor:.4f} at s=1:")
            for t, tau in enumerate(cfg.taus):
                print(f"  {tau:>7g} " + "  ".join(
                    f"s={s}: {iso['full'][t, j]:.4f}/{iso['split'][t, j]:.4f}"
                    for j, s in enumerate(cfg.s)))
            print(f"-> {out / 'jinv.csv'}, {out / 'jinv_isotropic.csv'}")
    for s in sorted({r[4] for r in rows}):
        path = out / f"n2s_s{s}.png"
        draw_curve(rows, s, cfg.x, cfg.hue, f"{out.name}: held-out "
                   f"noise2self partition score, tau={cfg.tau}", path)
        print(f"-> {path}")


if __name__ == "__main__":
    main()

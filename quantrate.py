"""Operational rate--distortion curves: quantize every continuous number a
code sends, entropy-code the symbols, and sweep the step.

pareto.py puts index bits and continuous coefficients on separate axes,
which cannot rank a code that sends coefficients against one that does
not: a real number carries unbounded information. Here every continuous
number a code sends is quantized and every symbol entropy-coded, so each
code becomes a curve of whitened FVU against total bits per sample, on the
same axis as a hard code's index bits and pareto.py's Gaussian reference.

Quantizer: uniform with a deadzone. A coefficient z with origin o is sent
as the symbol q = round((z - o) / step) and decoded as o + q * step, so a
coefficient within half a step of its origin is not sent at all. Each
coefficient's step is the shared --steps value divided by the whitened
norm of the direction it multiplies, so one step costs every coefficient
about the same whitened squared error (step^2 / 12 at fine steps). The
origin is 0 for sparse codes (ReLU, top-k, L1 and grouped top-1 SAEs),
whose zeros are then free, and for dense ones (grouped-softmax SAEs and an
Ontologizer's soft code) the mean over the first --fit-rows rows of
--fit-cache, the E[p] origin of pareto.py's deviation codes. Those rows,
which also give the gains' statistics, come from the scored cache short
of its tail by default; scoring a fresh cache, point --fit-cache at the
training one.

Rate: each slot's symbol (0 = at the origin) is coded with its own
empirical distribution over the evaluation rows, so a code's bits per
sample are the sum of its slots' entropies (plug-in, Miller-Madow
corrected). A grouped top-1 SAE codes one symbol per head instead, its
winner and level or silence, since its latents are exclusive within a
head. The CSV also gives the convention of pareto.py's index bits: the
nonzero slots named as a set, log2 C(slots, s) per sample plus the entropy
of s (for grouped top-1, which heads fire and each one's winner), with the
values entropy-coded given that support. For a hard code that column is
the nominal l*h*log2(k), and the first is the realized per-head entropy.

Ontologizers run closed-loop: each layer's quantized output is what the
next layer's residual sees, so later layers correct earlier quantization
error, as they must for a decoder that only has the quantized code. A soft
code quantizes its assignments around E[p], with per-entry steps from the
entries' decoded directions (autointerp.entry_directions). A hard code
sends its entries and, under gain-shape, one gain per layer, quantized
around its mean with a step of --gain-steps times its standard deviation;
"pinned" sends no gain, and a soft code with gain-shape sends its gains
at the finest --gain-steps.

Every code goes through the same codec and none is re-optimized for it: no
refit of coefficients after quantization (refit.py bounds what that could
buy), no rate-distortion-optimized choice of support. Each curve is what
the trained code achieves under this codec, not the best it could; the
rates leave out the cost of sending the coders' tables.

  uv run python quantrate.py                  # sae_conv SAEs + both Ontologizers
  uv run python quantrate.py --sae P.npz ... --onto CKPT@TEMPERATURE ...
  uv run python quantrate.py --cache data/sonar_embeddings/mc4_fresh.npy \\
      --fit-cache data/sonar_embeddings/mc4_4M.npy \\
      --out data/out/sonar/quantrate_fresh          # rows no model trained on
  uv run python quantrate.py --replot --out <dir>

Writes <out>/quantrate.csv (label, kind, setting, nonzero, bits,
bits_subset, fvu_w; one row per code and step, "float" rows unquantized),
quantrate.json (the settings and each code's readouts at --target-bits and
at the pinned hard code's FVU) and quantrate.png. Ontologizer checkpoints
restore on GPU JAX only.
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from jaxtyping import Float, Int

import pareto

LN2 = math.log(2.0)
SPAN = 1 << 24        # a symbol must lie in (-SPAN/2, SPAN/2)
LEVELS = 1 << 16      # grouped top-1 symbol: 1 + winner * LEVELS + level

DEFAULT_ONTO = ["data/out/sonar/multilingual/ste_h76_init01@0.00015",
                "data/out/sonar/multilingual/resid_nc@0.03"]
FIELDS = ["label", "kind", "setting", "nonzero", "bits", "bits_subset",
          "fvu_w"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sae", nargs="*", default=None,
                   help="params.npz paths (default: data/out/sonar/sae_conv/*/)")
    p.add_argument("--onto", nargs="*", default=DEFAULT_ONTO,
                   help="Ontologizer checkpoint dirs as PATH@TEMPERATURE")
    p.add_argument("--steps", type=float, nargs="+",
                   default=[2.0 ** (e / 2) for e in range(-6, -29, -1)],
                   help="quantizer steps in whitened units, coarse to fine "
                        "(the MSE weights have mean 1, so one coefficient's "
                        "whitened contribution is of order 0.001-0.1)")
    p.add_argument("--gain-steps", type=float, nargs="+",
                   default=[2.0 ** -e for e in range(-1, 6)],
                   help="gain steps in units of each layer's gain sd")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy",
                   help="cache whose tail rows are scored")
    p.add_argument("--fit-cache", default=None,
                   help="cache whose head rows origins and gain statistics "
                        "are measured on (default: --cache, short of its "
                        "scored tail); the training cache when --cache is "
                        "a fresh one")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="rows scored, from the tail of --cache (on the "
                        "training cache, the sae.py eval split)")
    p.add_argument("--fit-rows", type=int, default=65536,
                   help="head rows of --fit-cache that origins and gain "
                        "statistics are measured on")
    p.add_argument("--b", type=int, default=1024)
    p.add_argument("--target-bits", type=float, default=1900.0)
    p.add_argument("--target-fvu", type=float, default=None,
                   help="default: the first hard code's pinned-gain FVU")
    p.add_argument("--out", default="data/out/sonar/quantrate")
    p.add_argument("--replot", action="store_true",
                   help="redraw quantrate.png from quantrate.csv")
    return p.parse_args(argv)


# ---------- pure helpers ----------

class SymbolCounts:
    """Each slot's histogram of integer symbols over the rows added so far,
    0 meaning "at the origin". Only nonzero symbols are stored, as unique
    (slot, symbol) keys per batch, so a sparse code costs memory in
    proportion to what it sends; each slot's zero count is the remainder."""

    def __init__(self, slots: int):
        self.slots = slots
        self.n = 0
        self.parts: list = []
        self.nnz: list = []

    def add(self, slot: Int[np.ndarray, "b K"], sym: Int[np.ndarray, "b K"]
            ) -> None:
        """One batch as (slot, symbol) pairs per row. Pairs whose symbol is
        0 are skipped, so a fixed-width slice of a sparse code's largest
        coefficients and a dense code's full width are both valid input."""
        keep = sym != 0
        v = sym[keep].astype(np.int64)
        if v.size and np.abs(v).max() >= SPAN // 2:
            raise ValueError(f"symbol {np.abs(v).max()} out of range")
        keys = slot[keep].astype(np.int64) * SPAN + v + SPAN // 2
        self.parts.append(np.unique(keys, return_counts=True))
        self.nnz.append(keep.sum(1))
        self.n += sym.shape[0]

    def add_dense(self, sym: Int[np.ndarray, "b slots"]) -> None:
        self.add(np.broadcast_to(np.arange(sym.shape[1]), sym.shape), sym)

    def merged(self) -> tuple:
        """(slot, count) of every distinct nonzero (slot, symbol) pair."""
        if not self.parts:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        keys = np.concatenate([u for u, _ in self.parts])
        cnts = np.concatenate([c for _, c in self.parts])
        u, inv = np.unique(keys, return_inverse=True)
        return u // SPAN, np.bincount(inv, weights=cnts).astype(np.int64)

    def nonzeros(self) -> Int[np.ndarray, "n"]:
        return (np.concatenate(self.nnz) if self.nnz
                else np.zeros(0, np.int64))


def plugin_bits(counts: Int[np.ndarray, "s"], n: int) -> float:
    """Plug-in entropy, in bits, of a distribution given by its counts over
    n draws, plus the Miller-Madow correction (K - 1) / (2 n ln 2) for its
    K observed symbols."""
    c = np.asarray(counts, np.float64)
    c = c[c > 0]
    if not c.size:
        return 0.0
    p = c / n
    return float(-(p * np.log2(p)).sum() + (len(c) - 1) / (2 * n * LN2))


def slot_bits(sc: SymbolCounts) -> tuple:
    """Per slot, the entropy of its symbol, zero included, and of whether
    the symbol is nonzero, both Miller-Madow corrected: (H(q_j), H_b(f_j)).
    H(q_j) - H_b(f_j) is the entropy of the value given that it is sent."""
    n, S = sc.n, sc.slots
    slot, c = sc.merged()
    c = c.astype(np.float64)
    h = np.bincount(slot, weights=-(c / n) * np.log2(c / n), minlength=S)
    active = np.bincount(slot, weights=c, minlength=S)
    kinds = np.bincount(slot, minlength=S).astype(np.float64)
    zero = n - active

    def term(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x > 0, -(x / n) * np.log2(x / n), 0.0)

    mm = 1.0 / (2 * n * LN2)
    H = h + term(zero) + np.maximum(kinds + (zero > 0) - 1, 0) * mm
    Hb = term(active) + term(zero) + ((active > 0) & (zero > 0)) * mm
    return H, Hb


def subset_bits(nnz: Int[np.ndarray, "n"], slots: int, groups: int = 0
                ) -> float:
    """Mean bits that name each row's nonzero slots as a set, plus the
    entropy of how many there are: log2 C(slots, s) per row, or for a
    grouped top-1 code, which heads fire and each one's winner,
    log2 C(groups, s) + s * log2(slots / groups)."""
    vals, cnt = np.unique(np.asarray(nnz), return_counts=True)
    if groups:
        per = [pareto.log2_choose(groups, int(s))
               + int(s) * math.log2(slots // groups) for s in vals]
    else:
        per = [pareto.log2_choose(slots, int(s)) for s in vals]
    return (float(np.dot(per, cnt) / cnt.sum())
            + plugin_bits(cnt, int(cnt.sum())))


def group_symbols(win: Int[np.ndarray, "b g"], lev: Int[np.ndarray, "b g"]
                  ) -> Int[np.ndarray, "b g"]:
    """One symbol per head of a grouped top-1 code: 0 when the head is
    silent, else 1 + winner * LEVELS + level."""
    if lev.size and (lev.min() < 0 or lev.max() >= LEVELS):
        raise ValueError("grouped top-1 level out of range")
    return np.where(lev != 0, 1 + win.astype(np.int64) * LEVELS + lev, 0)


def whitened_norms(W: Float[np.ndarray, "f d"], w: Float[np.ndarray, "d"]
                   ) -> Float[np.ndarray, "f"]:
    """Each direction's norm under the whitened inner product, floored so
    that a zero direction gets an infinite step (its coefficient is never
    sent) rather than a division by zero."""
    return np.maximum(np.sqrt((np.asarray(W, np.float64) ** 2 * w).sum(-1)),
                      1e-12)


def curve(rows: Sequence[dict], key: str = "bits") -> list:
    """(bits, fvu) of a code's quantized rows, sorted by bits."""
    pts = [(float(r[key]), float(r["fvu_w"])) for r in rows]
    return sorted((b, f) for b, f in pts
                  if math.isfinite(b) and b > 0 and f > 0)


def fvu_at_bits(pts: Sequence[tuple], target: float) -> Optional[float]:
    """FVU at `target` bits, interpolated linearly in log-log between the
    two sweep points that bracket it; None off the ends of the curve."""
    for (b0, f0), (b1, f1) in zip(pts, pts[1:]):
        if b0 <= target <= b1:
            if b1 == b0:
                return min(f0, f1)
            t = math.log(target / b0) / math.log(b1 / b0)
            return math.exp(math.log(f0) + t * math.log(f1 / f0))
    return None


def bits_at_fvu(pts: Sequence[tuple], target: float) -> Optional[float]:
    """Fewest bits at which the curve reaches `target` FVU, interpolated
    in log-log at its first crossing; None if it never does, and the first
    point's bits if even that is below the target."""
    if pts and pts[0][1] <= target:
        return pts[0][0]
    for (b0, f0), (b1, f1) in zip(pts, pts[1:]):
        if f0 > target >= f1:
            t = math.log(target / f0) / math.log(f1 / f0)
            return math.exp(math.log(b0) + t * math.log(b1 / b0))
    return None


# ---------- codes ----------

def eval_rows(cfg):
    mm = np.load(cfg.cache, mmap_mode="r")
    X = np.asarray(mm[-cfg.eval_rows:], dtype=np.float32)
    w = np.load(cfg.mse_weights) if cfg.mse_weights else np.ones(mm.shape[1])
    return X, w.astype(np.float64), float((X.var(0) * w).mean())


def fit_count(n_fit: int, fit_rows: int, eval_rows: int, same: bool) -> int:
    """How many head rows of the fit cache origins and gain statistics are
    measured on: --fit-rows, short of the scored tail when the fit cache is
    the scored one."""
    return max(0, min(fit_rows, n_fit - (eval_rows if same else 0)))


def fit_source(cfg, d: int):
    """A callable yielding the fit rows in batches."""
    path = cfg.fit_cache or cfg.cache
    fit = np.load(path, mmap_mode="r")
    if fit.shape[1] != d:
        raise SystemExit(f"--fit-cache {path} is {fit.shape[1]} wide, the "
                         f"scored cache {d}")
    same = Path(path).resolve() == Path(cfg.cache).resolve()
    n = fit_count(fit.shape[0], cfg.fit_rows, cfg.eval_rows, same)
    print(f"fit rows: the first {n} of {path}", flush=True)

    def batches():
        for i in range(0, n - cfg.b + 1, cfg.b):
            yield np.asarray(fit[i:i + cfg.b], dtype=np.float32)
    return batches


def fvu_of(err: Float[np.ndarray, "d"], n: int, w, base_w) -> float:
    return float(((err / n) * w).mean() / base_w)


def sae_rows(path, cfg, fit_batches, X, w, base_w) -> list:
    import jax
    import jax.numpy as jnp
    import sae

    meta_path = Path(path).parent / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        m, topk = meta["m"], meta["topk"]
        groups, gfn = meta["groups"], meta["group_fn"]
    else:
        (m, topk), groups, gfn = pareto.parse_sae_name(path), 0, "top1"
    params = {k: jnp.asarray(v) for k, v in np.load(path).items()}
    kind = ("sae_dense" if groups and gfn == "softmax"
            else "sae_group" if groups else "sae")
    norms = whitened_norms(np.asarray(params["W_dec"]), w)
    label = f"sae {Path(path).parent.name}"

    encode = jax.jit(lambda p, x: sae.encode(p, x, topk, groups, gfn))
    origin = np.zeros(m, np.float32)
    if kind == "sae_dense":
        acc, nb = np.zeros(m), 0
        for Xb in fit_batches():
            acc += np.asarray(encode(params, jnp.asarray(Xb))).mean(0)
            nb += 1
        origin = (acc / nb).astype(np.float32)
    o = jnp.asarray(origin)

    # unquantized pass: the float FVU, and the widest support a row has,
    # which bounds the nonzero symbols of a sparse code at any step
    err, K = np.zeros(X.shape[1]), 0
    for i in range(0, len(X) - cfg.b + 1, cfg.b):
        Xb = jnp.asarray(X[i:i + cfg.b])
        z = encode(params, Xb)
        err += np.asarray(((sae.decode(params, z) - Xb) ** 2).sum(0))
        K = max(K, int((z != 0).sum(-1).max()))
    n = (len(X) // cfg.b) * cfg.b
    rows = [dict(label=label, kind=kind, setting="float", nonzero=K,
                 bits=float("nan"), bits_subset=float("nan"),
                 fvu_w=fvu_of(err, n, w, base_w))]

    @jax.jit
    def batch(p, x, step):
        z = sae.encode(p, x, topk, groups, gfn)
        q = jnp.round((z - o) / step)
        e = ((sae.decode(p, o + q * step) - x) ** 2).sum(0)
        qi = q.astype(jnp.int32)
        if kind == "sae_dense":
            return e, qi, qi
        if kind == "sae_group":
            g = qi.reshape(qi.shape[0], groups, -1)
            win = jnp.argmax(g, -1)
            return e, win, jnp.take_along_axis(g, win[..., None], -1)[..., 0]
        _, idx = jax.lax.top_k(jnp.abs(z), K)
        return e, idx, jnp.take_along_axis(qi, idx, -1)

    span = m // groups if groups else 0
    for s in cfg.steps:
        step = jnp.asarray((s / norms).astype(np.float32))
        sc = SymbolCounts(m)
        sg = SymbolCounts(groups) if kind == "sae_group" else None
        err = np.zeros(X.shape[1])
        for i in range(0, len(X) - cfg.b + 1, cfg.b):
            e, a, v = batch(params, jnp.asarray(X[i:i + cfg.b]), step)
            err += np.asarray(e)
            a, v = np.asarray(a), np.asarray(v)
            if kind == "sae_dense":
                sc.add_dense(v)
            elif kind == "sae_group":
                sc.add(np.arange(groups) * span + a, v)
                sg.add_dense(group_symbols(a, v))
            else:
                sc.add(a, v)
        H, Hb = slot_bits(sc)
        values = float((H - Hb).sum())
        if kind == "sae_group":
            bits = float(slot_bits(sg)[0].sum())
            subset = subset_bits(sg.nonzeros(), m, groups) + values
            nonzero = float(sg.nonzeros().mean())
        else:
            bits = float(H.sum())
            subset = subset_bits(sc.nonzeros(), m) + values
            nonzero = float(sc.nonzeros().mean())
        rows.append(dict(label=label, kind=kind, setting=f"{s:.6g}",
                         nonzero=nonzero, bits=bits, bits_subset=subset,
                         fvu_w=fvu_of(err, n, w, base_w)))
        print(f"  {label} step {s:.4g}: {nonzero:.1f} nonzero, "
              f"{bits:.0f} bits ({subset:.0f} subset), "
              f"FVU {rows[-1]['fvu_w']:.4f}", flush=True)
    return rows


def onto_forward(module, X, T, origin, pstep, gmean, gstep, quant_p: bool,
                 gain: str):
    """The model's forward with its code quantized as the decoder receives
    it (pareto.py's probe, closed-loop). Returns the reconstruction, the
    assignment symbols (entries for a hard code, deviation levels for a
    quantized soft one) and the gain symbols. `gain` is "float", "pin" or
    "quant"."""
    import jax.numpy as jnp

    E, _ = module.encode(X, 0.0, None)
    R = module.resid(E)
    E_in = module.constinput(E)
    psyms, gsyms = [], []
    for i, de in enumerate(module.dictencs):
        U, G = de.gainshape_in(E_in)
        P = de.dict.cluster(de.classifier(U), T)
        if quant_p:
            q = jnp.round((P - origin[i]) / pstep[i])
            P = origin[i] + q * pstep[i]
            psyms.append(q.astype(jnp.int32).reshape(X.shape[0], -1))
        else:
            psyms.append(jnp.argmax(P, -1).astype(jnp.int32))
        if G is not None and gain != "float":
            live = gstep[i] > 0
            qg = jnp.zeros_like(G) if gain == "pin" else jnp.where(
                live, jnp.round((G - gmean[i]) / jnp.where(live, gstep[i], 1.0)),
                0.0)
            G = gmean[i] + qg * gstep[i]
            gsyms.append(qg[..., 0].astype(jnp.int32))
        R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
        if i < module.l - 1:
            E_in = module.nextinput(X, R, P.reshape(P.shape[0], -1))
    gs = (jnp.stack(gsyms, -1) if gsyms
          else jnp.zeros((X.shape[0], 0), jnp.int32))
    return module.decode(R), jnp.concatenate(psyms, -1), gs


def onto_stats(module, X, T):
    """Per-layer mean assignment (l, h, k) and per-layer gains (b, l), the
    gains 0 under no gain-shape, from the unquantized forward."""
    import jax.numpy as jnp

    E, _ = module.encode(X, 0.0, None)
    R = module.resid(E)
    E_in = module.constinput(E)
    Pm, Gs = [], []
    for i, de in enumerate(module.dictencs):
        U, G = de.gainshape_in(E_in)
        P = de.dict.cluster(de.classifier(U), T)
        Pm.append(P.mean(0))
        Gs.append(jnp.zeros(X.shape[0]) if G is None else G[..., 0])
        R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
        if i < module.l - 1:
            E_in = module.nextinput(X, R, P.reshape(P.shape[0], -1))
    return jnp.stack(Pm), jnp.stack(Gs, -1)


def onto_rows(spec, cfg, fit_batches, X, w, base_w) -> list:
    import jax
    import jax.numpy as jnp

    ckpt, T = spec.rsplit("@", 1)
    T = float(T)
    model, raw, step_no = pareto.load_onto(ckpt, 0)
    l, h, k = model.l, model.h, model.k
    hard = model.select in ("ste", "argmax")
    label = f"onto {Path(ckpt).name}"
    kind = "onto_hard" if hard else "onto_soft"
    print(f"{label}: step {step_no}, {l}x{h}x{k}, select={model.select}, "
          f"resid_gain={model.resid_gain}, T={T}", flush=True)

    stats = jax.jit(lambda p, x: model.apply({"params": p}, x, T,
                                             method=onto_stats))
    acc, G1, G2, nb = np.zeros((l, h, k)), np.zeros(l), np.zeros(l), 0
    for Xb in fit_batches():
        Pm, G = stats(raw, jnp.asarray(Xb))
        G = np.asarray(G, np.float64)
        acc += np.asarray(Pm)
        G1 += G.mean(0)
        G2 += (G ** 2).mean(0)
        nb += 1
    origin = acc / nb
    gmean = G1 / nb
    gsd = np.sqrt(np.maximum(G2 / nb - gmean ** 2, 0.0))
    gsd = np.where(gsd > 1e-6 * np.maximum(gmean, 1e-12), gsd, 0.0)

    norms = np.ones((l, h, k))
    if not hard:
        from autointerp import onto_acts_fn, entry_directions
        _, embed, _, _ = onto_acts_fn(ckpt, step_no, T)
        dirs, _ = entry_directions(embed, l, h, k)
        norms = whitened_norms(dirs, w).reshape(l, h, k)

    fwd = jax.jit(
        lambda p, x, o, ps, gm, gst, quant_p, gain: model.apply(
            {"params": p}, x, T, o, ps, gm, gst, quant_p, gain,
            method=onto_forward),
        static_argnames=("quant_p", "gain"))
    o = jnp.asarray(origin, jnp.float32)
    gm = jnp.asarray(gmean[:, None, None], jnp.float32)
    live_gains = int((gsd > 0).sum())

    def run(pstep, gstep, quant_p, gain):
        ps = jnp.asarray(pstep, jnp.float32)
        gst = jnp.asarray(np.asarray(gstep)[:, None, None], jnp.float32)
        sp = SymbolCounts(l * h * k if quant_p else l * h)
        sgn = SymbolCounts(l)
        err = np.zeros(X.shape[1])
        for i in range(0, len(X) - cfg.b + 1, cfg.b):
            Y, P, G = fwd(raw, jnp.asarray(X[i:i + cfg.b]), o, ps, gm, gst,
                          quant_p, gain)
            err += np.asarray(((Y - jnp.asarray(X[i:i + cfg.b])) ** 2).sum(0))
            sp.add_dense(np.asarray(P))
            if G.shape[-1]:
                sgn.add_dense(np.asarray(G))
        n = sp.n
        H, Hb = slot_bits(sp)
        gbits = float(slot_bits(sgn)[0].sum()) if sgn.n else 0.0
        if quant_p:
            bits = float(H.sum()) + gbits
            subset = (subset_bits(sp.nonzeros(), l * h * k)
                      + float((H - Hb).sum()) + gbits)
            nonzero = float(sp.nonzeros().mean())
        else:
            bits = float(H.sum()) + gbits
            subset = l * h * math.log2(k) + gbits
            nonzero = float(live_gains if gain != "pin" else 0)
        return nonzero, bits, subset, fvu_of(err, n, w, base_w)

    rows = []
    zero_g = np.zeros(l)
    if hard:
        settings = [("float", "float", zero_g), ("pinned", "pin", zero_g)]
        if model.resid_gain:
            settings += [(f"gain {r:.6g}", "quant", r * gsd)
                         for r in cfg.gain_steps]
        for name, gain, gstep in settings:
            nonzero, bits, subset, fvu = run(norms, gstep, False, gain)
            if gain == "float":
                bits = subset = float("nan")
            rows.append(dict(label=label, kind=kind, setting=name,
                             nonzero=nonzero, bits=bits, bits_subset=subset,
                             fvu_w=fvu))
            print(f"  {label} {name}: {bits:.1f} bits ({subset:.1f} nominal), "
                  f"FVU {fvu:.4f}", flush=True)
        return rows

    gain = "quant" if model.resid_gain else "float"
    gstep = cfg.gain_steps[-1] * gsd
    nonzero, _, _, fvu = run(norms, zero_g, False, "float")
    rows.append(dict(label=label, kind=kind, setting="float",
                     nonzero=l * h * k, bits=float("nan"),
                     bits_subset=float("nan"), fvu_w=fvu))
    for s in cfg.steps:
        nonzero, bits, subset, fvu = run(s / norms, gstep, True, gain)
        rows.append(dict(label=label, kind=kind, setting=f"{s:.6g}",
                         nonzero=nonzero, bits=bits, bits_subset=subset,
                         fvu_w=fvu))
        print(f"  {label} step {s:.4g}: {nonzero:.1f} nonzero, {bits:.0f} "
              f"bits ({subset:.0f} subset), FVU {fvu:.4f}", flush=True)
    return rows


# ---------- output ----------

def readouts(rows: Sequence[dict], target_bits: float,
             target_fvu: Optional[float]) -> dict:
    """Per code: its float FVU, its FVU at `target_bits` and the bits at
    which it reaches `target_fvu`, under both rate conventions."""
    out = {}
    for label in dict.fromkeys(r["label"] for r in rows):
        mine = [r for r in rows if r["label"] == label]
        q = [r for r in mine if r["setting"] != "float"]
        fl = [r["fvu_w"] for r in mine if r["setting"] == "float"]
        res = {"kind": mine[0]["kind"], "float_fvu": fl[0] if fl else None}
        for key in ("bits", "bits_subset"):
            pts = curve(q, key)
            res[f"fvu_at_target_{key}"] = fvu_at_bits(pts, target_bits)
            res[f"{key}_at_target_fvu"] = (
                bits_at_fvu(pts, target_fvu) if target_fvu else None)
        out[label] = res
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)


def read_csv(path) -> list:
    with open(path) as f:
        return [{k: (v if k in ("label", "kind", "setting") else float(v))
                 for k, v in r.items()} for r in csv.DictReader(f)]


STYLE = dict(onto_hard=dict(color="tab:green", marker="D", lw=1.8),
             onto_soft=dict(color="tab:olive", marker="o", lw=1.4),
             sae_dense=dict(color="tab:purple", marker="v", lw=1.0),
             sae_group=dict(color="tab:gray", marker="v", lw=1.0),
             sae=dict(color="tab:blue", marker="s", lw=1.0))


def plot(rows, ref, target_bits, path):
    """(a) every code's curve over the range the codes span; (b) the same
    near `target_bits`, each curve labeled where it crosses the panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2))
    windows = ((a, (50, 3e4), (5e-5, 2.0)),
               (b, (target_bits / 2.2, target_bits * 2.2), (0.06, 0.6)))
    for ax, (x0, x1), (y0, y1) in windows:
        if ref is not None:
            ax.plot(*ref, color="0.5", ls="--", lw=1,
                    label="Gaussian reference")
        for label in dict.fromkeys(r["label"] for r in rows):
            mine = [r for r in rows if r["label"] == label]
            pts = curve([r for r in mine if r["setting"] != "float"])
            if not pts:
                continue
            ax.plot(*zip(*pts), ms=3, alpha=0.85,
                    **STYLE.get(mine[0]["kind"], STYLE["sae"]))
            inside = [(x, y) for x, y in pts if x0 <= x <= x1 and y0 <= y <= y1]
            if inside and ax is b:
                ax.annotate(label.split(" ", 1)[1], inside[-1], fontsize=7,
                            xytext=(4, 0), textcoords="offset points",
                            va="center")
        ax.axvline(target_bits, color="0.7", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel("total bits per sample (entropy-coded)")
        ax.grid(True, which="both", alpha=0.25)
    a.set_ylabel("FVU (whitened), evaluation tail")
    handles = [plt.Line2D([], [], **{k: v for k, v in s.items() if k != "lw"},
                          lw=s["lw"], label=n)
               for n, s in (("hard Ontologizer", STYLE["onto_hard"]),
                            ("soft Ontologizer", STYLE["onto_soft"]),
                            ("SAE", STYLE["sae"]),
                            ("grouped top-1 SAE", STYLE["sae_group"]),
                            ("grouped softmax SAE", STYLE["sae_dense"]))]
    a.legend(handles=handles + a.get_legend_handles_labels()[0][:1],
             fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    cfg = parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    if cfg.replot:
        settings = json.loads((out / "quantrate.json").read_text())
        rows = read_csv(out / "quantrate.csv")
        ref = settings.get("reference")
        plot(rows, (ref["bits"], ref["fvu"]) if ref else None,
             settings["target_bits"], out / "quantrate.png")
        print(f"-> {out / 'quantrate.png'}")
        return

    X, w, base_w = eval_rows(cfg)
    fit_batches = fit_source(cfg, X.shape[1])
    paths = cfg.sae
    if paths is None:
        paths = sorted(Path("data/out/sonar/sae_conv").glob("*/params.npz"))
    rows = []
    for spec in cfg.onto:
        rows += onto_rows(spec, cfg, fit_batches, X, w, base_w)
    for path in paths:
        print(f"sae: {path}", flush=True)
        rows += sae_rows(path, cfg, fit_batches, X, w, base_w)
    write_csv(out / "quantrate.csv", rows)

    target_fvu = cfg.target_fvu
    if target_fvu is None:
        pinned = [r for r in rows
                  if r["kind"] == "onto_hard" and r["setting"] == "pinned"]
        target_fvu = pinned[0]["fvu_w"] if pinned else None
    ref_bits = np.geomspace(10, 40000, 300)
    ref_fvu = pareto.gaussian_reference(pareto.whitened_spectrum(X, w),
                                        ref_bits)
    res = readouts(rows, cfg.target_bits, target_fvu)
    (out / "quantrate.json").write_text(json.dumps({
        "settings": {k: v for k, v in vars(cfg).items() if k != "replot"},
        "target_bits": cfg.target_bits, "target_fvu": target_fvu,
        "reference": {"bits": ref_bits.tolist(), "fvu": ref_fvu.tolist()},
        "readouts": res}, indent=2, default=str))
    plot(rows, (ref_bits, ref_fvu), cfg.target_bits, out / "quantrate.png")

    print(f"\n{'code':<28} {'float':>8} {'@' + str(int(cfg.target_bits)):>8} "
          f"{'(subset)':>9} {'bits@FVU':>9} {'(subset)':>9}")
    fmt = lambda v, f: f"{v:{f}}" if v is not None else "-"
    for label, r in res.items():
        print(f"{label:<28} {fmt(r['float_fvu'], '8.4f'):>8} "
              f"{fmt(r['fvu_at_target_bits'], '8.4f'):>8} "
              f"{fmt(r['fvu_at_target_bits_subset'], '9.4f'):>9} "
              f"{fmt(r['bits_at_target_fvu'], '9.0f'):>9} "
              f"{fmt(r['bits_subset_at_target_fvu'], '9.0f'):>9}")
    print(f"\n(target FVU {target_fvu}) -> {out}")


if __name__ == "__main__":
    main()

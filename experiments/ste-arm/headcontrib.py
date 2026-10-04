"""Do two models' heads WRITE the same thing?

`partition.py` compares what a head separates. A head is also a
vector-valued function of its input, and two heads could carve the rows
differently while adding the same thing to the output, or carve them
alike while adding different things. This compares what they add.

A head's contribution on a row is what the decoder emits for that head's
slice of the layer output alone, gain applied, less `decode(0)` so a
biased decoder's constant is not credited to every head. The decoder is
linear, so contributions sum to the reconstruction. Contributions are
measured in the whitened output frame the objective scores, centered
over rows, and compared by cosine between whole (rows, d_out) matrices;
heads are matched one-to-one by maximizing total cosine (Hungarian).

Centering is not optional. The dictionary is non-negative, so every head
carries a shared offset, and uncentered cosine reads that offset as
agreement -- its row-shuffled null reaches 0.85.

Reported against two references: B's rows shuffled, which breaks the
pairing while keeping both marginals, and (run separately, with `--b`
set to `--a` and `--step-b` earlier) the same run against an earlier
checkpoint as the positive control.

Bigger heads reproduce better, and layer-0 heads are the biggest, so the
script also reports agreement against head size: per layer, and as
regressions on size and a layer-0 indicator. Within one stack the two
are confounded, so `--ref-a/--ref-b` scores a second seed pair -- a flat
arm, all heads on the raw input -- and compares each layer with
reference heads of the same size.

Size itself splits into the layer's input gain, shared by all of a
layer's heads, and the head's decoded atoms; the script reports both,
since only the atoms can vary between heads of one layer.

  uv run python experiments/ste-arm/headcontrib.py \\
      --a data/out/gpt2_l8/ste_h20_cat128 \\
      --b data/out/gpt2_l8/ste_h20_cat128-43

  uv run python experiments/ste-arm/headcontrib.py \\
      --cache data/sonar_embeddings/mc4_4M.npy \\
      --mse-weights data/out/sonar/mse_weights.npy --rows 4096 \\
      --a data/out/sonar/multilingual/ste_h76_init01 \\
      --b data/out/sonar/multilingual/ste_h76_i01_s43 \\
      --ref-a data/out/sonar/multilingual/ste_l1_h380_i01 \\
      --ref-b data/out/sonar/multilingual/ste_l1_h380_i01_s43
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from scipy.optimize import linear_sum_assignment

from pareto import load_onto


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="checkpoint dir")
    p.add_argument("--b", required=True, help="checkpoint dir")
    p.add_argument("--step-a", type=int, default=0)
    p.add_argument("--step-b", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.00015,
                   help="classification temperature; irrelevant under "
                        "select='ste', whose cluster is the argmax")
    p.add_argument("--cache", default="data/activations/gpt2_l8.npy")
    p.add_argument("--mse-weights",
                   default="data/activations/gpt2_l8.mse_weights_matched.npy",
                   help="per-dimension weights defining the whitened frame")
    # (heads, rows, d_out) float32 per model: 100 heads x 8192 rows x 768
    # is 2.5 GB, so rows is bounded by memory rather than by the holdout
    p.add_argument("--rows", type=int, default=8192)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ref-a", default=None,
                   help="reference seed pair (e.g. a flat arm) scored the "
                        "same way, for agreement at matched head size")
    p.add_argument("--ref-b", default=None)
    p.add_argument("--band", type=float, default=0.1,
                   help="relative widening of each layer's size range when "
                        "selecting size-matched reference heads")
    cfg = p.parse_args()
    if (cfg.ref_a is None) != (cfg.ref_b is None):
        p.error("--ref-a and --ref-b go together")
    return cfg


def contributions(ckpt: str, step: int, T: float,
                  X: Float[np.ndarray, "n d_in"],
                  w_sqrt: Float[np.ndarray, "d_out"], batch: int
                  ) -> Tuple[Float[np.ndarray, "heads n d_out"],
                             Int[np.ndarray, "heads"], int]:
    """Whitened per-head output contributions, with the layer index of
    each head and the step actually loaded."""
    model, raw, used = load_onto(ckpt, step)
    params = {"params": raw}

    def probe(module, Xb: Float[Array, "b d_in"]
              ) -> Float[Array, "b lh d_out"]:
        E, _ = module.encode(Xb, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        zero = module.decode(jnp.zeros_like(R))
        out = []
        for i, de in enumerate(module.dictencs):
            U, G = de.gainshape_in(Ein)
            P = de.dict.cluster(de.classifier(U), T)
            Y = de.head_outputs(U, P)                          # (b, h, dh)
            h = Y.shape[1]
            # one copy of the layer output per head, every other head
            # zeroed, so `combine` places it as the layer itself would
            # (summed or concatenated)
            Yh = Y[:, None] * jnp.eye(h, dtype=Y.dtype)[None, :, :, None]
            Gh = None if G is None else G[:, None]
            Ch = de.gained(de.dict.combine(Yh), Gh)            # (b, h, e)
            out.append(module.decode(Ch) - zero[:, None])
            R = R + de.gained(de.dict.combine(Y), G)
            if i < module.l - 1:
                Ein = module.nextinput(Xb, R, P.reshape(Xb.shape[0], -1))
        return jnp.concatenate(out, 1)

    f = jax.jit(lambda p, x: model.apply(p, x, method=probe))
    C = np.concatenate([np.asarray(f(params, jnp.asarray(X[i:i + batch])))
                        for i in range(0, len(X), batch)], 0)
    C *= w_sqrt[None, None, :]
    layer = np.repeat(np.arange(model.l), C.shape[1] // model.l)
    return C.transpose(1, 0, 2), layer, used


def atoms_and_codes(ckpt: str, step: int, T: float,
                    X: Float[np.ndarray, "n d_in"],
                    w_sqrt: Float[np.ndarray, "d_out"], batch: int
                    ) -> Tuple[Float[np.ndarray, "heads k d_out"],
                               Int[np.ndarray, "heads n"],
                               Float[np.ndarray, "l n"],
                               Int[np.ndarray, "heads"], int]:
    """Every head's decoded atoms in the whitened frame (each isolated as
    `contributions` isolates a head, less `decode(0)`), the entry each
    head selects on each row, and each layer's input gain per row.
    Returns (atoms, selections, gains, layer of each head, step loaded)."""
    model, raw, used = load_onto(ckpt, step)
    params = {"params": raw}
    k = model.k
    if model.scaled or model.fiber_rank:
        # a head's contribution is then gain x router(x) x atom (plus a
        # fiber term), so "gain x atom" no longer decomposes it exactly
        raise ValueError(
            f"{ckpt}: the atom decomposition assumes contribution = gain x "
            f"atom, which the router (scaled) and fibers break")

    def atoms(module) -> Float[Array, "lh k d_out"]:
        zero = module.decode(jnp.zeros((1, module.e_dec), module.dtype))
        out = []
        for de in module.dictencs:
            h = de.dict.h
            # row j selects entry j in every head; then isolate each head
            # as `contributions` does
            P = jnp.zeros((k, h, k), module.dtype)
            P = P.at[jnp.arange(k), :, jnp.arange(k)].set(1.0)
            Y = de.dict.hfwd(P)                                # (k, h, dh)
            Yh = Y[:, None] * jnp.eye(h, dtype=Y.dtype)[None, :, :, None]
            out.append(module.decode(de.dict.combine(Yh)) - zero)
        return jnp.concatenate(out, 1).transpose(1, 0, 2)

    def codes(module, Xb: Float[Array, "b d_in"]
              ) -> Tuple[Int[Array, "l b h"], Float[Array, "l b"]]:
        E, _ = module.encode(Xb, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        idx, gains = [], []
        for i, de in enumerate(module.dictencs):
            U, G = de.gainshape_in(Ein)
            P = de.dict.cluster(de.classifier(U), T)
            idx.append(jnp.argmax(P, -1))
            gains.append(jnp.ones(Xb.shape[:1], module.dtype) if G is None
                         else G[:, 0])
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
            if i < module.l - 1:
                Ein = module.nextinput(Xb, R, P.reshape(Xb.shape[0], -1))
        return jnp.stack(idx), jnp.stack(gains)

    # jitted so XLA fuses the per-head isolation: eagerly it materializes
    # (k, h, h, e), 35 GB for a 380-head layer
    D = np.asarray(jax.jit(lambda p: model.apply(p, method=atoms))(params)
                   ) * w_sqrt                                  # (H, k, d)
    f = jax.jit(lambda p, x: model.apply(p, x, method=codes))
    parts = [f(params, jnp.asarray(X[i:i + batch]))
             for i in range(0, len(X), batch)]
    I = np.concatenate([np.asarray(a) for a, _ in parts], 1)    # (l, n, h)
    G = np.concatenate([np.asarray(g) for _, g in parts], 1)    # (l, n)
    l, n, h = I.shape
    I = I.transpose(0, 2, 1).reshape(l * h, n)
    return D, I, G, np.repeat(np.arange(l), h), used


def decompose(ckpt: str, step: int, T: float, X: Float[np.ndarray, "n d_in"],
              w_sqrt: Float[np.ndarray, "d_out"], batch: int
              ) -> Tuple[Float[np.ndarray, "heads"], Float[np.ndarray, "heads"],
                         Float[np.ndarray, "heads"]]:
    """Splits each head's contribution size into its layer's input gain
    and its own atoms. A head's contribution on a row is its gain times
    the decoded atom it selected, so its centered energy is, up to the
    gain's small variation, gain^2 * spread^2 -- spread being the
    usage-weighted RMS distance of its decoded atoms from their
    usage-weighted mean. Returns (gain RMS, usage-weighted decoded atom
    norm, spread), all in the whitened frame.

    Gain is one number per layer per row, shared by every head in the
    layer, so across layers it cannot be told apart from the layer
    itself; only the atom terms vary between heads of one layer."""
    D, I, G, layer, _ = atoms_and_codes(ckpt, step, T, X, w_sqrt, batch)
    k, n = D.shape[1], I.shape[1]
    usage = np.stack([np.bincount(row, minlength=k) / n for row in I])
    gain = np.sqrt((G ** 2).mean(1))[layer]
    norm = (usage * np.linalg.norm(D, axis=-1)).sum(1)
    mean = (usage[..., None] * D).sum(1)
    spread = np.sqrt((usage * ((D - mean[:, None]) ** 2).sum(-1)).sum(1))
    return gain, norm, spread


def cosine_matrix(A: Float[np.ndarray, "ha n d"],
                  B: Float[np.ndarray, "hb n d"], center: bool
                  ) -> Float[np.ndarray, "ha hb"]:
    """Cosine between every head of A and every head of B, each head's
    contribution flattened over (rows, d_out)."""
    if center:
        A = A - A.mean(1, keepdims=True)
        B = B - B.mean(1, keepdims=True)
    a = A.reshape(A.shape[0], -1)
    b = B.reshape(B.shape[0], -1)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-30)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-30)
    return a @ b.T


def report(name: str, M: Float[np.ndarray, "ha hb"],
           la: Int[np.ndarray, "ha"], lb: Int[np.ndarray, "hb"]
           ) -> Tuple[Float[np.ndarray, "m"], Int[np.ndarray, "m"],
                      Int[np.ndarray, "m"]]:
    """Prints the assignment's summary; returns (matched cosine, rows,
    cols)."""
    r, c = linear_sum_assignment(-M)
    v = M[r, c]
    print(f"  {name:<22} mean {v.mean():+.4f}  median {np.median(v):+.4f}"
          f"  max {v.max():+.4f}  same-layer {(la[r] == lb[c]).mean():.0%}")
    return v, r, c


def compare(a: str, step_a: int, b: str, step_b: int,
            X: Float[np.ndarray, "n d_in"], w_sqrt: Float[np.ndarray, "d_out"],
            cfg: argparse.Namespace
            ) -> Tuple[Float[np.ndarray, "heads"], Int[np.ndarray, "heads"],
                       Float[np.ndarray, "heads"]]:
    """Scores A against B and prints the matching. Returns, per head of A
    in A's own order, its matched centered cosine, its layer, and its
    size: centered contribution energy as a share of the whitened target
    variance, so it measures what varies rather than the shared offset."""
    A, la, sa = contributions(a, step_a, cfg.temperature, X, w_sqrt,
                              cfg.batch)
    B, lb, sb = contributions(b, step_b, cfg.temperature, X, w_sqrt,
                              cfg.batch)
    print(f"A {Path(a).name} step {sa}: {A.shape[0]} heads")
    print(f"B {Path(b).name} step {sb}: {B.shape[0]} heads")
    print(f"{len(X)} rows")

    Xw = X * w_sqrt
    share = (((A - A.mean(1, keepdims=True)) ** 2).sum((1, 2))
             / ((Xw - Xw.mean(0)) ** 2).sum())

    rng = np.random.default_rng(cfg.seed)
    Bs = B[:, rng.permutation(B.shape[1])]
    nl = int(la.max()) + 1
    for center in (True, False):
        print(f"\n {'centered' if center else 'uncentered (offset-dominated)'}")
        M = cosine_matrix(A, B, center)
        v, r, c = report("matched", M, la, lb)
        report("null (rows shuffled)", cosine_matrix(A, Bs, center), la, lb)
        if not center:
            continue
        matched = np.empty_like(v)
        matched[r] = v
        for i in range(nl if nl > 1 else 0):
            m = la[r] == i
            print(f"    layer {i}: cosine {v[m].mean():+.4f}  lands in "
                  f"layer {np.bincount(lb[c][m], minlength=nl)}")
    return matched, la, share


def _fit(cols: list, y: Float[np.ndarray, "m"]
         ) -> Tuple[Float[np.ndarray, "p"], float]:
    """Least squares with an intercept; returns (coefficients, R^2)."""
    Z = np.column_stack([np.ones_like(y)] + cols)
    beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    return beta, 1.0 - (y - Z @ beta).var() / y.var()


def size_report(v: Float[np.ndarray, "heads"], layer: Int[np.ndarray, "heads"],
                share: Float[np.ndarray, "heads"]) -> None:
    """Does agreement track head size, and can size stand in for layer?

    Within one stack each layer's heads tend to be nearly one size, with
    ranges that do not overlap across layers, so size and layer are
    confounded and the split depends on the model: additive on cosine,
    or multiplicative (on log cosine). Both are printed; the layer-0
    coefficient is what survives size."""
    nl = int(layer.max()) + 1
    print(f"\n size (centered contribution share of target variance)")
    for i in range(nl):
        m = layer == i
        print(f"    layer {i}: median {100 * np.median(share[m]):.3f}%  "
              f"[{100 * share[m].min():.3f}, {100 * share[m].max():.3f}]  "
              f"cosine {v[m].mean():+.4f}")
    ls = np.log(share)
    print(f"  corr(cosine, log size) {np.corrcoef(v, ls)[0, 1]:+.3f}")
    for i in range(nl if nl > 1 else 0):
        m = layer == i
        if m.sum() > 2:
            print(f"    within layer {i}: {np.corrcoef(v[m], ls[m])[0, 1]:+.3f}")
    if nl == 1:
        return
    l0 = (layer == 0).astype(float)
    # floor before the log: a matched cosine can sit at or below zero
    lv = np.log(np.maximum(v, 1e-4))
    for name, y in (("cosine", v), ("log cosine", lv)):
        for label, cols in (("size", [ls]), ("layer 0", [l0]),
                            ("size + layer 0", [ls, l0])):
            beta, r2 = _fit(cols, y)
            print(f"  {name:<10} ~ {label:<14} R^2 {r2:.3f}  "
                  f"coef {np.array2string(beta[1:], precision=4)}")


def atom_report(v: Float[np.ndarray, "heads"], layer: Int[np.ndarray, "heads"],
                share: Float[np.ndarray, "heads"],
                gain: Float[np.ndarray, "heads"],
                norm: Float[np.ndarray, "heads"],
                spread: Float[np.ndarray, "heads"], var: float) -> None:
    """Which part of size tracks agreement: the layer's gain or the
    head's atoms? `var` is the whitened target variance per row, so
    gain^2 * spread^2 / var should reproduce `share`; the check is
    printed. Gain is constant within a layer, so the within-layer
    correlations are the atoms' alone, and across layers the
    atom-spread + layer-0 fit gives the lift that atoms leave to the
    layer."""
    nl = int(layer.max()) + 1
    approx = (gain * spread) ** 2 / var
    print(f"\n size = gain^2 * atom spread^2: corr(log) "
          f"{np.corrcoef(np.log(share), np.log(approx))[0, 1]:+.3f}, "
          f"median ratio {np.median(share / approx):.3f}")
    for i in range(nl):
        m = layer == i
        print(f"    layer {i}: gain {np.median(gain[m]):.4g}  "
              f"atom norm {np.median(norm[m]):.4g}  "
              f"spread {np.median(spread[m]):.4g}  "
              f"spread/norm {np.median(spread[m] / norm[m]):.3f}")
    lv = np.log(np.maximum(v, 1e-4))
    for label, x in (("atom norm", np.log(norm)), ("atom spread", np.log(spread))):
        within = [np.corrcoef(lv[layer == i], x[layer == i])[0, 1]
                  for i in range(nl) if (layer == i).sum() > 2]
        print(f"  corr(log cosine, log {label}) {np.corrcoef(lv, x)[0, 1]:+.3f}"
              f"  within layers " + " ".join(f"{t:+.2f}" for t in within))
    if nl == 1:
        return
    beta, r2 = _fit([np.log(spread), (layer == 0).astype(float)], lv)
    print(f"  log cosine ~ atom spread + layer 0  R^2 {r2:.3f}  "
          f"coef {np.array2string(beta[1:], precision=3)}  "
          f"(layer-0 lift {np.exp(beta[2]):.1f}x at equal atoms)")


def reference_report(v: Float[np.ndarray, "heads"],
                     layer: Int[np.ndarray, "heads"],
                     share: Float[np.ndarray, "heads"],
                     rv: Float[np.ndarray, "rheads"],
                     rshare: Float[np.ndarray, "rheads"], band: float) -> None:
    """Agreement at matched size against a reference pair -- a flat arm,
    whose heads all see the raw input with nothing downstream, gives the
    depth-free curve. Fitted curves extrapolate badly past the
    reference's range, so this reports reference heads inside each
    layer's own size range widened by `band`, with their count."""
    q = np.quantile(rshare, [0, .25, .5, .75, .9, 1])
    print(f"\n reference: cosine by size quantile "
          f"(corr with log size {np.corrcoef(rv, np.log(rshare))[0, 1]:+.3f})")
    for lo, hi in zip(q[:-1], q[1:]):
        m = (rshare >= lo) & (rshare <= hi)
        print(f"    {100 * lo:.3f}-{100 * hi:.3f}%: {rv[m].mean():+.4f}"
              f"  n={m.sum()}")
    print(f"\n at matched size (reference heads within +-{band:.0%} of the "
          f"layer's range)")
    for i in range(int(layer.max()) + 1):
        m = layer == i
        lo, hi = share[m].min() * (1 - band), share[m].max() * (1 + band)
        rm = (rshare >= lo) & (rshare <= hi)
        ref = (f"{rv[rm].mean():+.4f}  n={rm.sum()}  gap "
               f"{rv[rm].mean() / max(v[m].mean(), 1e-9):.1f}x"
               if rm.any() else "outside the reference's range")
        print(f"    layer {i}: {v[m].mean():+.4f}  reference {ref}")


def main() -> None:
    cfg = parse_args()
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float32)
    w_sqrt = np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
    v, layer, share = compare(cfg.a, cfg.step_a, cfg.b, cfg.step_b, X,
                              w_sqrt, cfg)
    size_report(v, layer, share)
    Xw = X * w_sqrt
    try:
        parts = decompose(cfg.a, cfg.step_a, cfg.temperature, X, w_sqrt,
                          cfg.batch)
    except ValueError as e:
        print(f"\n(atom report skipped: {e})")
    else:
        atom_report(v, layer, share, *parts,
                    ((Xw - Xw.mean(0)) ** 2).sum() / len(X))
    if cfg.ref_a:
        print(f"\n=== reference pair")
        rv, _, rshare = compare(cfg.ref_a, 0, cfg.ref_b, 0, X, w_sqrt, cfg)
        reference_report(v, layer, share, rv, rshare, cfg.band)


if __name__ == "__main__":
    main()

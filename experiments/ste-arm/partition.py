"""Do two models learn the same PARTITIONS, and in the same layers?

`splitting.py` asks whether individual entries reproduce, by two-sided
containment of firing sets at a threshold. That is the right question
for a sparse autoencoder latent, whose firing set is a half-space cut
by a threshold on one direction. It is a harsh question for a one-hot
code: an entry selects one cell of a k-way partition, and the cell's boundaries
are set jointly by all k entries of its head plus the classifier, so
two models can learn the same partition and still fail cell-wise
containment.

The natural unit here is the head. This scores every head of A against
every head of B by the normalized mutual information between their
partitions of the same rows, which is invariant to how either model
happens to index its entries, then matches heads one-to-one by
maximizing total NMI (Hungarian). Reported against two nulls: rows of B
shuffled, which breaks the pairing while keeping both marginals, and
the same matching restricted within each layer.

The layer question comes free from the assignment: if a head in A's
layer i preferentially matches a head in B's layer i, the depth
structure reproduces even where individual cells do not.

  uv run python experiments/ste-arm/partition.py \\
      --a data/out/sonar/multilingual/ste_h76 \\
      --b data/out/sonar/multilingual/ste_h76_s43
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
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--rows", type=int, default=32768)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def codes(ckpt: str, step: int, T: float, X: Float[np.ndarray, "n d_in"],
          batch: int) -> Tuple[Int[np.ndarray, "heads n"],
                               Int[np.ndarray, "heads"], int, int]:
    """Per-head argmax over the rows, as (n_heads, rows) with the layer
    index of each head, that model's k, and the step actually loaded."""
    model, raw, used = load_onto(ckpt, step)
    params = {"params": raw}

    def probe(module, Xb: Float[Array, "b d_in"]) -> Int[Array, "l b h"]:
        E, _ = module.encode(Xb, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        A = []
        for i, de in enumerate(module.dictencs):
            U, G = de.gainshape_in(Ein)
            P = de.dict.cluster(de.classifier(U), T)
            A.append(jnp.argmax(P, -1))
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
            if i < module.l - 1:
                Ein = module.nextinput(Xb, R, P.reshape(Xb.shape[0], -1))
        return jnp.stack(A)

    f = jax.jit(lambda p, x: model.apply(p, x, method=probe))
    out = [np.asarray(f(params, jnp.asarray(X[i:i + batch])))
           for i in range(0, len(X), batch)]
    A = np.concatenate(out, axis=1)                       # (l, rows, h)
    l, n, h = A.shape
    layer = np.repeat(np.arange(l), h)
    return A.transpose(0, 2, 1).reshape(l * h, n), layer, model.k, used


def _tiny(x: Float[np.ndarray, "..."]) -> float:
    """Smallest positive normal of x's dtype, as a log floor. A literal
    like 1e-300 silently underflows to zero in float32 and clamps
    nothing."""
    return float(np.finfo(x.dtype).tiny)


def nmi_matrix(A: Int[np.ndarray, "na n"], ka: int,
               B: Int[np.ndarray, "nb n"], kb: int, chunk: int = 32
               ) -> Float[np.ndarray, "na nb"]:
    """NMI between every head of A and every head of B."""
    na, n = A.shape
    nb = B.shape[0]
    Bh = np.zeros((nb * kb, n), np.float32)
    Bh[B.reshape(-1) + np.repeat(np.arange(nb), n) * kb,
       np.tile(np.arange(n), nb)] = 1.0
    Bh = Bh.reshape(nb, kb, n)
    pb = Bh.mean(-1)                                       # (nb, kb)
    # clamp inside the log rather than masking it: `np.log(p, where=...)`
    # without an `out=` leaves the masked entries uninitialized, and
    # relying on the surrounding `np.where` to discard them is one
    # refactor away from reading garbage. The floor has to come from the
    # dtype -- these are float32, where a literal 1e-300 underflows to
    # zero and clamps nothing.
    hb = -np.where(pb > 0, pb * np.log(np.maximum(pb, _tiny(pb))),
                   0.0).sum(-1)

    out = np.zeros((na, nb), np.float32)
    for s in range(0, na, chunk):
        e = min(s + chunk, na)
        Ah = np.zeros(((e - s) * ka, n), np.float32)
        Ah[A[s:e].reshape(-1) + np.repeat(np.arange(e - s), n) * ka,
           np.tile(np.arange(n), e - s)] = 1.0
        Ah = Ah.reshape(e - s, ka, n)
        pa = Ah.mean(-1)
        ha = -np.where(pa > 0, pa * np.log(np.maximum(pa, _tiny(pa))),
                       0.0).sum(-1)
        # joint: (chunk, ka, nb, kb). `optimize` is load-bearing, not
        # tidiness: without it einsum runs its own nested loop instead of
        # reshaping to a matmul and calling BLAS, which at this size is
        # 81.8s per chunk against 0.9s -- 80x, and it dominated the whole
        # analysis. float32 is exact here regardless, since the entries
        # are 0/1 and every partial sum stays under 2**24.
        J = np.einsum("ain,bjn->aibj", Ah, Bh, optimize=True) / n
        outer = pa[:, :, None, None] * pb[None, None, :, :]
        t = np.where(J > 0, J * np.log(np.maximum(J, _tiny(J))
                                       / np.maximum(outer, _tiny(outer))),
                     0.0)
        mi = t.sum((1, 3))                                 # (chunk, nb)
        denom = np.maximum(0.5 * (ha[:, None] + hb[None, :]), 1e-12)
        out[s:e] = mi / denom
    return out


def report(name: str, M: Float[np.ndarray, "na nb"],
           la: Int[np.ndarray, "na"], lb: Int[np.ndarray, "nb"]
           ) -> Tuple[Float[np.ndarray, "m"], Int[np.ndarray, "m"],
                      Int[np.ndarray, "m"]]:
    """Prints the assignment's summary; returns (matched NMI, rows, cols)."""
    r, c = linear_sum_assignment(-M)
    v = M[r, c]
    same = (la[r] == lb[c]).mean()
    print(f"  {name:<22} mean matched NMI {v.mean():.4f}  "
          f"median {np.median(v):.4f}  max {v.max():.4f}  "
          f"same-layer {same:.1%}")
    return v, r, c


def main() -> None:
    cfg = parse_args()
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float32)
    A, la, ka, sa = codes(cfg.a, cfg.step_a, cfg.temperature, X, cfg.batch)
    B, lb, kb, sb = codes(cfg.b, cfg.step_b, cfg.temperature, X, cfg.batch)
    print(f"A {Path(cfg.a).name} step {sa}: {A.shape[0]} heads, k={ka}")
    print(f"B {Path(cfg.b).name} step {sb}: {B.shape[0]} heads, k={kb}")
    print(f"{cfg.rows} rows\n")

    M = nmi_matrix(A, ka, B, kb)
    v, r, c = report("matched", M, la, lb)

    rng = np.random.default_rng(cfg.seed)
    Bs = B[:, rng.permutation(B.shape[1])]
    Mn = nmi_matrix(A, ka, Bs, kb)
    vn, _, _ = report("null (rows shuffled)", Mn, la, lb)

    print(f"\n  excess over null: {v.mean() - vn.mean():+.4f} "
          f"({(v.mean() - vn.mean()) / max(vn.mean(), 1e-9):+.0%})")
    nl = int(la.max()) + 1
    if nl > 1:
        print(f"\n  matched NMI by A's layer")
        for i in range(nl):
            m = la[r] == i
            if m.any():
                print(f"    layer {i}: NMI {v[m].mean():.4f}  "
                      f"lands in layer {np.bincount(lb[c][m], minlength=nl)}")


if __name__ == "__main__":
    main()

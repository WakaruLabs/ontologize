"""Linear slow-feature analysis on a sentence cache: the slow heads'
closed-form baseline.

Linear SFA finds the directions along which two sentences of one document
stay correlated. With C the covariance of a sentence and S the symmetrized
cross-covariance of a pair's two sentences, they are the generalized
eigenvectors S v = rho C v, scaled so that v'Cv = 1. rho is the correlation
the pair keeps along v, and E[(v'x - v'x')^2] = 2 (1 - rho). This is CCA
with one map shared by both sentences, and what a whitened linear JEPA on
the same pairs learns (Balestriero & LeCun 2026, Proposition 1): no
k-dimensional linear code with whitened output keeps more summed
correlation across the pair than the top k directions.

Pairs (`--pairs`; `--ckpt` compares against the first listed), from the
cache's first `--rows` rows, as `train_pair.py` draws them:
- `doc`: each row with a partner from its own document at segment distance
  at least `--min-d`, uniform over those. The moments are the expectation
  over the draw, not one draw of it.
- `adjacent`: each row and the next segment of its document.
The input is each sentence's direction x / |x|, which gain-shape
classifies (`--input dir`), or the embedding itself (`raw`).

Each pair design gets two fits from the one pass:
- `plain`: SFA on the inputs;
- `langc`: SFA after subtracting each language's mean (the mC4 config of
  `<cache>.langs.npy`). Every sentence of a document shares its language,
  so this removes what language contributes linearly to both moments.

On the held-out tail (the last `--holdout` rows, as `transitions.py`), for
every direction:
- `rho_held`: the correlation the pair design keeps there;
- `corr_<type>`: the correlation on `transitions.py`'s pair types, with its
  partners (`d1` ... `d12`, `d1_sent`, `lang`, `random`);
- `eta2_lang`: the share of its variance between languages. `lang` pairs
  keep the same share in expectation.

`--ckpt` compares checkpoints' heads with the top `--top` directions of the
first design, on the last `--probe` rows (the tail and the rows before it).
Heads come in blocks of `--slow-heads` per layer: `slow`, the first, which
are `train_pair.py`'s slow heads, and `next`, the next as many.
- `r2`: the share of each direction's variance on the tail that a block's
  one-hot code explains, by ridge fitted on the rows before the tail. The
  language label is scored the same way, as block `lang`.
- each head's `eta2`: the share of each direction's variance between its
  entries, over all the probe rows.
- `code_rho`: SFA on the block's one-hot code, `plain` and `langc`, fitted
  on the rows before the tail and held out on it. Its first value is the
  most correlation any linear function of the code keeps across the pair.

    JAX_PLATFORMS=cpu uv run python experiments/next-sentence/sfa.py \\
        --ckpt data/out/sonar/next-sentence/long/doc2_fano_sp1e-3_ph38 \\
               data/out/sonar/next-sentence/long/doc2_sp0_ph38 \\
        --out data/out/sonar/next-sentence/sfa
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from jaxtyping import Float, Int  # noqa: E402

from ontologize.data.loaders import doc_holdout  # noqa: E402
from pairs import adjacent  # noqa: E402

CACHE = ROOT / "data" / "sonar_embeddings" / "mc4_sent16.npy"
OUT = ROOT / "data" / "out" / "sonar" / "next-sentence" / "sfa"
RANKS = (1, 2, 4, 8, 16, 38, 64, 128, 256, 512)
TOPS = (1, 8, 38, 64)


def features(x: Float[np.ndarray, "n d"], inp: str) -> Float[np.ndarray, "n d"]:
    """What SFA sees, in float64: the direction x / |x| (`dir`) or x (`raw`)."""
    x = np.asarray(x, np.float64)
    if inp == "raw":
        return x
    if inp != "dir":
        raise ValueError(f"input {inp!r}: 'dir' or 'raw'")
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def chunkedges(doc: Int[np.ndarray, "n"], chunk: int) -> List[int]:
    """Edges 0 < ... < n about `chunk` rows apart, each at a document's first
    row, so that no document spans two chunks."""
    n = len(doc)
    starts = np.flatnonzero(np.r_[True, doc[1:] != doc[:-1]])
    at = np.searchsorted(starts, np.arange(chunk, n, chunk))
    cut = np.where(at < len(starts), starts[np.minimum(at, len(starts) - 1)], n)
    return sorted({0, n, *cut.tolist()})


def pairmatrix(index: Int[np.ndarray, "n 3"], pairs: str, min_d: int = 2
               ) -> sp.csr_matrix:
    """The pair distribution over the rows of `index` as an (n, n) matrix P
    whose row i holds the probabilities of row i's partners: for
    `adjacent`, 1 at the next segment of its document; for `doc`, 1/m at
    each of the m rows of its document at segment distance at least
    `min_d`. Rows without a partner are zero. A document cut by either end
    of `index` keeps only the partners inside it, as `doc_partners` does."""
    n = len(index)
    if pairs == "adjacent":
        i = adjacent(index, 0, n)
        return sp.csr_matrix((np.ones(len(i)), (i, i + 1)), shape=(n, n))
    if pairs != "doc":
        raise ValueError(f"pairs {pairs!r}: 'doc' or 'adjacent'")
    doc = np.asarray(index[:, 0])
    seg = np.asarray(index[:, 2]).astype(np.int64)
    start = np.flatnonzero(np.r_[True, doc[1:] != doc[:-1]])
    length = np.diff(np.r_[start, n])
    run = np.repeat(np.arange(len(start)), length)
    t = np.arange(int(length.max()))
    cand = start[run][:, None] + t                   # every row of the document
    real = t < length[run][:, None]
    ok = real & (np.abs(seg[np.where(real, cand, 0)] - seg[:, None]) >= min_d)
    i, c = np.nonzero(ok)
    return sp.csr_matrix((1.0 / ok.sum(1)[i], (i, cand[i, c])), shape=(n, n))


def groupsum(group: Int[np.ndarray, "n"], g: int, Y: Float[np.ndarray, "n ..."]
             ) -> Float[np.ndarray, "g ..."]:
    """Sums of the rows of `Y` within each of `g` groups."""
    G = sp.csr_matrix((np.ones(len(group)), (group, np.arange(len(group)))),
                      shape=(g, len(group)))
    return np.asarray(G @ Y)


@dataclass
class Moments:
    """Sums over the pairs of pair matrices (`pairmatrix`), per group. With
    a = P1 the anchors' weights and b = P'1 the partners': `w`, `s_a` and
    `s_b` hold, per group, the sum of a and the a- and b-weighted sums of
    the rows; `m2` is sum_i (a_i + b_i) x_i x_i' and `cross` is
    sum_ij P_ij x_i x_j'. A pair never crosses groups, so a group's partners
    weigh what its anchors do."""
    w: Float[np.ndarray, "g"]
    s_a: Float[np.ndarray, "g d"]
    s_b: Float[np.ndarray, "g d"]
    m2: Float[np.ndarray, "d d"]
    cross: Float[np.ndarray, "d d"]

    @classmethod
    def zeros(cls, d: int, g: int) -> "Moments":
        return cls(np.zeros(g), np.zeros((g, d)), np.zeros((g, d)),
                   np.zeros((d, d)), np.zeros((d, d)))

    def add(self, X: Float[np.ndarray, "n d"], P: sp.csr_matrix,
            group: Int[np.ndarray, "n"]) -> None:
        nz = P.tocoo()
        if (group[nz.row] != group[nz.col]).any():
            raise ValueError("a pair crosses groups")
        a = np.asarray(P.sum(1)).ravel()
        b = np.asarray(P.sum(0)).ravel()
        g = len(self.w)
        self.w += groupsum(group, g, a)
        self.s_a += groupsum(group, g, a[:, None] * X)
        self.s_b += groupsum(group, g, b[:, None] * X)
        self.m2 += X.T @ ((a + b)[:, None] * X)
        self.cross += X.T @ np.asarray(P @ X)


def covariances(mom: Moments, bygroup: bool
                ) -> Tuple[Float[np.ndarray, "d d"], Float[np.ndarray, "d d"],
                           Float[np.ndarray, "g_or_1 d"]]:
    """C, the covariance of one sentence (anchors and partners pooled), and
    S, the symmetrized covariance of a pair's two sentences, about the
    pooled mean, or `bygroup` about each group's pooled mean. Also the
    mean(s) subtracted, (1, d) or (g, d); a group without pairs gets the
    pooled mean."""
    w, s_a, s_b = mom.w, mom.s_a, mom.s_b
    if not bygroup:
        w, s_a, s_b = (w.sum(keepdims=True), s_a.sum(0, keepdims=True),
                       s_b.sum(0, keepdims=True))
    keep = w > 0
    pi = w[keep] / w.sum()                          # each group's share of pairs
    m_a = s_a[keep] / w[keep, None]                 # its anchors' mean
    m_b = s_b[keep] / w[keep, None]                 # its partners' mean
    mu = (m_a + m_b) / 2                            # its pooled mean
    piT = (pi[:, None] * mu).T
    C = mom.m2 / (2 * w.sum()) - piT @ mu
    X = (mom.cross / w.sum() - (pi[:, None] * m_a).T @ mu - piT @ m_b
         + piT @ mu)
    means = np.tile(pi @ mu, (len(w), 1))
    means[keep] = mu
    return C, (X + X.T) / 2, means


def sfa(C: Float[np.ndarray, "d d"], S: Float[np.ndarray, "d d"],
        floor: float = 1e-10
        ) -> Tuple[Float[np.ndarray, "r"], Float[np.ndarray, "d r"]]:
    """The generalized eigenvectors S v = rho C v, by whitening C: rho in
    descending order, and V with V'CV = I and V'SV = diag(rho). Directions
    of C whose variance is below `floor` times the largest are dropped, so
    r <= d."""
    lam, U = np.linalg.eigh(C)
    keep = lam > floor * lam[-1]
    Wh = U[:, keep] / np.sqrt(lam[keep])
    K = Wh.T @ S @ Wh
    rho, Q = np.linalg.eigh((K + K.T) / 2)
    return rho[::-1], Wh @ Q[:, ::-1]


@dataclass
class Fit:
    """One SFA fit: the correlations its directions keep on the rows it was
    fitted on, the directions, and the mean subtracted before projecting,
    one for all rows or one per group."""
    rho: Float[np.ndarray, "r"]
    V: Float[np.ndarray, "d r"]
    means: Float[np.ndarray, "g_or_1 d"]

    def project(self, X: Float[np.ndarray, "n d"], group: Int[np.ndarray, "n"],
                top: int = 0) -> Float[np.ndarray, "n r"]:
        m = self.means[0] if len(self.means) == 1 else self.means[group]
        return (X - m) @ (self.V[:, :top] if top else self.V)


def pairrho(Z: Float[np.ndarray, "n r"], P: sp.csr_matrix,
            group: Int[np.ndarray, "n"]) -> Float[np.ndarray, "r"]:
    """The correlation each column of `Z` keeps across the pair distribution
    P, about the pooled mean."""
    mom = Moments.zeros(Z.shape[1], int(group.max()) + 1)
    mom.add(Z, P, group)
    C, S, _ = covariances(mom, False)
    return np.diag(S) / np.diag(C)


def paircorr(Z: Float[np.ndarray, "n r"],
             pairs: Dict[str, Tuple[Int[np.ndarray, "p"], Int[np.ndarray, "p"]]]
             ) -> Dict[str, Float[np.ndarray, "r"]]:
    """Each column's correlation across the pairs of every type, with the
    columns standardized over all rows of `Z`."""
    Zs = (Z - Z.mean(0)) / Z.std(0)
    return {t: (Zs[i] * Zs[j]).mean(0) for t, (i, j) in pairs.items()}


def eta2(Z: Float[np.ndarray, "n r"], labels: Int[np.ndarray, "n"]
         ) -> Float[np.ndarray, "r"]:
    """Each column's share of variance between the groups of `labels`: the
    R^2 of its group means."""
    Zc = Z - Z.mean(0)
    _, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    sums = groupsum(inv, len(cnt), Zc)
    return (sums ** 2 / cnt[:, None]).sum(0) / (Zc ** 2).sum(0)


def onehot(A: Int[np.ndarray, "heads n"], k: int) -> Float[np.ndarray, "n hk"]:
    """The heads' entries as one block of indicators, head-major."""
    heads, n = A.shape
    F = np.zeros((n, heads * k), np.float64)
    F[np.arange(n)[:, None], A.T + np.arange(heads) * k] = 1.0
    return F


def probe(F_fit: Float[np.ndarray, "m p"], Z_fit: Float[np.ndarray, "m r"],
          F_test: Float[np.ndarray, "t p"], Z_test: Float[np.ndarray, "t r"],
          alpha: float = 1e-4) -> Float[np.ndarray, "r"]:
    """R^2 of each column of `Z_test` from `F_test`, by ridge fitted on the
    fit rows, with `alpha` scaled by the mean feature variance."""
    m_f, m_z = F_fit.mean(0), Z_fit.mean(0)
    Fc = F_fit - m_f
    G = Fc.T @ Fc / len(Fc)
    lam = alpha * float(np.trace(G)) / len(G)
    B = np.linalg.solve(G + lam * np.eye(len(G)), Fc.T @ (Z_fit - m_z) / len(Fc))
    err = Z_test - ((F_test - m_f) @ B + m_z)
    return 1.0 - (err ** 2).sum(0) / ((Z_test - Z_test.mean(0)) ** 2).sum(0)


def fitpass(arr: Float[np.ndarray, "n d"], index: Int[np.ndarray, "n 3"],
            gid: Int[np.ndarray, "hi"], g: int, hi: int, designs: List[str],
            min_d: int, inp: str, chunk: int
            ) -> Tuple[Float[np.ndarray, "d"], Dict[str, Moments]]:
    """One pass over the cache's first `hi` rows, in chunks of whole
    documents, accumulating every pair design's moments about a fixed shift
    (the first chunk's mean, so that no sum loses its digits to the
    mean)."""
    edges = chunkedges(np.asarray(index[:hi, 0]), chunk)
    shift = features(arr[:edges[1]], inp).mean(0)
    moms = {q: Moments.zeros(arr.shape[1], g) for q in designs}
    t0 = time.time()
    for a, b in zip(edges[:-1], edges[1:]):
        X = features(arr[a:b], inp) - shift
        idx = np.asarray(index[a:b])
        for q in designs:
            moms[q].add(X, pairmatrix(idx, q, min_d), gid[a:b])
        print(f"  {b}/{hi} rows ({time.time() - t0:.0f} s)", flush=True)
    return shift, moms


def doc_start(doc: Int[np.ndarray, "n"], i: int) -> int:
    """The first row of the document holding row `i`."""
    other = np.flatnonzero(doc[:i + 1] != doc[i])
    return int(other[-1] + 1) if len(other) else 0


def compare(cfg: argparse.Namespace, arr: Float[np.ndarray, "n d"],
            index: Int[np.ndarray, "n 3"], langs: np.ndarray,
            names: np.ndarray, fits: Dict[str, Fit], q: str, n: int
            ) -> Tuple[Dict, List[Dict], List[Dict]]:
    """The heads of every `--ckpt` against the top directions of `fits`
    (`plain` and `langc`), with the languages `names` as groups."""
    from partition import codes
    lo = n - cfg.probe
    raw = np.asarray(arr[lo:], np.float32)
    X = features(raw, cfg.input)
    idx = np.asarray(index[lo:])
    gid = np.searchsorted(names, np.asarray(langs[lo:]))
    tail = len(X) - cfg.holdout
    cut = doc_start(idx[:, 0], tail)        # no document in both
    P_fit = pairmatrix(idx[:cut], q, cfg.min_d)
    P_tail = pairmatrix(idx[tail:], q, cfg.min_d)
    Z = {kind: f.project(X, gid, cfg.top) for kind, f in fits.items()}
    print(f"\nheads against the top {cfg.top} `{q}` directions: probe rows "
          f"{lo}..{lo + cut} fit, {n - cfg.holdout}..{n} held out")

    def block(F: Float[np.ndarray, "n p"], code: bool) -> Dict:
        out = {"r2": {}, "r2_mean": {}}
        for kind, Zk in Z.items():
            r2 = probe(F[:cut], Zk[:cut], F[tail:], Zk[tail:], cfg.alpha)
            out["r2"][kind] = r2.tolist()
            out["r2_mean"][kind] = {t: float(r2[:t].mean()) for t in TOPS
                                    if t <= cfg.top}
        if code:
            mom = Moments.zeros(F.shape[1], len(names))
            mom.add(F[:cut], P_fit, gid[:cut])
            out["code_rho_fit"], out["code_rho_held"] = {}, {}
            for kind in Z:
                C, S, means = covariances(mom, kind == "langc")
                rho, V = sfa(C, S, cfg.floor)
                held = pairrho(Fit(rho, V, means).project(
                    F[tail:], gid[tail:], cfg.top), P_tail, gid[tail:])
                out["code_rho_fit"][kind] = rho[:cfg.top].tolist()
                out["code_rho_held"][kind] = held.tolist()
        return out

    res = {"probe_rows": cfg.probe, "fit_rows": int(cut),
           "held_rows": int(cfg.holdout), "pairs": q,
           "blocks": {"lang": block(onehot(gid[None], len(names)), False)}}
    heads_csv, r2_csv = [], []
    for ckpt in cfg.ckpt:
        t0 = time.time()
        A, layer, k, step = codes(ckpt, cfg.step, cfg.temperature, raw,
                                  cfg.batch)
        h = len(layer) // (int(layer.max()) + 1)
        within = np.arange(len(layer)) % h
        s = cfg.slow_heads
        name = Path(ckpt).name
        print(f"  {name}: step {step}, codes in {time.time() - t0:.0f} s",
              flush=True)
        for bname, m in (("slow", within < s),
                         ("next", (within >= s) & (within < 2 * s))):
            r = block(onehot(A[m], k), True)
            r.update(ckpt=ckpt, step=int(step), heads=int(m.sum()))
            res["blocks"][f"{name}/{bname}"] = r
            for kind, Zk in Z.items():
                E = np.stack([eta2(Zk, a) for a in A[m]])      # (heads, top)
                for j, hd in enumerate(np.flatnonzero(m)):
                    heads_csv.append({
                        "ckpt": name, "block": bname, "kind": kind,
                        "layer": int(layer[hd]), "head": int(hd % h),
                        "best_rank": int(E[j].argmax()) + 1,
                        "best_eta2": float(E[j].max()),
                        "eta2_sum_top38": float(E[j, :38].sum()),
                        "eta2_sum_top": float(E[j].sum())})
                r["eta2_heads"] = r.get("eta2_heads", {})
                r["eta2_heads"][kind] = {
                    "best_mean": float(E.max(1).mean()),
                    "distinct_best": int(len(np.unique(E.argmax(1)))),
                    "sum_top38_mean": float(E[:, :38].sum(1).mean())}
    for bname, r in res["blocks"].items():
        for kind, v in r["r2"].items():
            r2_csv.extend({"block": bname, "kind": kind, "rank": j + 1,
                           "r2": x} for j, x in enumerate(v))
    return res, heads_csv, r2_csv


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--rows", type=int, default=4_000_000,
                   help="fit on pairs within the cache's first ROWS rows, "
                        "the pair runs' training prefix")
    p.add_argument("--holdout", type=int, default=32768,
                   help="the held-out tail rows, as transitions.py")
    p.add_argument("--pairs", nargs="+", default=["doc", "adjacent"],
                   choices=["doc", "adjacent"])
    p.add_argument("--min-d", type=int, default=2,
                   help="`doc` pairs: the least segment distance")
    p.add_argument("--input", default="dir", choices=["dir", "raw"])
    p.add_argument("--chunk", type=int, default=1 << 18)
    p.add_argument("--floor", type=float, default=1e-10,
                   help="drop directions of C below FLOOR x its largest "
                        "variance")
    p.add_argument("--seed", type=int, default=0,
                   help="transitions.py's draw of lang and random partners")
    p.add_argument("--ckpt", nargs="*", default=[])
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--slow-heads", type=int, default=38)
    p.add_argument("--top", type=int, default=64)
    p.add_argument("--probe", type=int, default=262144,
                   help="--ckpt: the cache's last PROBE rows, the tail "
                        "included")
    p.add_argument("--alpha", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--out", default=str(OUT))
    cfg = p.parse_args()

    # partition.py, which transitions imports, sets the XLA environment
    import transitions as tr

    cache = Path(cfg.cache)
    stem = cache.with_name(cache.stem)
    arr = np.load(cache, mmap_mode="r")
    n, d = arr.shape
    index = np.load(f"{stem}.index.npy", mmap_mode="r")
    langs = np.load(f"{stem}.langs.npy", mmap_mode="r")
    hi = min(cfg.rows, n - doc_holdout(cache, cfg.holdout))
    names, gid = np.unique(np.asarray(langs[:hi]), return_inverse=True)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"fit: pairs {cfg.pairs} in rows 0..{hi} of {cache.name}, "
          f"input {cfg.input}, {len(names)} languages")
    shift, moms = fitpass(arr, index, gid, len(names), hi, cfg.pairs,
                          cfg.min_d, cfg.input, cfg.chunk)
    fits: Dict[Tuple[str, str], Fit] = {}
    for q in cfg.pairs:
        for kind in ("plain", "langc"):
            C, S, means = covariances(moms[q], kind == "langc")
            rho, V = sfa(C, S, cfg.floor)
            fits[q, kind] = Fit(rho, V, shift + means)

    lo = n - cfg.holdout
    Xt = features(arr[lo:], cfg.input)
    idx = np.asarray(index[lo:])
    kind_t = np.asarray(np.load(f"{stem}.kind.npy", mmap_mode="r")[lo:])
    names_t = np.asarray(langs[lo:])
    gid_t = np.searchsorted(names, names_t)
    assert (names[np.minimum(gid_t, len(names) - 1)] == names_t).all(), \
        "a held-out language never occurs in the training prefix"
    pairs = tr.pair_sets(idx, kind_t, idx[:, 0].astype(np.int64), gid_t,
                         cfg.seed)
    print(f"\nheld out: rows {lo}..{n}; pairs: "
          + ", ".join(f"{t} {len(i)}" for t, (i, _) in pairs.items()))
    summary = {"cache": str(cache), "fit_rows": int(hi), "input": cfg.input,
               "min_d": cfg.min_d, "held_rows": cfg.holdout,
               "pair_counts": {t: int(len(i)) for t, (i, _) in pairs.items()},
               "fits": {}}
    rows_csv = []
    for (q, kind), f in fits.items():
        Z = f.project(Xt, gid_t)
        held = pairrho(Z, pairmatrix(idx, q, cfg.min_d), gid_t)
        corr = paircorr(Z, pairs)
        e2 = eta2(Z, gid_t)
        r = len(f.rho)
        s = {"directions": r, "dropped": d - r,
             "pair_weight": float(moms[q].w.sum()),
             "rho_fit": {k: float(f.rho[k - 1]) for k in RANKS if k <= r},
             "rho_held": {k: float(held[k - 1]) for k in RANKS if k <= r},
             "held_above": {c: int((held > c).sum())
                            for c in (0.1, 0.2, 0.3, 0.5, 0.7)},
             "top": {k: {"rho_fit": float(f.rho[:k].mean()),
                         "rho_held": float(held[:k].mean()),
                         "eta2_lang": float(e2[:k].mean()),
                         **{f"corr_{t}": float(v[:k].mean())
                            for t, v in corr.items()}}
                     for k in TOPS}}
        summary["fits"][f"{q}/{kind}"] = s
        for j in range(r):
            rows_csv.append({"pairs": q, "kind": kind, "rank": j + 1,
                             "rho_fit": float(f.rho[j]),
                             "rho_held": float(held[j]),
                             **{f"corr_{t}": float(v[j])
                                for t, v in corr.items()},
                             "eta2_lang": float(e2[j])})
        print(f"\n{q} / {kind}: {r} directions ({d - r} dropped)")
        print("  rank          " + "".join(f"{k:>7}" for k in RANKS if k <= r))
        for lab, v in (("rho fit", f.rho), ("rho held", held),
                       ("corr d1", corr["d1"]), ("corr d12", corr["d12"]),
                       ("corr lang", corr["lang"]), ("eta2 lang", e2)):
            print(f"  {lab:<14}" + "".join(f"{v[k - 1]:7.3f}" for k in RANKS
                                          if k <= r))
        print("  held-out rho above 0.1/0.2/0.3/0.5/0.7: "
              + " / ".join(str(x) for x in s["held_above"].values()))
        for k in (8, 38):
            tk = s["top"][k]
            print(f"  top {k} means: rho held {tk['rho_held']:.3f}; corr "
                  + " ".join(f"{t} {tk[f'corr_{t}']:.3f}" for t in corr)
                  + f"; eta2 lang {tk['eta2_lang']:.3f}")

    np.savez(out / f"sfa_{cfg.input}.npz", names=names, shift=shift,
             **{f"{q}_{kind}_{a}": getattr(f, a)
                for (q, kind), f in fits.items()
                for a in ("rho", "V", "means")})
    with open(out / f"directions_{cfg.input}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        wr.writeheader()
        wr.writerows(rows_csv)

    if cfg.ckpt:
        q = cfg.pairs[0]
        res, heads_csv, r2_csv = compare(
            cfg, arr, index, langs, names,
            {kind: fits[q, kind] for kind in ("plain", "langc")}, q, n)
        summary["compare"] = res
        for bname, r in res["blocks"].items():
            print(f"  {bname}: r2 means (top " + "/".join(
                str(t) for t in r["r2_mean"]["plain"]) + "): "
                + "; ".join(f"{kind} " + " ".join(
                    f"{x:.3f}" for x in v.values())
                    for kind, v in r["r2_mean"].items()))
            for kind in r.get("code_rho_held", {}):
                fit, held = r["code_rho_fit"][kind], r["code_rho_held"][kind]
                print(f"    code SFA ({kind}) rho fit / held at ranks 1 2 4 8 "
                      "16 38: " + " ".join(f"{fit[k - 1]:.3f}/{held[k - 1]:.3f}"
                                           for k in (1, 2, 4, 8, 16, 38)
                                           if k <= cfg.top)
                      + f"; held mean over the top 8 / 38: "
                      f"{np.mean(held[:8]):.3f} / {np.mean(held[:38]):.3f}")
            for kind, e in r.get("eta2_heads", {}).items():
                print(f"    heads' eta2 ({kind}): best {e['best_mean']:.3f} "
                      f"on {e['distinct_best']} distinct directions; "
                      f"summed over the top 38 {e['sum_top38_mean']:.3f}")
        for name, rows in (("heads", heads_csv), ("r2", r2_csv)):
            with open(out / f"{name}_{cfg.input}.csv", "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
    (out / f"summary_{cfg.input}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

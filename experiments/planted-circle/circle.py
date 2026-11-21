"""Do heads cut a planted circle at the same places across seeds?

A circle has no preferred place to cut it: rotating a partition of the
angle into arcs changes nothing about how well the arcs reconstruct the
data, so the cuts are fixed only up to rotation (Karkada et al.'s periodic
concepts), and two seeds should cut the same circle at different places
while carrying the same information about the angle. A segment has ends,
which pin its cuts. This plants both in one data set:

    x = R (cos θ u₁ + sin θ u₂) + t u₃ + sigma * eps

with u₁, u₂, u₃ orthonormal in R^d, θ uniform on the circle (or, with
`--tilt a`, of density proportional to 1 + a cos θ, which breaks the
symmetry and should pin the cuts) and t uniform on [-√3 S, √3 S]. The
data are the same for every seed; only the model seed changes, as in a
seed pair on a cache. The model is `experiments/planted-hierarchy`'s
(dictenc-toy's schedule), by default one layer of two heads.

Scored on held-out rows, per seed: which head carries θ and which t (the
share of each one's entropy, binned, that the head's entry carries), and
where each head cuts its factor: the bin edges at which the modal entry
changes. Between every pair of seeds: the distance from each cut of one to
the nearest cut of the other, over half the mean gap between cuts (0 for
the same cuts, about 0.5 for cuts placed independently), and the
partition NMI of the two heads.

    uv run python experiments/planted-circle/circle.py --seeds 8
    uv run python experiments/planted-circle/circle.py --seeds 8 --tilt 0.5
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from jaxtyping import Float, Int  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "planted-hierarchy"))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("planted circle and segment")
    g.add_argument("--d", type=int, default=16)
    g.add_argument("--radius", type=float, default=1.0)
    g.add_argument("--segment-sd", type=float, default=1.0)
    g.add_argument("--tilt", type=float, default=0.0,
                   help="density of θ proportional to 1 + tilt cos θ")
    g.add_argument("--sigma", type=float, default=0.05)
    g.add_argument("--n-train", type=int, default=32768)
    g.add_argument("--n-eval", type=int, default=16384)
    g.add_argument("--data-seed", type=int, default=0)
    g.add_argument("--bins", type=int, default=64)
    g.add_argument("--clean-share", type=float, default=0.35,
                   help="a seed is clean when one head carries at least this "
                        "share of each factor (at most 0.6 at k=8)")
    g.add_argument("--cross-share", type=float, default=0.1,
                   help="... and no other head more than this of it")
    m = p.add_argument_group("model (planted-hierarchy's builder)")
    m.add_argument("--l", type=int, default=1)
    m.add_argument("--h", type=int, default=2)
    m.add_argument("--k", type=int, default=8)
    m.add_argument("--e-dec", type=int, default=32)
    m.add_argument("--select", default="ste")
    m.add_argument("--no-deepsup", action="store_true")
    t = p.add_argument_group("training (dictenc-toy's schedule)")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--anneal-steps", type=int, default=3000)
    t.add_argument("--b", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument("--temperature-end", type=float, default=0.03)
    t.add_argument("--p-drop", type=float, default=0.1)
    t.add_argument("--sd-k", type=float, default=0.02)
    t.add_argument("--sd-f", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, default=8,
                   help="model seeds; the data stay the same")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default=None,
                   help="default experiments/planted-circle/out/<name>")
    cfg = p.parse_args(argv)
    if cfg.name is None:
        cfg.name = (f"l{cfg.l}h{cfg.h}k{cfg.k}_tilt{cfg.tilt:g}"
                    f"_x{cfg.seeds}")
    return cfg


# ---------- planted data ----------

def angles(rng: np.random.Generator, n: int, tilt: float
           ) -> Float[np.ndarray, "n"]:
    """θ of density (1 + tilt cos θ) / 2π on [0, 2π), by rejection."""
    assert 0 <= tilt < 1
    out = np.empty(0)
    while len(out) < n:
        th = rng.uniform(0, 2 * np.pi, 2 * n)
        keep = rng.uniform(0, 1 + tilt, 2 * n) < 1 + tilt * np.cos(th)
        out = np.concatenate([out, th[keep]])
    return out[:n]


def plant(cfg: argparse.Namespace, rng: np.random.Generator
          ) -> Float[np.ndarray, "3 d"]:
    """Three orthonormal directions: the circle's plane and the segment."""
    Q, _ = np.linalg.qr(rng.normal(size=(cfg.d, 3)))
    return Q.T


def sample(cfg: argparse.Namespace, U: Float[np.ndarray, "3 d"],
           rng: np.random.Generator, n: int
           ) -> tuple[Float[np.ndarray, "n d"], Float[np.ndarray, "n"],
                      Float[np.ndarray, "n"]]:
    """(X, θ, t)."""
    th = angles(rng, n, cfg.tilt)
    half = np.sqrt(3) * cfg.segment_sd
    t = rng.uniform(-half, half, n)
    X = (cfg.radius * (np.cos(th)[:, None] * U[0] + np.sin(th)[:, None] * U[1])
         + t[:, None] * U[2] + cfg.sigma * rng.normal(size=(n, cfg.d)))
    return X.astype(np.float32), th, t


# ---------- scoring ----------

def explained(a: Int[np.ndarray, "n"], z: Int[np.ndarray, "n"]) -> float:
    """I(a; z) / H(z)."""
    from tree import explained as ex
    return ex(a, z)


def nmi(a: Int[np.ndarray, "n"], b: Int[np.ndarray, "n"]) -> float:
    """I(a; b) over the mean of the two entropies."""
    ka, kb = int(a.max()) + 1, int(b.max()) + 1
    J = np.zeros((ka, kb))
    np.add.at(J, (a, b), 1.0)
    J /= J.sum()
    pa, pb = J.sum(1), J.sum(0)
    nz = J > 0
    mi = (J[nz] * np.log(J[nz] / np.outer(pa, pb)[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(mi / max(0.5 * (ha + hb), 1e-12))


def cuts(a: Int[np.ndarray, "n"], z: Float[np.ndarray, "n"], lo: float,
         hi: float, bins: int, periodic: bool, min_run: int = 2
         ) -> Float[np.ndarray, "c"]:
    """Where a head's modal entry changes along a factor: the edges between
    adjacent bins of z with different modal entries, wrapping around if
    `periodic`. An empty bin takes its predecessor's mode, and a run of
    fewer than `min_run` bins takes its predecessor's, so a mode that
    flickers at a boundary counts as one cut."""
    b = np.clip(((z - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
    k = int(a.max()) + 1
    C = np.zeros((bins, k))
    np.add.at(C, (b, a), 1.0)
    mode = C.argmax(1)
    for i in range(1, bins):
        if C[i].sum() == 0:
            mode[i] = mode[i - 1]
    i = 0
    while i < bins:
        j = i
        while j < bins and mode[j] == mode[i]:
            j += 1
        if j - i < min_run and i > 0:
            mode[i:j] = mode[i - 1]
        i = j
    edges = lo + (hi - lo) * np.arange(1, bins) / bins
    out = list(edges[mode[1:] != mode[:-1]])
    if periodic and mode[0] != mode[-1]:
        out.append(lo)
    return np.sort(np.array(out))


def cut_distance(c1: Float[np.ndarray, "c1"], c2: Float[np.ndarray, "c2"],
                 period: float | None, span: float) -> float:
    """Mean distance from each cut of c1 to the nearest of c2, over half the
    mean gap between c2's cuts (`span` over their number): 0 for the same
    cuts, about 0.5 for cuts placed independently. `period` makes the
    distance circular."""
    if len(c1) == 0 or len(c2) == 0:
        return float("nan")
    D = np.abs(c1[:, None] - c2[None, :])
    if period is not None:
        D = np.minimum(D, period - D)
    return float(D.min(1).mean() / (0.5 * span / len(c2)))


def rotated_nmi(m: int) -> float:
    """Expected NMI between two partitions of a uniform circle into m equal
    arcs whose offsets are independent: an offset of a fraction f of an arc
    leaves H(A | B) = h(f), whose mean over f is 1/2 nat, so the NMI is
    1 - 1 / (2 ln m). Cuts that wander freely still read this high."""
    return 1.0 - 1.0 / (2.0 * np.log(m))


def score(A: Int[np.ndarray, "heads n"], th: Float[np.ndarray, "n"],
          t: Float[np.ndarray, "n"], cfg: argparse.Namespace) -> dict:
    """Which head carries each factor, and that head's cuts."""
    half = np.sqrt(3) * cfg.segment_sd
    zb = np.clip((th / (2 * np.pi) * 32).astype(int), 0, 31)
    tb = np.clip(((t + half) / (2 * half) * 32).astype(int), 0, 31)
    sh_th = [explained(a, zb) for a in A]
    sh_t = [explained(a, tb) for a in A]
    hc, hs = int(np.argmax(sh_th)), int(np.argmax(sh_t))
    others = [i for i in range(len(A)) if i not in (hc, hs)]
    clean = (hc != hs and sh_th[hc] >= cfg.clean_share
             and sh_t[hs] >= cfg.clean_share
             and sh_th[hs] <= cfg.cross_share and sh_t[hc] <= cfg.cross_share
             and all(sh_th[i] <= cfg.cross_share and sh_t[i] <= cfg.cross_share
                     for i in others))
    mixed = sum(sh_th[i] > cfg.cross_share and sh_t[i] > cfg.cross_share
                for i in range(len(A)))
    return {"share_theta": sh_th, "share_t": sh_t, "clean": bool(clean),
            "mixed_heads": int(mixed),
            "circle_head": hc, "segment_head": hs,
            "circle_cuts": cuts(A[hc], th, 0, 2 * np.pi, cfg.bins, True).tolist(),
            "segment_cuts": cuts(A[hs], t, -half, half, cfg.bins, False).tolist(),
            "circle_entries": int(len(np.unique(A[hc]))),
            "segment_entries": int(len(np.unique(A[hs])))}


def pairs(runs: list, A_all: list, cfg: argparse.Namespace) -> dict:
    """Between every pair of seeds: cut distances both ways and partition
    NMI, for the circle head and the segment head."""
    half = np.sqrt(3) * cfg.segment_sd
    out = {"circle_cut": [], "segment_cut": [], "circle_nmi": [],
           "segment_nmi": [], "circle_null_nmi": [], "segment_null_nmi": []}
    rng = np.random.default_rng(0)
    for i, j in itertools.combinations(range(len(runs)), 2):
        r, s = runs[i], runs[j]
        for f, key, period, span in (("circle", "circle_cuts", 2 * np.pi, 2 * np.pi),
                                     ("segment", "segment_cuts", None, 2 * half)):
            c1, c2 = np.array(r[key]), np.array(s[key])
            d = 0.5 * (cut_distance(c1, c2, period, span)
                       + cut_distance(c2, c1, period, span))
            out[f"{f}_cut"].append(d)
            a = A_all[i][r[f"{f}_head"]]
            b = A_all[j][s[f"{f}_head"]]
            out[f"{f}_nmi"].append(nmi(a, b))
            out[f"{f}_null_nmi"].append(nmi(a, rng.permutation(b)))
    return out


def main() -> None:
    from tree import build, run_model, train
    cfg = parse_args()
    out = Path(cfg.out) if cfg.out else \
        ROOT / "experiments/planted-circle/out" / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(cfg.data_seed)
    U = plant(cfg, data)
    X_tr, _, _ = sample(cfg, U, data, cfg.n_train)
    X, th, t = sample(cfg, U, data, cfg.n_eval)
    print(f"{cfg.name}: circle (radius {cfg.radius:g}, tilt {cfg.tilt:g}) "
          f"and segment (sd {cfg.segment_sd:g}) in d={cfg.d}, sigma "
          f"{cfg.sigma:g}; model l={cfg.l} h={cfg.h} k={cfg.k}; "
          f"{cfg.seeds} seeds", flush=True)
    t0 = time.time()
    runs, A_all = [], []
    for i in range(cfg.seeds):
        seed = cfg.seed + i
        hyper, model, state = build(cfg, seed)
        state = train(cfg, hyper, state, X_tr, seed)
        Y, A = run_model(model, state.params, X, cfg.temperature_end)
        fvu = float(((Y[-1] - X) ** 2).mean() / ((X - X.mean(0)) ** 2).mean())
        r = {"seed": seed, "fvu": fvu, **score(A, th, t, cfg)}
        runs.append(r)
        A_all.append(A)
        print(f"seed {seed}: FVU {fvu:.4f}; θ share by head "
              f"{np.round(r['share_theta'], 3).tolist()}, t share "
              f"{np.round(r['share_t'], 3).tolist()}; circle head "
              f"{r['circle_head']} ({r['circle_entries']} entries, "
              f"{len(r['circle_cuts'])} cuts), segment head "
              f"{r['segment_head']} ({r['segment_entries']} entries, "
              f"{len(r['segment_cuts'])} cuts)", flush=True)
    keep = [i for i, r in enumerate(runs) if r["clean"]]
    fv = {c: [r["fvu"] for r in runs if r["clean"] == c] for c in (True, False)}
    print(f"clean seeds (one head per factor): {len(keep)}/{len(runs)}; FVU "
          f"clean {np.mean(fv[True]) if fv[True] else float('nan'):.4f}, "
          f"mixed {np.mean(fv[False]) if fv[False] else float('nan'):.4f}")
    sep = [r["fvu"] for r in runs if r["mixed_heads"] == 0]
    mix = [r["fvu"] for r in runs if r["mixed_heads"] > 0]
    print(f"seeds with no head carrying both factors: {len(sep)}/{len(runs)}; "
          f"FVU {np.mean(sep) if sep else float('nan'):.4f} against "
          f"{np.mean(mix) if mix else float('nan'):.4f}; mixed heads per seed "
          f"{np.mean([r['mixed_heads'] for r in runs]):.2f}")
    tallies = {}
    for name, idx in (("clean", keep), ("all", list(range(len(runs))))):
        if len(idx) < 2:
            continue
        pr = pairs([runs[i] for i in idx], [A_all[i] for i in idx], cfg)
        tl = {key: float(np.nanmean(v)) for key, v in pr.items()}
        m = float(np.median([runs[i]["circle_entries"] for i in idx]))
        tl["circle_nmi_rotated"] = rotated_nmi(int(m))
        tl["seeds"], tl["pairs"] = len(idx), len(pr["circle_nmi"])
        tallies[name] = {"tally": tl, "pairs": pr}
        print(f"{name} pairs ({tl['pairs']}): cut distance circle "
              f"{tl['circle_cut']:.3f}, segment {tl['segment_cut']:.3f} (0 same, "
              f"~0.5 independent); NMI circle {tl['circle_nmi']:.3f} (equal arcs "
              f"at independent offsets {tl['circle_nmi_rotated']:.3f}), segment "
              f"{tl['segment_nmi']:.3f} (shuffled {tl['circle_null_nmi']:.3f}, "
              f"{tl['segment_null_nmi']:.3f})")
    print(f"{time.time() - t0:.0f}s")
    (out / "summary.json").write_text(json.dumps(
        {"config": vars(cfg), "seeds": runs, "clean_seeds": keep,
         "fvu_clean": fv[True], "fvu_mixed": fv[False], **tallies}, indent=1))


if __name__ == "__main__":
    main()

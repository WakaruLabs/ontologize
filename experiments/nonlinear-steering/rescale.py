"""How much does GPT-2 small's behaviour depend on the overall scale of its
layer-8 residual, which a gain-shape Ontologizer's code cannot see?

A gain-shape layer classifies the direction of its input and scales its
output by the input's norm. With an unbiased encoder, decoder and
classifier, multiplying the input by c > 0 therefore leaves every code
unchanged, and whatever the network does differently with c·h than with h
is invisible to the code. This measures that difference against steps of
the same length that move the residual's direction instead:

  rescale     h -> c h
  orth        h -> h + |c - 1| |h| u, with u a random direction orthogonal to h
  orth_data   the same, with u taken from another held-out activation
              (centred) and orthogonalized against h

The edits apply at positions >= 1 of held-out `gpt2_l8` windows, either at
every position (`all`) or at every other one (`alt`, which also changes
each position's norm relative to its neighbours'). Behaviour is the
next-token distribution: KL from the unedited model, the change in
cross-entropy on the actual next tokens, and top-1 agreement.

It encodes the same edits with dictionaries trained at this site:
- for an Ontologizer, the share of heads whose winning entry changes;
- for an SAE, the Jaccard distance between active sets and the relative
  change of the code.
It also reports how much the residual's norm varies naturally, so the grid
of c can be read against it.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from jaxtyping import Float

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (OUT, ROOT, capture, device_for, load_gpt2,  # noqa: E402
                    next_token_stats, orth_step, parity, run_temperature,
                    site_edit, tail_windows)

KINDS = ("rescale", "orth", "orth_data")
SCHEMES = ("all", "alt")
RUNS = ROOT / "data" / "out" / "gpt2_l8"
ONTOS = ("ste_h20_cat128", "ste_h76", "ste_h76_sgn", "ste-sigmoid")
SAES = ("m12160_k32", "m12160_k128", "m12160_l10.01", "m12160_l10.01_gated")

Key = Tuple[str, str, float]                  # (scheme or "-", kind, c)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--c", type=float, nargs="+",
                   default=[0.5, 0.67, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0])
    p.add_argument("--models", nargs="*", type=Path,
                   default=[RUNS / r for r in ONTOS],
                   help="Ontologizer run directories (latest step)")
    p.add_argument("--saes", nargs="*", type=Path,
                   default=[RUNS / "sae" / r for r in SAES],
                   help="sae.py run directories")
    p.add_argument("--rows", type=int, default=4096,
                   help="held-out activations encoded per condition")
    p.add_argument("--windows", type=int, default=0,
                   help="use only the first N held-out windows (0 = all)")
    p.add_argument("--batch", type=int, default=16, help="windows per GPT-2 pass")
    p.add_argument("--code-batch", type=int, default=8192)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT / "rescale")
    return p.parse_args()


def edited(h: Float[torch.Tensor, "b t d"], kind: str, c: float,
           g: Float[torch.Tensor, "b t d"], scheme: str
           ) -> Float[torch.Tensor, "b t d"]:
    """`h` (positions >= 1) under one edit. `g` holds the raw directions the
    orthogonal steps use; `alt` edits absolute positions 1, 3, 5, ..., which
    are the even local indices because local index 0 is position 1."""
    if kind == "rescale":
        e = c * h
    else:
        e = h + orth_step(h, abs(c - 1.0), g)
    if scheme == "alt":
        keep = torch.arange(h.shape[1], device=h.device) % 2 == 0
        e = torch.where(keep[None, :, None], e, h)
    return e


def natural_norms(H: Float[np.ndarray, "n t d"]) -> Dict[str, object]:
    """How the site's norm varies over held-out positions: percentiles of
    each norm over the global median and over its own window's median, and
    its rank correlation with position."""
    from scipy.stats import spearmanr
    nrm = np.linalg.norm(H, axis=-1)                       # (n, t)
    q = [1, 5, 25, 50, 75, 95, 99]
    pos = np.broadcast_to(np.arange(1, nrm.shape[1] + 1), nrm.shape)
    return {
        "median": float(np.median(nrm)),
        "percentiles": q,
        "over_global_median": np.percentile(nrm / np.median(nrm), q).round(4).tolist(),
        "over_window_median": np.percentile(
            nrm / np.median(nrm, axis=1, keepdims=True), q).round(4).tolist(),
        "spearman_with_position": float(spearmanr(nrm.ravel(), pos.ravel())[0]),
    }


def behaviour(model, toks: np.ndarray, pool: Float[torch.Tensor, "m d"],
              mu: Float[torch.Tensor, "d"], cs: List[float], batch: int,
              device: torch.device, seed: int
              ) -> Dict[Key, Dict[str, np.ndarray]]:
    """Per-window means of KL, delta cross-entropy and top-1 agreement for
    every (scheme, kind, c). Each batch draws one set of random and
    data-shaped directions and reuses it across c and schemes, so the
    conditions differ only in the edit."""
    gen = torch.Generator(device=device).manual_seed(seed)
    out: Dict[Key, Dict[str, list]] = {}
    for b0 in range(0, len(toks), batch):
        tk = torch.as_tensor(toks[b0:b0 + batch], device=device)
        with torch.no_grad():
            ref = model(tk, use_cache=False).logits
        shape = (tk.shape[0], tk.shape[1] - 1, pool.shape[1])
        g_rand = torch.randn(shape, generator=gen, device=device)
        idx = torch.randint(len(pool), shape[:2], generator=gen, device=device)
        g_data = pool[idx] - mu
        for scheme in SCHEMES:
            for kind in KINDS:
                g = g_data if kind == "orth_data" else g_rand
                for c in cs:
                    fn: Callable = (lambda h, kind=kind, c=c, g=g, s=scheme:
                                    edited(h, kind, c, g, s))
                    with site_edit(model, fn), torch.no_grad():
                        lg = model(tk, use_cache=False).logits
                    kl, ce_r, ce_e, same = next_token_stats(ref, lg, tk)
                    rec = out.setdefault((scheme, kind, c),
                                         {"kl": [], "dce": [], "top1": []})
                    rec["kl"] += kl.mean(1).tolist()
                    rec["dce"] += (ce_e - ce_r).mean(1).tolist()
                    rec["top1"] += same.float().mean(1).tolist()
        print(f"  windows {min(b0 + batch, len(toks))}/{len(toks)}", flush=True)
    return {k: {m: np.asarray(v) for m, v in r.items()} for k, r in out.items()}


def code_inputs(X0: Float[np.ndarray, "n d"], pool: Float[np.ndarray, "m d"],
                cs: List[float], rng: np.random.Generator
                ) -> Dict[Key, Float[np.ndarray, "n d"]]:
    """The same three edits applied to sampled rows, with one random and one
    data-shaped direction per row shared across c, plus `float_floor`: a
    rescale by 1 + 1e-6. A code that is invariant to rescaling in exact
    arithmetic can still flip near-tied decisions under float32 rounding,
    and that row measures how often; a power-of-two c rounds exactly and
    flips none."""
    X = torch.as_tensor(X0)
    g_rand = torch.as_tensor(rng.standard_normal(X0.shape).astype(np.float32))
    mu = pool.mean(0)
    g_data = torch.as_tensor(pool[rng.integers(len(pool), size=len(X0))] - mu)
    conds: Dict[Key, np.ndarray] = {("-", "float_floor", 1.000001): X0 * 1.000001}
    for c in cs:
        conds[("-", "rescale", c)] = (c * X).numpy()
        conds[("-", "orth", c)] = (X + orth_step(X, abs(c - 1.0), g_rand)).numpy()
        conds[("-", "orth_data", c)] = (X + orth_step(X, abs(c - 1.0), g_data)).numpy()
    return conds


def onto_changes(run: Path, X0: np.ndarray, conds: Dict[Key, np.ndarray],
                 batch: int) -> Tuple[Dict[Key, float], int]:
    """Share of (head, row) pairs whose winning entry differs from the
    unedited row's, per condition, and the checkpoint step used."""
    from partition import codes
    keys = list(conds)
    Xall = np.concatenate([X0] + [conds[k] for k in keys]).astype(np.float32)
    A, _, _, used = codes(str(run), None, run_temperature(run), Xall, batch)
    n = len(X0)
    A0 = A[:, :n]
    return ({k: float((A[:, n * (i + 1):n * (i + 2)] != A0).mean())
             for i, k in enumerate(keys)}, used)


def sae_changes(run: Path, X0: np.ndarray, conds: Dict[Key, np.ndarray],
                batch: int) -> Dict[Key, Tuple[float, float]]:
    """Per condition, the mean Jaccard distance between a row's active set
    and the unedited row's, and the mean relative change of its code."""
    import jax.numpy as jnp
    import sae as S
    params = {k: jnp.asarray(v) for k, v in np.load(run / "params.npz").items()}
    meta = json.loads((run / "meta.json").read_text())

    def enc(X: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(S.encode(
            params, jnp.asarray(X[i:i + batch]), meta.get("topk", 0),
            meta.get("groups", 0), meta.get("group_fn", "top1")))
            for i in range(0, len(X), batch)])

    z0 = enc(X0)
    a0 = z0 != 0
    out = {}
    for k, Xc in conds.items():
        z = enc(Xc)
        a = z != 0
        union = np.maximum((a0 | a).sum(1), 1)
        jac = 1.0 - (a0 & a).sum(1) / union
        rel = np.linalg.norm(z - z0, axis=1) / np.maximum(np.linalg.norm(z0, axis=1), 1e-12)
        out[k] = (float(jac.mean()), float(rel.mean()))
    return out


def main() -> None:
    cfg = parse_args()
    cfg.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))
    sys.path.insert(0, str(ROOT))
    dev = device_for(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = load_gpt2(dev)
    toks, rows = tail_windows()
    if cfg.windows:
        toks, rows = toks[:cfg.windows], rows[:cfg.windows]
    err = parity(model, toks, rows, dev)
    print(f"{len(toks)} held-out windows; site vs harvested rows, max relative error {err:.2e}")
    if err > 1e-3:
        raise SystemExit("the hook does not read the harvested stream; stopping")

    H = torch.cat([capture(model, torch.as_tensor(toks[i:i + cfg.batch], device=dev))
                   for i in range(0, len(toks), cfg.batch)])       # (n, t, d)
    norms = natural_norms(H.float().cpu().numpy())
    print("natural norm, over its window's median, percentiles "
          f"{norms['percentiles']}: {norms['over_window_median']}")

    pool = H.reshape(-1, H.shape[-1])
    beh = behaviour(model, toks, pool, pool.mean(0), cfg.c, cfg.batch, dev, cfg.seed)
    with open(cfg.out / "behaviour.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "kind", "c", "size", "kl", "kl_se", "dce", "dce_se",
                    "top1_agree", "n_windows"])
        for (scheme, kind, c), r in beh.items():
            n = len(r["kl"])
            w.writerow([scheme, kind, c, round(abs(c - 1), 4),
                        r["kl"].mean(), r["kl"].std(ddof=1) / np.sqrt(n),
                        r["dce"].mean(), r["dce"].std(ddof=1) / np.sqrt(n),
                        r["top1"].mean(), n])

    pool_np = pool.float().cpu().numpy()
    pick = rng.choice(len(pool_np), size=min(cfg.rows, len(pool_np)), replace=False)
    X0 = pool_np[pick]
    conds = code_inputs(X0, pool_np, cfg.c, rng)
    code_rows = []
    for run in cfg.models:
        ch, used = onto_changes(run, X0, conds, cfg.code_batch)
        code_rows += [[f"{run.name}@{used}", "onto", k[1], k[2], v, ""]
                      for k, v in ch.items()]
        print(f"  coded {run.name} step {used}", flush=True)
    for run in cfg.saes:
        for k, (jac, rel) in sae_changes(run, X0, conds, 1024).items():
            code_rows.append([run.name, "sae", k[1], k[2], jac, rel])
        print(f"  coded {run.name}", flush=True)
    with open(cfg.out / "codes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "family", "kind", "c", "changed", "code_rel_change"])
        w.writerows(code_rows)

    (cfg.out / "norms.json").write_text(json.dumps(
        {**norms, "parity_max_rel_err": err, "n_windows": len(toks),
         "code_rows": len(X0), "seed": cfg.seed}, indent=1))

    print("\nbehaviour, every position (KL nats/prediction; dCE nats/token):")
    print(f"{'c':>6} {'rescale KL':>11} {'orth KL':>9} {'data KL':>9} "
          f"{'rescale dCE':>12} {'orth dCE':>9}")
    for c in cfg.c:
        r, o, d = (beh[("all", k, c)] for k in KINDS)
        print(f"{c:>6} {r['kl'].mean():>11.4f} {o['kl'].mean():>9.4f} "
              f"{d['kl'].mean():>9.4f} {r['dce'].mean():>12.4f} {o['dce'].mean():>9.4f}")
    print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    main()

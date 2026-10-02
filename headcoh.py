"""Head/group semantic-coherence assessment: do a model's trained (or
discovered) groups collect features that MEAN similar things? headstruct.py
measures categorical form (exhaustive/exclusive firing); this measures
whether a head's entries look like alternative values of one shared
dimension rather than an arbitrary partition cell.

Geometry view (quota-free, covers every head): each entry has an exact
decode direction in SONAR output space -- Ontologizer: the row of the
end-to-end linear map G with Y = P_flat @ G (refit.onto_linear_model);
SAE: its W_dec row. SONAR space is a semantic sentence space, so
within-head direction similarity is semantic similarity of what the
entries decode to. Per head: mean pairwise cosine and top-singular-value
energy fraction, each z-scored against size-matched random subsets of the
same pool (same layer for Ontologizer heads). --metric whitened measures
the same similarities under the training objective's inverse-variance
inner product instead of raw SONAR cosine (writes to <out>_w).

Description view (--desc-dir): SONAR-encode the autointerp description
texts of the campaign's sampled features and pool within-head pairwise
cosine over heads with >=2 described entries, against a head-label
permutation null. Coverage is sparse (~256 sampled features), so this is
one global z, not per-head.

  uv run python headcoh.py --model onto \\
      --ckpt data/out/sonar/multilingual/resid_nc \\
      --desc-dir data/out/sonar/autointerp/onto/onto --desc-mode cacts
  uv run python headcoh.py --model sae \\
      --ckpt data/out/sonar/sae/m5120_g160top1/params.npz
  uv run python headcoh.py --model sae \\
      --ckpt data/out/sonar/sae/m11264_k32/params.npz \\
      --assignment data/out/sonar/sae/m11264_k32/headstruct/assignment.npy
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import json
import numpy as np
from pathlib import Path


# ---------- pure helpers (unit-tested) ----------

def unit(R, eps=1e-9):
    return R / (np.linalg.norm(R, axis=-1, keepdims=True) + eps)


def mean_pairwise_cos(R):
    """Mean off-diagonal cosine similarity of the rows of R (n, d)."""
    n = len(R)
    if n < 2:
        return np.nan
    Rh = unit(R)
    s = Rh.sum(0)
    return float((s @ s - n) / (n * (n - 1)))


def sv_energy(R):
    """Top singular value's share of total energy of unit-row R: 1/n for
    isotropic rows, 1.0 when all rows are parallel."""
    if len(R) < 2:
        return np.nan
    s = np.linalg.svd(unit(R), compute_uv=False)
    return float(s[0] ** 2 / (s ** 2).sum())


def null_table(pool, sizes, n_null, seed):
    """Null mean/std of both stats for random size-s subsets of pool rows,
    for each distinct s in sizes. Returns {s: (mc_mu, mc_sd, sv_mu, sv_sd)}."""
    rng = np.random.default_rng(seed)
    out = {}
    for s in sorted(set(int(s) for s in sizes)):
        if s < 2 or s > len(pool):
            continue
        mc, sv = [], []
        for _ in range(n_null):
            sub = pool[rng.choice(len(pool), s, replace=False)]
            mc.append(mean_pairwise_cos(sub))
            sv.append(sv_energy(sub))
        out[s] = (np.mean(mc), np.std(mc) + 1e-12,
                  np.mean(sv), np.std(sv) + 1e-12)
    return out


def group_zscores(R, labels, pools, n_null, seed):
    """Per-group coherence stats + z vs size-matched nulls. labels: (n,)
    group id per row, -1 = ungrouped; pools: (n,) pool id per row (nulls
    are drawn within-pool, e.g. within-layer)."""
    rows = []
    tables = {}
    for g in np.unique(labels[labels >= 0]):
        idx = np.where(labels == g)[0]
        pid = int(pools[idx[0]])
        pool = R[pools == pid]
        key = pid
        if key not in tables:
            gsizes = [np.sum(labels == gg) for gg in np.unique(labels)
                      if gg >= 0 and pools[labels == gg][0] == pid]
            tables[key] = null_table(pool, gsizes, n_null, seed + pid)
        mc, sv = mean_pairwise_cos(R[idx]), sv_energy(R[idx])
        tab = tables[key].get(len(idx))
        if tab is None:
            continue
        mc_mu, mc_sd, sv_mu, sv_sd = tab
        rows.append({"group": int(g), "pool": pid, "size": len(idx),
                     "mean_cos": mc, "z_cos": (mc - mc_mu) / mc_sd,
                     "sv1": sv, "z_sv": (sv - sv_mu) / sv_sd})
    return rows


def pooled_within_cos(E, labels):
    """Mean cosine over all within-group pairs (groups with >=2 rows)."""
    tot, cnt = 0.0, 0
    Eh = unit(E)
    for g in np.unique(labels[labels >= 0]):
        idx = np.where(labels == g)[0]
        n = len(idx)
        if n < 2:
            continue
        s = Eh[idx].sum(0)
        tot += s @ s - n
        cnt += n * (n - 1)
    return tot / cnt if cnt else np.nan


def perm_z(E, labels, pools, n_perm, seed):
    """z of pooled_within_cos against within-pool label permutations."""
    obs = pooled_within_cos(E, labels)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        lp = labels.copy()
        for p in np.unique(pools):
            idx = np.where(pools == p)[0]
            lp[idx] = lp[idx[rng.permutation(len(idx))]]
        null.append(pooled_within_cos(E, lp))
    return obs, (obs - np.mean(null)) / (np.std(null) + 1e-12)


# ---------- direction/label loading ----------

def onto_directions(ckpt, step, temperature):
    from refit import onto_linear_model
    _, G, meta = onto_linear_model(ckpt, step, temperature)
    l, h, k = meta["l"], meta["h"], meta["k"]
    R = np.asarray(G)                       # (l*h*k, d), f=(l_i*h+h_i)*k+k_i
    f = np.arange(len(R))
    labels = f // k                          # head id
    pools = f // (h * k)                     # layer id: null within layer
    return R, labels, pools, {"l": l, "h": h, "k": k}


def sae_directions(ckpt, assignment):
    raw = np.load(ckpt)
    R = np.asarray(raw["W_dec"])
    m = len(R)
    meta_path = Path(ckpt).parent / "meta.json"
    run_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if assignment:
        labels = np.load(assignment).astype(int)
        assert len(labels) == m, "assignment length != W_dec rows"
    elif run_meta.get("groups"):
        labels = np.arange(m) // (m // run_meta["groups"])
    else:
        raise SystemExit("ungrouped SAE: pass --assignment "
                         "(e.g. headstruct's assignment.npy)")
    pools = np.zeros(m, int)
    return R, labels, pools, run_meta


def load_descriptions(desc_dir, mode):
    feats, texts = [], []
    with open(Path(desc_dir) / "descriptions.jsonl") as f:
        for rec in map(json.loads, f):
            if rec["mode"] == mode and rec["description"].strip():
                feats.append(rec["feature"])
                texts.append(rec["description"])
    return np.array(feats), texts


def encode_texts(texts, device, b=64):
    import torch as t
    from ontologize.data.pretrained import pretrained_transformer
    from autointerp import ENCODER_ID
    enc, tok = pretrained_transformer(ENCODER_ID, "float32",
                                      dev=t.device(device))
    out = []
    with t.no_grad():
        for i in range(0, len(texts), b):
            batch = tok(texts[i:i + b], return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(device)
            hs = enc(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            e = (hs * mask).sum(1) / mask.sum(1)
            out.append(t.nn.functional.normalize(e, dim=-1).cpu().numpy())
    del enc
    return np.concatenate(out)


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["onto", "sae"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--assignment", help="per-latent group labels .npy "
                   "(discovered groups); default = trained groups from "
                   "meta.json")
    p.add_argument("--metric", choices=["raw", "whitened"], default="raw",
                   help="inner product for direction similarity: raw SONAR "
                   "cosine, or cosine under the training objective's "
                   "inverse-variance metric (rows scaled by sqrt(w) before "
                   "cosine); whitened runs write to <out>_w")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--n-null", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--desc-dir", help="autointerp dir with descriptions.jsonl")
    p.add_argument("--desc-mode", default="cacts")
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out")
    cfg = p.parse_args()

    if cfg.model == "onto":
        R, labels, pools, meta = onto_directions(
            cfg.ckpt, cfg.step, cfg.temperature)
        out = Path(cfg.out or Path(cfg.ckpt) / "headcoh")
    else:
        R, labels, pools, meta = sae_directions(cfg.ckpt, cfg.assignment)
        sub = "headcoh_discovered" if cfg.assignment else "headcoh"
        out = Path(cfg.out or Path(cfg.ckpt).parent / sub)
    if cfg.metric == "whitened":
        w = np.load(cfg.mse_weights)
        assert R.shape[1] == len(w), "mse-weights dim != direction dim"
        R = R * np.sqrt(w)
        if not cfg.out:
            out = out.with_name(out.name + "_w")
    out.mkdir(parents=True, exist_ok=True)

    rows = group_zscores(R, labels, pools, cfg.n_null, cfg.seed)
    with open(out / "heads.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    z_cos = np.array([r["z_cos"] for r in rows])
    z_sv = np.array([r["z_sv"] for r in rows])
    summary = {
        "model": cfg.model, "ckpt": str(cfg.ckpt),
        "assignment": cfg.assignment, "metric": cfg.metric,
        "n_groups": len(rows),
        "mean_z_cos": float(z_cos.mean()), "median_z_cos": float(np.median(z_cos)),
        "frac_z_cos_gt2": float((z_cos > 2).mean()),
        "frac_z_cos_lt_neg2": float((z_cos < -2).mean()),
        "mean_z_sv": float(z_sv.mean()),
        "frac_z_sv_gt2": float((z_sv > 2).mean()),
    }
    print(f"geometry: {len(rows)} groups | mean z_cos {z_cos.mean():+.2f} "
          f"median {np.median(z_cos):+.2f} | frac z_cos>2 "
          f"{(z_cos > 2).mean():.2f} | frac z_cos<-2 {(z_cos < -2).mean():.2f}"
          f" | mean z_sv {z_sv.mean():+.2f}")
    top = sorted(rows, key=lambda r: -r["z_cos"])[:5]
    for r in top:
        print(f"  most coherent: pool {r['pool']} group {r['group']} "
              f"size {r['size']} mean_cos {r['mean_cos']:.3f} "
              f"z {r['z_cos']:+.1f}")

    if cfg.desc_dir:
        feats, texts = load_descriptions(cfg.desc_dir, cfg.desc_mode)
        if len(feats) < 4:
            print(f"desc: <4 '{cfg.desc_mode}' descriptions in "
                  f"{cfg.desc_dir}; skipping")
        else:
            E = encode_texts(texts, cfg.device)
            dl, dp = labels[feats], pools[feats]
            n_pairs = sum(c * (c - 1) for c in
                          np.unique(dl, return_counts=True)[1] if c > 1)
            obs, z = perm_z(E, dl, dp, cfg.n_perm, cfg.seed)
            summary["desc_mode"] = cfg.desc_mode
            summary["desc_n"] = int(len(feats))
            summary["desc_within_cos"] = float(obs)
            summary["desc_z"] = float(z)
            print(f"desc[{cfg.desc_mode}]: n={len(feats)} "
                  f"within-head pairs={n_pairs // 2} "
                  f"within-cos {obs:.3f} perm z {z:+.2f}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

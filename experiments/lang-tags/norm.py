"""Does the cache's L2 normalization lose anything that matters?

`encode_corpus.py` stores each SONAR embedding divided by its norm. SONAR
itself does not normalize, and every script that decodes rescales a unit
vector to one corpus constant, `textfid.SONAR_NORM` (0.307). So the cache
keeps each row's direction and discards one number per row, its norm.
This script asks what that number carries, on `retag.py`'s re-encodes,
which kept the pre-normalization norm of every row under its intended
tag:

  stats    the norm's spread and what it tracks (length, language); the
           share of the raw embeddings' whitened variance it carries (FVU
           of the direction at one constant norm, against the raw
           embeddings); how much of it the direction predicts (ridge R^2,
           held-out rows)
  decode   SONAR's decoder, as the eval scripts run it (forced eng_Latn,
           greedy, 48 tokens), on the same rows at their true norm, at
           SONAR_NORM, and at the norm predicted from the direction, plus
           the cache's own vector (English source tag) at SONAR_NORM:
           chrF2 between the decodes, and how much of the embedding the
           decode keeps (cosine between the re-encoded decode and the
           true embedding's direction)

  uv run python experiments/lang-tags/norm.py      # after retag.py encode
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from jaxtyping import Float, Int

from retag import ENCODER, encode_rows, eta2, wfvu
from textfid import SONAR_NORM, SonarDecoder, chrf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--out", default="data/out/sonar/langtags")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--per-lang", type=int, default=12,
                   help="rows per language decoded")
    p.add_argument("--b", type=int, default=32, help="decode/encode batch")
    return p.parse_args()


def occurrence(y: Int[np.ndarray, "n"]) -> Int[np.ndarray, "n"]:
    """Each row's index within its group, for per-group splits."""
    occ = np.zeros(len(y), np.int64)
    for j in np.unique(y):
        idx = np.flatnonzero(y == j)
        occ[idx] = np.arange(len(idx))
    return occ


def ridge_fit(F: Float[np.ndarray, "n f"], t: Float[np.ndarray, "n"],
              lam: float) -> Float[np.ndarray, "f1"]:
    """Ridge weights for t on [F, 1], bias unpenalized only by scale."""
    A = np.concatenate([F, np.ones((len(F), 1))], 1).astype(np.float64)
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ t)


def ridge_predict(F: Float[np.ndarray, "n f"], W: Float[np.ndarray, "f1"]
                  ) -> Float[np.ndarray, "n"]:
    return np.concatenate([F, np.ones((len(F), 1))], 1) @ W


def decode_raw(sd: SonarDecoder, Y: Float[np.ndarray, "n d"]) -> List[str]:
    """`SonarDecoder.texts` without its rescaling, so each row is decoded
    at the norm it is given."""
    from transformers.modeling_outputs import BaseModelOutput
    t = sd.t
    out = []
    with t.no_grad():
        for i in range(0, len(Y), sd.b):
            E = t.from_numpy(np.ascontiguousarray(Y[i:i + sd.b])).to(
                sd.dev, t.float32)
            gen = sd.dec.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=E.unsqueeze(1)),
                forced_bos_token_id=sd.eng, max_length=sd.max_length,
                num_beams=1, repetition_penalty=1.2)
            out += [sd.text(g) for g in gen.cpu()]
    return out


def main() -> None:
    cfg = parse_args()
    os.chdir(ROOT)
    out = Path(cfg.out)
    E = np.load(out / "encodes.npz")
    rows, langs = E["rows"], E["langs"]
    U = E["X_int"].astype(np.float64)
    r = E["norm_int"].astype(np.float64)
    ntok = E["ntok_int"]
    names, y = np.unique(langs, return_inverse=True)
    w = np.load(cfg.mse_weights)
    R = r[:, None] * U                      # SONAR's own embeddings
    base = float((R.var(0) * w).mean())
    summary: Dict = {"n": int(len(r))}

    # ---- what the norm is
    med = np.array([np.median(r[y == j]) for j in range(len(names))])
    eta_lang = eta2(r[:, None], y, np.ones(1))
    rho = float(np.corrcoef(r, np.log(ntok))[0, 1])
    print(f"raw norm: median {np.median(r):.4f}, SD {r.std():.4f} "
          f"(CV {r.std() / r.mean():.3f}); per-language medians "
          f"{med.min():.3f}-{med.max():.3f}")
    print(f"  language explains {eta_lang:.3f} of its variance; corr with "
          f"log token count {rho:+.3f}")

    # ---- what dropping it costs, in the objective's metric
    c_ls = float((r * (U ** 2 * w).sum(1)).sum() / (U ** 2 * w).sum(1).sum())
    fvu_ls = wfvu(c_ls * U, R, w, base)
    fvu_ref = wfvu(SONAR_NORM * U, R, w, base)
    occ = occurrence(y)
    tr, te = occ % 4 != 0, occ % 4 == 0
    W = ridge_fit(U[tr], r[tr], cfg.ridge)
    r_hat = ridge_predict(U, W)
    r2 = float(1 - ((r[te] - r_hat[te]) ** 2).sum()
               / ((r[te] - r[te].mean()) ** 2).sum())
    fvu_pred = wfvu(r_hat[te, None] * U[te], R[te], w, base)
    print(f"\nwhitened FVU of the direction at one constant norm, against "
          f"SONAR's own embeddings: {fvu_ls:.4f} (best constant {c_ls:.4f}),"
          f" {fvu_ref:.4f} at SONAR_NORM {SONAR_NORM}")
    print(f"the direction predicts the norm with held-out R^2 {r2:.3f}; at "
          f"the predicted norm the FVU is {fvu_pred:.4f}")
    print(f"language eta^2: {eta2(U, y, np.ones_like(w)):.4f} on the unit "
          f"vectors, {eta2(R, y, np.ones_like(w)):.4f} on the raw ones")
    summary["stats"] = {
        "norm_median": float(np.median(r)), "norm_sd": float(r.std()),
        "lang_median_min": float(med.min()), "lang_median_max": float(med.max()),
        "norm_eta2_lang": eta_lang, "corr_log_tokens": rho,
        "fvu_const_best": fvu_ls, "best_const": c_ls,
        "fvu_const_sonar_norm": fvu_ref, "ridge_r2": r2,
        "fvu_predicted_norm": fvu_pred,
        "eta2_lang_unit": eta2(U, y, np.ones_like(w)),
        "eta2_lang_raw": eta2(R, y, np.ones_like(w))}

    # ---- decoding: true norm vs the constant the eval scripts use
    pick = np.flatnonzero(occ < cfg.per_lang)
    X_c = np.asarray(np.load(cfg.cache, mmap_mode="r")[rows[pick]], np.float64)
    inputs = {
        "true norm": R[pick],
        "SONAR_NORM": SONAR_NORM * U[pick],
        "predicted norm": r_hat[pick, None] * U[pick],
        "cache (eng tag), SONAR_NORM": SONAR_NORM * X_c,
    }
    import torch as t
    from ontologize.data.pretrained import pretrained_transformer
    dev = "cuda" if t.cuda.is_available() else "cpu"
    sd = SonarDecoder(device=dev, b_decode=cfg.b)
    texts = {k: decode_raw(sd, v.astype(np.float32)) for k, v in inputs.items()}
    del sd
    enc, tok = pretrained_transformer(ENCODER, "float32", dev=t.device(dev))
    # a decode's own row against another row of the same language: the
    # shared mean direction alone gives every pair ~0.3 cosine
    yp = y[pick]
    perm = np.arange(len(pick))
    for j in np.unique(yp):
        idx = np.flatnonzero(yp == j)
        perm[idx] = np.roll(idx, 1)
    keep, excess = {}, {}
    for k, txt in texts.items():
        V, _, _ = encode_rows(enc, tok, [{"text": s} for s in txt],
                              ["eng_Latn"] * len(txt), cfg.b, t.device(dev))
        keep[k] = (V * U[pick]).sum(1)
        excess[k] = keep[k] - (V * U[pick][perm]).sum(1)
    ref = texts["true norm"]
    print(f"\ndecoded {len(pick)} rows ({cfg.per_lang} per language), "
          f"forced English; kept = cosine of the re-encoded decode to the "
          f"row's true direction, excess = kept minus the same to another "
          f"row of its language, change = excess minus the true-norm "
          f"decode's (paired, with its SE):")
    print(f"  {'decoder input':<30} {'chrF2 to true':>14} {'same text':>10} "
          f"{'kept':>7} {'excess':>7} {'change':>16}")
    dec = {}
    for k, txt in texts.items():
        cf = np.array([chrf(a, b) for a, b in zip(ref, txt)])
        same = float(np.mean([a == b for a, b in zip(ref, txt)]))
        dif = excess[k] - excess["true norm"]
        se = float(dif.std(ddof=1) / np.sqrt(len(dif)))
        dec[k] = {"chrf_to_true": float(cf.mean()), "same_text": same,
                  "kept_cos": float(keep[k].mean()),
                  "excess_cos": float(excess[k].mean()),
                  "change": float(dif.mean()), "change_se": se}
        print(f"  {k:<30} {cf.mean():>14.3f} {same:>10.3f} "
              f"{keep[k].mean():>7.3f} {excess[k].mean():>7.3f} "
              f"{dif.mean():>+9.4f} ± {se:.4f}")
    summary["decode"] = {"rows": int(len(pick)), "conditions": dec}
    (out / "norm.json").write_text(json.dumps(summary, indent=2))
    with open(out / "norm_decodes.jsonl", "w") as f:
        for i, row in enumerate(rows[pick]):
            f.write(json.dumps({"row": int(row), "lang": str(langs[pick][i]),
                                **{k: texts[k][i] for k in texts}},
                               ensure_ascii=False) + "\n")
    print(f"\n-> {out / 'norm.json'}, {out / 'norm_decodes.jsonl'}")


if __name__ == "__main__":
    main()

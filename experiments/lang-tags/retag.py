"""What the 5-entry LANG_MAP cost the SONAR cache.

`encode_corpus.py` tokenized every row through `TokenizeTransform`, whose
`LANG_MAP` knows en/fr/es/de/zh and falls back to `eng_Latn`, so 80 of
the 86 mC4 configs were encoded under the English source tag rather than
their own (`langs.MC4_TO_SONAR`). The tag is a token at the head of the
sequence: every position attends to it and the mean pool includes it.
This script measures how far that moved the embeddings, and whether the
moves matter for what the project reports.

Stages (outputs under --out):

  texts    replay the deterministic mC4 stream for the first --per-lang
           rows of every language (round-robin interleave: rows
           [0, 86 * per_lang)), checking each row's language against the
           cache's .langs.npy sidecar, and keep the full text
           -> texts.jsonl
  encode   encode every row under its intended tag, and the first
           --verify rows of each language again under the tag the cache
           was built with, to check the replay reproduces the cache
           -> encodes.npz
  compare  the cached rows against their intended-tag versions:
           embedding shift (cosine, whitened FVU of one as a
           reconstruction of the other, how much of it is a per-language
           offset), language structure (eta^2, within/between cosine,
           86-way ridge probe), and the shipped models on both inputs
           (FVU, per-layer code agreement against a same-size random
           perturbation, best-head language NMI)
           -> summary.json, langs.csv

  uv run python experiments/lang-tags/retag.py texts
  uv run python experiments/lang-tags/retag.py encode
  uv run python experiments/lang-tags/retag.py compare

The sample is rows from the head of the cache, which the models trained
on; `compare` also scores the models on the cache tail so the in-sample
advantage is visible next to the tag effect.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))

import numpy as np
from jaxtyping import Float, Int

from ontologize.data.langs import MC4_TO_SONAR
from ontologize.data.loaders import LANG_MAP

ENCODER = "cointegrated/SONAR_200_text_encoder"
MAXLEN = 512


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["texts", "encode", "compare"])
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--out", default="data/out/sonar/langtags")
    p.add_argument("--per-lang", type=int, default=400,
                   help="rows per language; the sample is the first "
                        "86 * per_lang rows of the cache")
    p.add_argument("--verify", type=int, default=16,
                   help="rows per language re-encoded under the cached tag")
    p.add_argument("--b", type=int, default=32, help="encoding batch size")
    p.add_argument("--onto", nargs="*", default=[
        "data/out/sonar/multilingual/ste_h76_init01:0.00015",
        "data/out/sonar/multilingual/ste_l1_h380_i01:0.00015",
        "data/out/sonar/multilingual/resid_nc:0.03"],
        help="Ontologizer checkpoint dirs as path:temperature")
    p.add_argument("--sae", nargs="*", default=[
        "data/out/sonar/sae_conv/m11264_k32",
        "data/out/sonar/sae_conv/m11264_k160"],
        help="sae.py run dirs (params.npz + meta.json)")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--tail-rows", type=int, default=32768,
                   help="cache tail rows for the models' held-out FVU")
    p.add_argument("--eval-b", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cached_tag(lang: str) -> str:
    """The source tag `encode_corpus.py` actually used for an mC4 config."""
    return LANG_MAP.get(lang, "eng_Latn")


def script_of(lang: str) -> str:
    """Script a row is written in: the intended tag's suffix, except the
    romanized `-Latn` configs, whose intended tag names the native script."""
    return "Latn" if lang.endswith("-Latn") else MC4_TO_SONAR[lang].split("_")[1]


# ---------- stage: texts ----------

def texts(cfg: argparse.Namespace) -> None:
    from tqdm import tqdm
    from ontologize.data.loaders import HFDataSource
    from ontologize.data.multilingual import mc4_data

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    n = len(MC4_TO_SONAR) * cfg.per_lang
    L = np.load(Path(cfg.cache).with_name(
        Path(cfg.cache).name.replace(".npy", ".langs.npy")), mmap_mode="r")
    path = out / "texts.jsonl"
    have = 0
    if path.exists():
        with open(path) as f:
            have = sum(1 for _ in f)
    if have >= n:
        print(f"texts: {have} rows already in {path}")
        return

    ds = mc4_data("allenai/c4", split="train", streaming=True)
    it = iter(HFDataSource(ds, text_key="text"))
    with open(path, "a") as f:
        for row in tqdm(range(n), desc="streaming mC4"):
            item = next(it)
            if item["lang"] != L[row]:
                raise RuntimeError(
                    f"stream drift at row {row}: stream lang {item['lang']!r}"
                    f" != cached {L[row]!r}")
            if row >= have:
                f.write(json.dumps({"row": row, "lang": item["lang"],
                                    "text": item["text"]},
                                   ensure_ascii=False) + "\n")
    print(f"texts: {n} rows -> {path}")


# ---------- stage: encode ----------

def encode_rows(model, tokenizer, docs: List[Dict], tags: List[str], b: int,
                dev) -> Tuple[Float[np.ndarray, "n d"], Float[np.ndarray, "n"],
                              Int[np.ndarray, "n"]]:
    """SONAR embeddings of `docs` under per-row source `tags`, pooled as
    `encode_corpus.py` pools (masked mean over the 512-token window, then
    L2 normalization). Returns (embeddings, pre-normalization norms,
    token counts). Rows are batched by length and padded to the batch's
    longest row; padding is masked, so this matches the cache's
    max_length padding up to float rounding (the verify rows check it)."""
    import torch as t
    from tqdm import tqdm

    ids, masks = [], []
    for doc, tag in zip(docs, tags):
        tokenizer.src_lang = tag
        tok = tokenizer(doc["text"], max_length=MAXLEN, truncation=True)
        ids.append(tok["input_ids"])
        masks.append(tok["attention_mask"])
    ntok = np.array([len(x) for x in ids])
    order = np.argsort(-ntok, kind="stable")
    E = np.zeros((len(docs), 1024), np.float32)
    norms = np.zeros(len(docs), np.float32)
    pad = tokenizer.pad_token_id
    for i in tqdm(range(0, len(docs), b), desc="encoding"):
        sel = order[i:i + b]
        w = int(ntok[sel].max())
        I = np.full((len(sel), w), pad, np.int64)
        M = np.zeros((len(sel), w), np.int64)
        for j, r in enumerate(sel):
            I[j, :ntok[r]] = ids[r]
            M[j, :ntok[r]] = masks[r]
        with t.no_grad():
            H = model(input_ids=t.as_tensor(I, device=dev),
                      attention_mask=t.as_tensor(M, device=dev)).last_hidden_state
            m = t.as_tensor(M, device=dev, dtype=H.dtype)[..., None]
            mean = (H * m).sum(1) / m.sum(1).clamp(min=1e-9)
            nrm = mean.norm(dim=-1, keepdim=True)
            E[sel] = (mean / nrm.clamp(min=1e-9)).cpu().numpy()
            norms[sel] = nrm[:, 0].cpu().numpy()
    return E, norms, ntok


def encode(cfg: argparse.Namespace) -> None:
    import torch as t
    from ontologize.data.pretrained import pretrained_transformer

    out = Path(cfg.out)
    with open(out / "texts.jsonl") as f:
        docs = [json.loads(line) for line in f]
    docs.sort(key=lambda d: d["row"])
    rows = np.array([d["row"] for d in docs])
    assert (rows == np.arange(len(rows))).all(), "texts.jsonl has gaps"
    langs = [d["lang"] for d in docs]

    dev = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
    # float32, as encode_corpus.py: lower precision moves the embeddings
    model, tokenizer = pretrained_transformer(ENCODER, "float32", dev=dev)
    missing = sorted(c for c in {MC4_TO_SONAR[l] for l in langs}
                     if tokenizer.convert_tokens_to_ids(c)
                     == tokenizer.unk_token_id)
    assert not missing, f"tokenizer lacks language codes {missing}"

    X_int, n_int, t_int = encode_rows(
        model, tokenizer, docs, [MC4_TO_SONAR[l] for l in langs], cfg.b, dev)

    seen: Dict[str, int] = {}
    ver = []
    for i, l in enumerate(langs):
        if seen.get(l, 0) < cfg.verify:
            ver.append(i)
            seen[l] = seen.get(l, 0) + 1
    ver = np.array(ver)
    X_ver, n_ver, t_ver = encode_rows(
        model, tokenizer, [docs[i] for i in ver],
        [cached_tag(langs[i]) for i in ver], cfg.b, dev)

    np.savez(out / "encodes.npz", rows=rows, langs=np.array(langs),
             X_int=X_int, norm_int=n_int, ntok_int=t_int,
             ver_rows=ver, X_ver=X_ver, norm_ver=n_ver, ntok_ver=t_ver)
    print(f"encode: {len(rows)} rows, {len(ver)} verify rows "
          f"-> {out / 'encodes.npz'}")


# ---------- stage: compare: embedding-level ----------

def wfvu(A: Float[np.ndarray, "n d"], B: Float[np.ndarray, "n d"],
         w: Float[np.ndarray, "d"], base: float) -> float:
    """Whitened FVU of A as a reconstruction of B, against `base`, the
    whitened total variance per coordinate (pareto.py's convention)."""
    return float((((A - B) ** 2).mean(0) * w).mean() / base)


def eta2(X: Float[np.ndarray, "n d"], g: Int[np.ndarray, "n"],
         w: Float[np.ndarray, "d"]) -> float:
    """Between-group share of (w-weighted) variance, as headeta.py."""
    Xc = (X - X.mean(0)) * np.sqrt(w)
    n = int(g.max()) + 1
    cnt = np.bincount(g, minlength=n)
    M = np.zeros((n, X.shape[1]))
    np.add.at(M, g, Xc)
    M /= np.maximum(cnt[:, None], 1)
    return float(((cnt / len(g))[:, None] * M ** 2).sum()
                 / (Xc ** 2).sum(1).mean())


def within_between(X: Float[np.ndarray, "n d"], g: Int[np.ndarray, "n"]
                   ) -> Tuple[float, float]:
    """Mean pairwise cosine of unit rows within the same group (averaged
    over groups) and across groups, from per-group sums."""
    n = int(g.max()) + 1
    S = np.zeros((n, X.shape[1]))
    np.add.at(S, g, X.astype(np.float64))
    cnt = np.bincount(g, minlength=n).astype(np.float64)
    within = ((S ** 2).sum(1) - cnt) / (cnt * (cnt - 1))
    tot = S.sum(0)
    cross = (tot @ tot - (S ** 2).sum()) / (cnt.sum() ** 2 - (cnt ** 2).sum())
    return float(within.mean()), float(cross)


def offset_shares(D: Float[np.ndarray, "n d"], g: Int[np.ndarray, "n"],
                  w: Float[np.ndarray, "d"]) -> Tuple[float, float, float]:
    """How much of the shift's (w-weighted) energy is a constant: the
    share in its overall mean (every row moved alike), the share in the
    per-group means beyond that (each language moved alike), and the
    second share's chance level for row shifts unrelated to the group,
    (groups - 1) / rows."""
    Dw = D * np.sqrt(w)
    tot = float((Dw ** 2).sum())
    mu = Dw.mean(0)
    n = int(g.max()) + 1
    M = np.zeros((n, D.shape[1]))
    np.add.at(M, g, Dw)
    cnt = np.bincount(g, minlength=n)
    M = M / np.maximum(cnt[:, None], 1) - mu
    return (float(len(D) * (mu ** 2).sum() / tot),
            float((cnt[:, None] * M ** 2).sum() / tot),
            float((n - 1) / len(D)))


# ---------- stage: compare: models ----------

def onto_runner(ckpt: str, T: float):
    """(forward, model) for an Ontologizer: forward(X) returns the
    reconstruction and every head's argmax entry, with each layer's
    gain-shape split reproduced as pareto.py's `probe` does."""
    import functools
    import jax
    import jax.numpy as jnp
    from pareto import load_onto

    model, params, step = load_onto(ckpt, 0)

    def probe(module, X):
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        E_in = module.constinput(E)
        A = []
        for i, de in enumerate(module.dictencs):
            U, G = de.gainshape_in(E_in)
            P = de.dict.cluster(de.classifier(U), T)
            A.append(jnp.argmax(P, -1))
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
            if i < module.l - 1:
                E_in = module.nextinput(X, R, P.reshape(P.shape[0], -1))
        return module.decode(R), jnp.stack(A, 1)

    f = jax.jit(functools.partial(model.apply, {"params": params},
                                  method=probe))
    return f, model, step


def run_batched(f, X: Float[np.ndarray, "n d"], b: int):
    """Apply `f` in batches; a short final batch is padded so every call
    has one shape (no recompilation), and the padding is dropped."""
    outs = []
    for i in range(0, len(X), b):
        x = X[i:i + b]
        pad = b - len(x)
        if pad:
            x = np.concatenate([x, np.repeat(x[-1:], pad, 0)])
        y = f(x)
        y = y if isinstance(y, tuple) else (y,)
        outs.append([np.asarray(v)[:b - pad] for v in y])
    return [np.concatenate(v) for v in zip(*outs)]


def row_err(Y: Float[np.ndarray, "n d"], X: Float[np.ndarray, "n d"],
            w: Float[np.ndarray, "d"]) -> Float[np.ndarray, "n"]:
    """Per-row whitened squared error, mean over coordinates."""
    return (((Y - X) ** 2) * w).mean(1)


def head_agreement(A: Int[np.ndarray, "n l h"], B: Int[np.ndarray, "n l h"]
                   ) -> Float[np.ndarray, "l"]:
    return (A == B).mean((0, 2))


def chance_agreement(A: Int[np.ndarray, "n l h"], k: int
                     ) -> Float[np.ndarray, "l"]:
    """Agreement two independent rows would show: sum_j p_j^2 per head
    from the entries' usage, averaged over each layer's heads."""
    n, l, h = A.shape
    out = np.zeros(l)
    for i in range(l):
        P = np.stack([np.bincount(A[:, i, j], minlength=k) / n
                      for j in range(h)])
        out[i] = (P ** 2).sum(1).mean()
    return out


def random_shift(X: Float[np.ndarray, "n d"], D: Float[np.ndarray, "n d"],
                 rng: np.random.Generator) -> Float[np.ndarray, "n d"]:
    """X moved by isotropic random directions with each row's shift norm,
    then renormalized like the cache: the null of the same size as D."""
    Z = rng.standard_normal(X.shape).astype(np.float32)
    Z *= np.linalg.norm(D, axis=1, keepdims=True) / np.linalg.norm(
        Z, axis=1, keepdims=True)
    Y = X + Z
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def nmi_best(A: Int[np.ndarray, "n heads"], y: Int[np.ndarray, "n"], k: int,
             n_cls: int, rng: np.random.Generator) -> Tuple[float, int, float]:
    """Best per-head language NMI, its head index, and a label-shuffled
    null (mean over every 37th head, as headlang.py)."""
    from headlang import nmi
    S = np.array([nmi(A[:, j], y, k, n_cls)[0] for j in range(A.shape[1])])
    yp = y[rng.permutation(len(y))]
    null = float(np.mean([nmi(A[:, j], yp, k, n_cls)[0]
                          for j in range(0, A.shape[1], 37)]))
    return float(S.max()), int(S.argmax()), null


def compare(cfg: argparse.Namespace) -> None:
    from headlang import ridge_probe

    out = Path(cfg.out)
    rng = np.random.default_rng(cfg.seed)
    E = np.load(out / "encodes.npz")
    rows, langs = E["rows"], E["langs"]
    X_int = E["X_int"]
    mm = np.load(cfg.cache, mmap_mode="r")
    X_c = np.asarray(mm[rows], np.float32)
    w = np.load(cfg.mse_weights)
    base = float((X_c.var(0) * w).mean())
    names, y = np.unique(langs, return_inverse=True)
    changed = np.array([cached_tag(l) != MC4_TO_SONAR[l] for l in langs])
    roman = np.array([l.endswith("-Latn") for l in langs])
    scripts, ys = np.unique([script_of(l) for l in langs], return_inverse=True)
    summary: Dict = {"n": int(len(rows)), "per_lang": int(len(rows) // len(names)),
                     "langs": int(len(names)), "changed_share": float(changed.mean())}

    # replay check: the cached-tag re-encodes must reproduce the cache
    ver = E["ver_rows"]
    cv = (E["X_ver"] * X_c[ver]).sum(1)
    summary["verify"] = {"rows": int(len(ver)), "cos_min": float(cv.min()),
                         "cos_median": float(np.median(cv))}
    print(f"replay check on {len(ver)} rows: cos(re-encode, cache) "
          f"min {cv.min():.6f}  median {np.median(cv):.6f}")
    same = ~changed[ver]
    cu = (X_int[ver][same] * X_c[ver][same]).sum(1)
    print(f"  unchanged-tag rows, intended vs cache: min cos {cu.min():.6f}")

    # ---- embedding shift
    D = X_int - X_c
    cos = (X_int * X_c).sum(1)
    fvu_all = wfvu(X_int, X_c, w, base)
    fvu_ch = wfvu(X_int[changed], X_c[changed], w, base)
    s_glob, s_lang, s_null = offset_shares(D[changed], y[changed], w)
    ntok = E["ntok_int"]
    print(f"\n{changed.mean():.1%} of rows changed tag "
          f"({changed.sum()} of {len(rows)})")
    print(f"cos(intended, cached) on changed rows: median "
          f"{np.median(cos[changed]):.4f}  p05 "
          f"{np.quantile(cos[changed], .05):.4f}  min {cos[changed].min():.4f}")
    print(f"whitened FVU of the cached vector as a reconstruction of the "
          f"intended one: {fvu_all:.4f} over all rows, {fvu_ch:.4f} over "
          f"changed rows")
    print(f"shift energy in a common offset {s_glob:.3f}, in per-language "
          f"offsets beyond it {s_lang:.3f} (chance {s_null:.3f})")
    bins = [0, 32, 64, 128, 256, 511, 512]
    tab = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = changed & (ntok > lo) & (ntok <= hi)
        if s.any():
            tab.append({"tokens": f"{lo + 1}-{hi}", "rows": int(s.sum()),
                        "cos_median": float(np.median(cos[s])),
                        "fvu": wfvu(X_int[s], X_c[s], w, base)})
            print(f"  tokens {lo + 1:>3}-{hi:<3}: {s.sum():>5} rows  cos "
                  f"{np.median(cos[s]):.4f}  FVU {tab[-1]['fvu']:.4f}")
    summary["shift"] = {
        "cos_median_changed": float(np.median(cos[changed])),
        "cos_p05_changed": float(np.quantile(cos[changed], .05)),
        "fvu_all": fvu_all, "fvu_changed": fvu_ch,
        "fvu_romanized": wfvu(X_int[roman], X_c[roman], w, base),
        "offset_common": s_glob, "offset_lang": s_lang,
        "offset_lang_chance": s_null, "by_tokens": tab,
        "raw_norm_median": float(np.median(E["norm_int"])),
        "raw_norm_sd": float(np.std(E["norm_int"])),
        "variance_ratio": float((X_int.var(0) * w).mean() / base)}

    # ---- language structure, cached vs intended
    from pareto import gaussian_reference, whitened_spectrum
    # split on each row's index within its language: the stream is a
    # round robin over 86 languages, so a split on the row index itself
    # (row % 4) would hold out only the languages at even positions
    occ = np.zeros(len(rows), np.int64)
    for j in range(len(names)):
        idx = np.flatnonzero(y == j)
        occ[idx] = np.arange(len(idx))
    tr = occ % 4 != 0
    te = ~tr
    lang_rows = []
    struct = {}
    for tag, X in (("cached", X_c), ("intended", X_int)):
        lam = whitened_spectrum(X, w)
        g_ref = gaussian_reference(lam, np.array([380.0, 1900.0]))
        pr = float(lam.sum() ** 2 / (lam ** 2).sum())
        print(f"\n{tag:>9}: whitened participation ratio {pr:.1f}; Gaussian "
              f"reference FVU {g_ref[0]:.4f} at 380 bits, {g_ref[1]:.4f} "
              f"at 1900")
        e_raw = eta2(X, y, np.ones_like(w))
        e_w = eta2(X, y, w)
        e_s = eta2(X, ys, np.ones_like(w))
        wi, bt = within_between(X, y)
        acc = ridge_probe(X[tr].astype(np.float64), y[tr],
                          X[te].astype(np.float64), y[te], len(names), 1e-2)
        mu2 = float((X.mean(0).astype(np.float64) ** 2).sum())
        struct[tag] = {"eta2_lang": e_raw, "eta2_lang_w": e_w,
                       "eta2_script": e_s, "cos_within": wi,
                       "cos_between": bt, "mean_norm2": mu2,
                       "probe_acc": acc,
                       "participation_ratio": pr,
                       "gauss_ref_380": float(g_ref[0]),
                       "gauss_ref_1900": float(g_ref[1])}
        print(f"{tag:>9}: eta2 lang {e_raw:.4f} (whitened {e_w:.4f}), "
              f"script {e_s:.4f}; cos within {wi:.4f} between {bt:.4f} "
              f"(floor |mean|^2 {mu2:.4f}); 86-way ridge probe {acc:.3f}")
    summary["structure"] = struct

    # per-language probe accuracy needs predictions; refit once per version
    def probe_pred(X: Float[np.ndarray, "n d"]) -> Int[np.ndarray, "n_te"]:
        F = np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1).astype(np.float64)
        Y = np.zeros((tr.sum(), len(names)))
        Y[np.arange(tr.sum()), y[tr]] = 1.0
        W = np.linalg.solve(F[tr].T @ F[tr] + 1e-2 * np.eye(F.shape[1]),
                            F[tr].T @ Y)
        return (F[te] @ W).argmax(1)
    pc, pi = probe_pred(X_c), probe_pred(X_int)
    for tag, pred in (("cached", pc), ("intended", pi)):
        C = np.zeros((len(names), len(names)))
        np.add.at(C, (y[te], pred), 1.0)
        C /= C.sum(1, keepdims=True)
        np.fill_diagonal(C, 0.0)
        top = np.dstack(np.unravel_index(np.argsort(-C, None)[:8], C.shape))[0]
        struct[tag]["top_confusions"] = [
            [names[a], names[b], round(float(C[a, b]), 3)] for a, b in top]
        print(f"{tag:>9} probe's top confusions (true -> predicted, share of "
              f"true): " + ", ".join(f"{names[a]}->{names[b]} {C[a, b]:.2f}"
                                     for a, b in top))
    for j, l in enumerate(names):
        s = y == j
        st = y[te] == j
        lang_rows.append({
            "lang": l, "intended": MC4_TO_SONAR[l], "cached": cached_tag(l),
            "rows": int(s.sum()), "cos_median": round(float(np.median(cos[s])), 5),
            "fvu": round(wfvu(X_int[s], X_c[s], w, base), 5),
            "probe_cached": round(float((pc[st] == j).mean()), 3),
            "probe_intended": round(float((pi[st] == j).mean()), 3)})

    # ---- models
    models = {}
    tail = np.asarray(mm[-cfg.tail_rows:], np.float32)
    base_tail = float((tail.var(0) * w).mean())
    Xr = random_shift(X_c, D, rng)
    for spec in cfg.onto:
        ckpt, T = spec.rsplit(":", 1)
        f, model, step = onto_runner(ckpt, float(T))
        l, h, k = model.l, model.h, model.k
        Yc, Ac = run_batched(f, X_c, cfg.eval_b)
        Yi, Ai = run_batched(f, X_int, cfg.eval_b)
        _, Ar = run_batched(f, Xr, cfg.eval_b)
        Yt, _ = run_batched(f, tail, cfg.eval_b)
        Ac, Ai, Ar = (a.reshape(len(a), l, h) for a in (Ac, Ai, Ar))
        ec, ei = row_err(Yc, X_c, w), row_err(Yi, X_int, w)
        nb_c, _, nul_c = nmi_best(Ac.reshape(len(Ac), -1), y, k, len(names), rng)
        nb_i, _, nul_i = nmi_best(Ai.reshape(len(Ai), -1), y, k, len(names), rng)
        r = {"step": int(step), "l": l, "h": h, "k": k, "T": float(T),
             "fvu_tail": float(row_err(Yt, tail, w).mean() / base_tail),
             "fvu_cached": float(ec.mean() / base),
             "fvu_intended": float(ei.mean() / base),
             "fvu_changed_cached": float(ec[changed].mean() / base),
             "fvu_changed_intended": float(ei[changed].mean() / base),
             "agree": head_agreement(Ac[changed], Ai[changed]).tolist(),
             "agree_random": head_agreement(Ac[changed], Ar[changed]).tolist(),
             "agree_chance": chance_agreement(Ac, k).tolist(),
             "agree_unchanged": head_agreement(Ac[~changed], Ai[~changed]).tolist(),
             "nmi_best_cached": nb_c, "nmi_best_intended": nb_i,
             "nmi_null_cached": nul_c, "nmi_null_intended": nul_i}
        models[Path(ckpt).name] = r
        for j, row in enumerate(lang_rows):
            s = y == j
            row[f"fvu_{Path(ckpt).name}_cached"] = round(
                float(ec[s].mean() / base), 5)
            row[f"fvu_{Path(ckpt).name}_intended"] = round(
                float(ei[s].mean() / base), 5)
        print(f"\n{Path(ckpt).name} (step {step}, {l}x{h}, k={k}): FVU tail "
              f"{r['fvu_tail']:.4f}; sample cached {r['fvu_cached']:.4f} "
              f"intended {r['fvu_intended']:.4f} (changed rows "
              f"{r['fvu_changed_cached']:.4f} -> {r['fvu_changed_intended']:.4f})")
        print("  head agreement on changed rows, per layer: "
              + " ".join(f"{a:.3f}" for a in r["agree"]))
        print("  same-size random shift:                   "
              + " ".join(f"{a:.3f}" for a in r["agree_random"]))
        print("  chance (two unrelated rows):               "
              + " ".join(f"{a:.3f}" for a in r["agree_chance"]))
        print(f"  best-head language NMI: cached {nb_c:.3f} (null {nul_c:.3f})"
              f"  intended {nb_i:.3f} (null {nul_i:.3f})")

    import jax
    import jax.numpy as jnp
    import sae
    from pareto import parse_sae_name
    for run in cfg.sae:
        if (Path(run) / "meta.json").exists():
            meta = json.loads((Path(run) / "meta.json").read_text())
        else:  # pre-meta.json run: the encode rule is in the name
            m_, k_ = parse_sae_name(Path(run) / "params.npz")
            meta = {"m": m_, "topk": k_}
        params = {k_: jnp.asarray(v)
                  for k_, v in np.load(Path(run) / "params.npz").items()}
        topk, groups, gfn = meta["topk"], meta.get("groups", 0), meta.get(
            "group_fn", "top1")

        @jax.jit
        def g(p, X):
            z = sae.encode(p, X, topk, groups, gfn)
            return sae.decode(p, z), z > 0

        def f(X, g=g, params=params):
            return g(params, jnp.asarray(X))
        Yc, Zc = run_batched(f, X_c, cfg.eval_b)
        Yi, Zi = run_batched(f, X_int, cfg.eval_b)
        Yt, _ = run_batched(f, tail, cfg.eval_b)
        ec, ei = row_err(Yc, X_c, w), row_err(Yi, X_int, w)
        inter = (Zc & Zi)[changed].sum(1)
        union = (Zc | Zi)[changed].sum(1)
        r = {"topk": topk, "m": meta["m"],
             "fvu_tail": float(row_err(Yt, tail, w).mean() / base_tail),
             "fvu_cached": float(ec.mean() / base),
             "fvu_intended": float(ei.mean() / base),
             "fvu_changed_cached": float(ec[changed].mean() / base),
             "fvu_changed_intended": float(ei[changed].mean() / base),
             "support_jaccard": float(np.median(inter / np.maximum(union, 1)))}
        models[Path(run).name] = r
        print(f"\nsae {Path(run).name}: FVU tail {r['fvu_tail']:.4f}; sample "
              f"cached {r['fvu_cached']:.4f} intended {r['fvu_intended']:.4f}; "
              f"median support Jaccard on changed rows {r['support_jaccard']:.3f}")
    summary["models"] = models

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "langs.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(lang_rows[0]))
        wr.writeheader()
        wr.writerows(lang_rows)
    print(f"\n-> {out / 'summary.json'}, {out / 'langs.csv'}")


def main() -> None:
    cfg = parse_args()
    os.chdir(ROOT)
    {"texts": texts, "encode": encode, "compare": compare}[cfg.stage](cfg)


if __name__ == "__main__":
    main()

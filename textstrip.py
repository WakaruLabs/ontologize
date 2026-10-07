"""Text strips: what a sentence's decode loses when one head is removed.

The SONAR counterpart of the reconstruction strips and per-feature
ablation images the lineage repos drew for MNIST (SparseEncoders
`imgbatch`/`featimg`). For a few held-out cache rows, the embedding is
decoded under a set of conditions through the SONAR decoder
(`textfid.SonarDecoder`: forced eng_Latn, greedy, every embedding
rescaled to the reference norm, so only directions are compared), and
each condition is scored two ways against a reference decode:

  chrF   between the condition's decode and the reference decode: what
         the strip shows. Greedy decoding over 48 tokens flips on
         rounding-level input differences (at the last layer, where the
         live and frozen ablations below are one embedding up to float32
         rounding, up to a third of the pairs decode differently), so
         chrF is a display measure, not a comparison one.
  dNLL   the decoder's mean per-token NLL of the reference decode under
         the condition's embedding, less its NLL under the reference's
         own embedding (textfid's likelihood measure): smooth in the
         embedding, 0 for no change. Comparisons use this.

Per sentence x, with x_hat_l the decode of the residual after l layers
and x_hat = x_hat_L the model's reconstruction:

  condition    embedding decoded                            scored against
  ref          x                                            --
  recon        x_hat                                        decode(x)
  prefix l     x_hat_l, l = 1..L-1 (the SONAR runs train    decode(x)
               every prefix: deepsup)

and for the --heads-per-layer heads (l, i) of each layer whose removal
moves x_hat furthest (the whitened norm of the unif_frozen delta; a top
over all layers, --heads-per-row, is nearly always layer 0's):

  unif         head i of layer l uniform-ablated (P_i = 1/k)  decode(x_hat)
               in the live forward: later layers re-read the
               changed residual and re-classify
  unif_frozen  x_hat - phi_li + phi_li^unif: the same          decode(x_hat)
               ablation with every other head and layer held
  zero         head i zero-ablated (P_i = 0) in the live       decode(x_hat)
               forward. The dictionary is non-negative, so
               this also removes the head's share of the
               offset every head carries; uniform ablation
               keeps it, which is why it is the primary one
  null         x_hat plus a random step as long (whitened) as  decode(x_hat)
               unif_frozen's delta, along the difference of
               two random cache-tail rows: a data-matched
               direction, not an isotropic one
  only         decode(0) + phi_li: the head's write alone      decode(x)
  only_mean    decode(0) + the mean of phi_li over --mean-rows  decode(x)
               tail rows: the head's average write, the null
               for `only`

phi_li is head i's contribution: what the decoder emits for its slice of
layer l's output alone, gain applied, less decode(0)
(experiments/ste-arm/headcontrib.py), so the contributions sum to x_hat;
phi_li^unif is the same with that head's assignments uniform. unif minus
unif_frozen is the downstream reclassification, the split compose.py
makes. The script prints how closely its probes reproduce the model's
own forward, the contributions sum to x_hat, and the live and frozen
ablations agree in the last layer.

Cost is set by decoding, ~rows * (L + 1 + 5 * heads per row) generations
plus one per distinct head, and one scoring pass per condition, so this
is strip mode: a handful of sentences, not an aggregate over heads. The
summary also gives the removed contribution's size relative to x_hat
(both whitened): a frozen ablation that damages the text no more than a
null step of its length says the damage is set by the step's size, not
by what the head wrote.

  uv run python textstrip.py --ckpt data/out/sonar/multilingual/resid_nc \\
      --device cuda
  uv run python textstrip.py --ckpt data/out/sonar/multilingual/ste_h76_init01 \\
      --heads 0:12 2:5 --rows 8 --device cuda

--heads L:I ... shows those heads on every row instead of the top ones.
The temperature defaults to the run's schedule at the restored step.
Writes <out>/strips.jsonl (every decode and score), strips.html (one
table per sentence, cells shaded by dNLL), meta.json (the settings) and
summary.json; --replot rebuilds the HTML and summary from strips.jsonl
and meta.json. NOTE: Ontologizer
checkpoints restore on GPU JAX only.
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import html
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from jaxtyping import Float, Int

NONE, UNIF, ZERO = 0, 1, 2

# the reference decode each condition is scored against
AGAINST = {"recon": "x", "prefix": "x", "only": "x", "only_mean": "x",
           "unif": "recon", "unif_frozen": "recon", "zero": "recon",
           "null": "recon"}
HEAD_CONDS = ("unif", "unif_frozen", "zero", "null", "only", "only_mean")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="Ontologizer checkpoint dir")
    p.add_argument("--step", type=int, default=0, help="default: latest")
    p.add_argument("--temperature", type=float, default=None,
                   help="default: the run's schedule at --step")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="held-out tail rows of the cache (sae.py's default)")
    p.add_argument("--rows", type=int, default=12,
                   help="sentences, drawn at random from the tail")
    p.add_argument("--row", type=int, nargs="+", default=None,
                   help="absolute cache rows to show instead")
    p.add_argument("--heads-per-layer", type=int, default=2,
                   help="heads shown per layer and row, those whose "
                   "removal moves x_hat furthest")
    p.add_argument("--heads-per-row", type=int, default=0,
                   help="if > 0, the top heads over all layers instead "
                   "(these are nearly always layer 0's, the largest)")
    p.add_argument("--heads", nargs="+", default=None, metavar="L:I",
                   help="show these heads on every row instead")
    p.add_argument("--mean-rows", type=int, default=1024,
                   help="tail rows averaged for each head's mean write")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--b", type=int, default=64, help="jax batch")
    p.add_argument("--b-decode", type=int, default=16)
    p.add_argument("--device", default="cpu",
                   help="torch device for the decoder (cpu protects training)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="default <ckpt>/textstrip")
    p.add_argument("--replot", action="store_true",
                   help="rebuild strips.html and summary.json from "
                   "<out>/strips.jsonl without the model")
    return p.parse_args()


# ---------- pure helpers (unit-tested) ----------

def ablate(P: Float[np.ndarray, "b h k"], layer: int,
           abl_layer: Int[np.ndarray, "b"], abl_head: Int[np.ndarray, "b"],
           abl_mode: Int[np.ndarray, "b"]) -> Float[np.ndarray, "b h k"]:
    """`P` with each row's ablation applied where it targets `layer`: head
    `abl_head[n]` set to 1/k (UNIF) or to 0 (ZERO). Rows whose ablation
    targets another layer, or whose mode is NONE, pass through. Shapes are
    static, so one compile serves every (layer, head, mode) in a batch."""
    import jax.numpy as jnp
    h, k = P.shape[-2:]
    hit = ((abl_layer == layer)[:, None]
           & (jnp.arange(h)[None, :] == abl_head[:, None]))
    P = jnp.where((hit & (abl_mode == UNIF)[:, None])[..., None],
                  jnp.asarray(1.0 / k, P.dtype), P)
    return jnp.where((hit & (abl_mode == ZERO)[:, None])[..., None],
                     jnp.zeros((), P.dtype), P)


def whitened_norm(D: Float[np.ndarray, "... d"], w: Float[np.ndarray, "d"]
                  ) -> Float[np.ndarray, "..."]:
    """Norm under the objective's inverse-variance metric."""
    return np.sqrt((np.asarray(D) ** 2 * w).sum(-1))


def top_heads(norms: Float[np.ndarray, "l h"], m: int) -> list:
    """The m (layer, head) pairs with the largest norms, largest first;
    ties keep (layer, head) order."""
    flat = np.argsort(-norms.reshape(-1), kind="stable")[:m]
    return [tuple(int(v) for v in np.unravel_index(f, norms.shape))
            for f in flat]


def choose_heads(norms: Float[np.ndarray, "l h"], per_layer: int,
                 per_row: int = 0) -> list:
    """The heads one row shows: the `per_row` largest over all layers when
    set, else the `per_layer` largest within each layer, layer by layer.
    Layer 0's contributions dwarf the rest, so a global top is all layer 0."""
    if per_row:
        return top_heads(norms, per_row)
    return [(L, int(i)) for L in range(norms.shape[0])
            for i in np.argsort(-norms[L], kind="stable")[:per_layer]]


def parse_heads(specs: Sequence[str], l: int, h: int) -> list:
    """'L:I' strings -> [(layer, head)], range-checked."""
    out = []
    for s in specs:
        a, b = (int(v) for v in s.split(":"))
        assert 0 <= a < l and 0 <= b < h, f"head {s} outside {l} x {h}"
        out.append((a, b))
    return out


def null_steps(A: Float[np.ndarray, "m d"], B: Float[np.ndarray, "m d"],
               norms: Float[np.ndarray, "m"], w: Float[np.ndarray, "d"]
               ) -> Float[np.ndarray, "m d"]:
    """Steps along A - B (differences of random data rows, a direction
    with the data's covariance) rescaled to whitened length `norms`."""
    U = np.asarray(A, np.float64) - np.asarray(B, np.float64)
    U /= np.maximum(whitened_norm(U, w), 1e-12)[:, None]
    return (U * np.asarray(norms)[:, None]).astype(np.float32)


METRICS = ("chrf", "dnll")
# the sign of a difference that means more damage, per metric
WORSE = {"chrf": -1.0, "dnll": 1.0}


def summarize(records: Sequence[dict]) -> dict:
    """Mean chrF and dNLL (each against the condition's own reference) per
    condition, with the paired comparisons the strips exist for."""
    by = {}
    for r in records:
        for c in r["cells"]:
            by.setdefault(c["cond"], []).append(c)

    def stat(v):
        v = np.asarray([x for x in v if x is not None], float)
        return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None,
                "se": float(v.std(ddof=1) / np.sqrt(len(v)))
                if len(v) > 1 else None}

    s = {"conditions": {c: {m: stat([x.get(m) for x in v]) for m in METRICS}
                        for c, v in by.items()}}

    def paired(a, b):
        """a minus b over the (row, head) cells both have, per metric;
        `frac_worse` is the share where a is the more damaged."""
        key = lambda c: (c["row"], c["layer"], c["head"])
        out = {}
        for m in METRICS:
            cb = {key(c): c.get(m) for c in by.get(b, [])}
            d = np.asarray([c[m] - cb[key(c)] for c in by.get(a, [])
                            if c.get(m) is not None
                            and cb.get(key(c)) is not None], float)
            out[m] = None if not len(d) else {
                "n": int(len(d)), "mean_diff": float(d.mean()),
                "frac_worse": float((WORSE[m] * d > 0).mean()),
                "frac_equal": float((d == 0).mean())}
        return None if all(v is None for v in out.values()) else out

    s["unif_frozen_vs_null"] = paired("unif_frozen", "null")
    s["unif_vs_unif_frozen"] = paired("unif", "unif_frozen")
    s["only_vs_only_mean"] = paired("only", "only_mean")
    for cond in ("unif", "unif_frozen", "null"):
        s[f"{cond}_by_layer"] = {
            str(L): {m: stat([c.get(m) for c in by.get(cond, [])
                              if c["layer"] == L]) for m in METRICS}
            for L in sorted({c["layer"] for c in by.get(cond, [])})}
    # how big the removed contributions are next to the reconstruction
    xn = {r["row"]: r.get("xnorm") for r in records}
    ratio = [c["dnorm"] / xn[c["row"]] for c in by.get("unif_frozen", [])
             if xn.get(c["row"])]
    s["delta_over_xhat_median"] = float(np.median(ratio)) if ratio else None
    return s


DNLL_TOP = 2.0   # nats per token: full shade in the HTML


def _shade(v: Optional[float]) -> str:
    """Cell background: white at dNLL 0 (the reference decode is as likely
    as under its own embedding), deeper red up to DNLL_TOP, on a square-root
    scale so ablations (hundredths of a nat) stay visible beside a head's
    write alone (nats)."""
    if v is None:
        return "#ffffff"
    a = float(np.sqrt(np.clip(v / DNLL_TOP, 0.0, 1.0)))
    mix = lambda c: int(round(255 + (c - 255) * a))
    return f"#{mix(0xe8):02x}{mix(0x7a):02x}{mix(0x7a):02x}"


def _fmt(v: Optional[float], spec: str) -> str:
    return "" if v is None else format(v, spec)


def render_html(records: Sequence[dict], title: str, summary: dict) -> str:
    """One section per sentence: its reference and reconstruction decodes
    and prefixes, then a head x condition table. Every decode is escaped."""
    e = html.escape

    def cell(c):
        if c is None:
            return "<td></td>"
        return (f'<td style="background:{_shade(c.get("dnll"))}">'
                f'<div class="t">{e(c["text"])}</div>'
                f'<div class="s">dNLL {_fmt(c.get("dnll"), "+.2f")} &middot; '
                f'chrF {c["chrf"]:.2f}</div></td>')

    out = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{e(title)}</title><style>",
        "body{font:14px/1.4 system-ui,sans-serif;margin:16px;color:#222;"
        "background:#fff}",
        "table{border-collapse:collapse;margin:6px 0 18px;width:100%}",
        "td,th{border:1px solid #ddd;padding:4px 6px;vertical-align:top;"
        "text-align:left}",
        "th{background:#f4f4f4;font-weight:600}",
        ".t{white-space:pre-wrap}.s{font:11px monospace;color:#333}",
        ".k{font:12px monospace;white-space:nowrap}",
        "h2{font-size:15px;margin:22px 0 4px}",
        "</style></head><body>",
        f"<h1 style='font-size:18px'>{e(title)}</h1>",
        "<p>Each cell is a decode, scored against its reference decode: "
        "ablations and the null against decode(x&#770;); recon, prefixes, "
        "only and only_mean against decode(x). dNLL is the decoder's "
        "per-token NLL of the reference decode under the cell's embedding, "
        f"less under the reference's own; cells are shaded by it on a "
        f"square-root scale, white at 0 and full red at {DNLL_TOP:g} nats per "
        f"token. chrF compares the two texts "
        "and flips on rounding-level changes, so read it as a display, "
        "not a score.</p>",
        "<table><tr><th>condition</th><th>n</th><th>mean dNLL</th>"
        "<th>SE</th><th>mean chrF</th><th>SE</th></tr>"]
    for c, v in summary["conditions"].items():
        d, f = v["dnll"], v["chrf"]
        out.append(f"<tr><td class='k'>{e(c)}</td><td>{f['n']}</td>"
                   f"<td>{_fmt(d['mean'], '+.3f')}</td>"
                   f"<td>{_fmt(d['se'], '.3f')}</td>"
                   f"<td>{_fmt(f['mean'], '.3f')}</td>"
                   f"<td>{_fmt(f['se'], '.3f')}</td></tr>")
    out.append("</table>")
    for r in records:
        cells = r["cells"]
        lang = f" ({e(r['lang'])})" if r.get("lang") else ""
        out.append(f"<h2>row {r['row']}{lang}</h2>")
        out.append(f"<p><b>decode(x):</b> {e(r['ref'])}</p>")
        out.append("<table><tr><th>prefix</th><th>decode</th></tr>")
        for c in cells:
            if c["cond"] in ("prefix", "recon"):
                name = ("recon" if c["cond"] == "recon"
                        else f"x&#770;<sub>{c['layer'] + 1}</sub>")
                out.append(f"<tr><td class='k'>{name}</td>{cell(c)}</tr>")
        out.append("</table>")
        heads = []
        for c in cells:
            if c["cond"] == "unif_frozen":
                heads.append((c["layer"], c["head"], c["dnorm"]))
        if not heads:
            continue
        out.append("<table><tr><th>head</th>" + "".join(
            f"<th>{x}</th>" for x in HEAD_CONDS) + "</tr>")
        get = {(c["cond"], c["layer"], c["head"]): c for c in cells
               if c["cond"] in HEAD_CONDS}
        for L, i, dn in heads:
            out.append(f"<tr><td class='k'>L{L}.h{i}<br>"
                       f"<span class='s'>&Delta;<sub>w</sub> {dn:.3f}</span></td>"
                       + "".join(cell(get.get((x, L, i))) for x in HEAD_CONDS)
                       + "</tr>")
        out.append("</table>")
    out.append("</body></html>")
    return "\n".join(out)


# ---------- model probes ----------

def forward_probe(module, X, abl_layer, abl_head, abl_mode, T: float):
    """Decoded prefixes (l, b, d_out) of the forward with each row's
    ablation applied (`ablate`); the last is x_hat. Mirrors textfid's
    probe: `head_outputs` applies the router gain and fibers."""
    import jax.numpy as jnp
    E, _ = module.encode(X, 0.0, None)
    R = module.resid(E)
    Ein = module.constinput(E)
    Ys = []
    for i, de in enumerate(module.dictencs):
        U, G = de.gainshape_in(Ein)
        P0 = de.dict.cluster(de.classifier(U), T)
        P = ablate(P0, i, abl_layer, abl_head, abl_mode)
        R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
        Ys.append(module.decode(R))
        if i < module.l - 1:
            # the unablated classification, as `withArgs` forwards it
            # (only `resid_labels` reads it)
            Ein = module.nextinput(X, R, P0.reshape(X.shape[0], -1))
    return jnp.stack(Ys)


def contrib_probe(module, X, T: float):
    """Per-head output contributions under the plain forward, and under
    that head's own uniform ablation with the layer input held: each is
    what the decoder emits for the head's slice of its layer's output
    alone, gain applied, less decode(0). Returns (phi, phi_unif), both
    (b, l, h, d_out), and decode(0)."""
    import jax.numpy as jnp
    E, _ = module.encode(X, 0.0, None)
    R = module.resid(E)
    Ein = module.constinput(E)
    zero = module.decode(jnp.zeros_like(R[:1]))[0]
    phi, phi_u = [], []
    for i, de in enumerate(module.dictencs):
        U, G = de.gainshape_in(Ein)
        P = de.dict.cluster(de.classifier(U), T)
        Y = de.head_outputs(U, P)                               # (b, h, e)
        Yu = de.head_outputs(U, jnp.full_like(P, 1.0 / P.shape[-1]))
        # one copy of the layer output per head, every other head zeroed,
        # so `combine` places it as the layer itself would
        eye = jnp.eye(Y.shape[1], dtype=Y.dtype)[None, :, :, None]
        Gh = None if G is None else G[:, None]
        for Z, acc in ((Y, phi), (Yu, phi_u)):
            C = de.gained(de.dict.combine(Z[:, None] * eye), Gh)
            acc.append(module.decode(C) - zero)
        R = R + de.gained(de.dict.combine(Y), G)
        if i < module.l - 1:
            Ein = module.nextinput(X, R, P.reshape(X.shape[0], -1))
    return jnp.stack(phi, 1), jnp.stack(phi_u, 1), zero


# ---------- driver ----------

def write_outputs(out: Path, records: list, title: str, meta: dict):
    summary = {**meta, **summarize(records)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "strips.html").write_text(render_html(records, title, summary))
    s = summary
    pm = lambda v, f: ("   -   " if v["mean"] is None else
                       format(v["mean"], f) + ("" if v["se"] is None
                                               else f" +- {v['se']:.3f}"))
    print("\ncondition      n   dNLL             chrF             against")
    for c, v in s["conditions"].items():
        print(f"  {c:<12} {v['chrf']['n']:>4}  {pm(v['dnll'], '+.3f'):<16} "
              f"{pm(v['chrf'], '.3f'):<16} "
              f"{'decode(x)' if AGAINST[c] == 'x' else 'decode(x_hat)'}")
    for key, txt in (("unif_frozen_vs_null", "frozen ablation vs null"),
                     ("unif_vs_unif_frozen", "live vs frozen ablation"),
                     ("only_vs_only_mean", "head's write vs its mean")):
        v = s[key]
        for m in METRICS if v else ():
            if v[m]:
                print(f"  {txt}, {m}: mean diff {v[m]['mean_diff']:+.3f}, "
                      f"first worse in {v[m]['frac_worse']:.0%} of "
                      f"{v[m]['n']} cells")
    if s["delta_over_xhat_median"] is not None:
        print(f"  removed contribution / x_hat (whitened), median "
              f"{s['delta_over_xhat_median']:.3f}")
    print("  dNLL by layer:   unif     unif_frozen   null")
    for L in s["unif_frozen_by_layer"]:
        g = lambda c: s[f"{c}_by_layer"].get(L, {}).get("dnll", {}).get("mean")
        print(f"    layer {L}:    " + "     ".join(
            "  -   " if g(c) is None else f"{g(c):+.3f}"
            for c in ("unif", "unif_frozen", "null")))
    print(f"-> {out / 'strips.html'}\n-> {out / 'strips.jsonl'}")


def main():
    cfg = parse_args()
    assert cfg.out or cfg.ckpt, "--ckpt (or --out with --replot) is required"
    out = Path(cfg.out) if cfg.out else Path(cfg.ckpt) / "textstrip"
    if cfg.replot:
        records = [json.loads(s) for s in
                   (out / "strips.jsonl").read_text().splitlines()]
        meta = json.loads((out / "meta.json").read_text())
        write_outputs(out, records, title_of(meta), meta)
        return

    import jax
    import jax.numpy as jnp
    from assignmap import read_hyper, temperature_at
    from pareto import load_onto
    from textfid import SonarDecoder, chrf

    model, params, step = load_onto(cfg.ckpt, cfg.step)
    l, h = model.l, model.h
    T = cfg.temperature
    if T is None:
        hyper = read_hyper(cfg.ckpt)
        assert hyper is not None, \
            "no env_config in log.jsonl; pass --temperature"
        T = temperature_at(hyper, step)
    w = np.load(cfg.mse_weights).astype(np.float64)
    meta = {"run": Path(cfg.ckpt).name, "step": int(step),
            "temperature": float(T), "seed": cfg.seed}
    print(f"{meta['run']} step {step}: {l} layers x {h} heads x {model.k} "
          f"entries, T={T:g}")

    mm = np.load(cfg.cache, mmap_mode="r")
    n_all = mm.shape[0]
    lo = n_all - cfg.eval_rows
    rng = np.random.default_rng(cfg.seed)
    rows = (np.asarray(cfg.row) if cfg.row else
            lo + np.sort(rng.choice(cfg.eval_rows, cfg.rows, replace=False)))
    X = np.asarray(mm[rows], dtype=np.float32)
    assert X.shape[1] == len(w), "mse weights do not match the cache width"
    side = Path(cfg.cache).with_name(
        Path(cfg.cache).name.replace(".npy", ".langs.npy"))
    langs = (np.asarray(np.load(side, mmap_mode="r")[rows]).astype(str)
             if side.exists() else [None] * len(rows))

    # params are an argument, not a closure: closed-over weights become
    # compile-time constants that XLA tries to fold
    p = {"params": params}
    fwd_j = jax.jit(lambda q, x, a, b, c: model.apply(
        q, x, a, b, c, T, method=forward_probe))
    con_j = jax.jit(lambda q, x: model.apply(q, x, T, method=contrib_probe))
    con = lambda x: con_j(p, x)

    def forward(Xb, al, ah, am):
        return np.concatenate(
            [np.asarray(fwd_j(p, *(jnp.asarray(v[i:i + cfg.b])
                                   for v in (Xb, al, ah, am))))
             for i in range(0, len(Xb), cfg.b)], 1)

    none = np.zeros(len(rows), np.int32)
    prefixes = forward(X, none, none, none)               # (l, n, d)
    x_hat = prefixes[-1]
    phi, phi_u, zero = (np.asarray(v) for v in con(jnp.asarray(X)))

    # the probes against the model's own forward, and the contributions
    # against x_hat: both should agree to float32 rounding
    own = np.asarray(model.apply(p, jnp.asarray(X), temperature=T))
    scale = np.abs(x_hat).max()
    print(f"probe vs model forward: max |diff| {np.abs(own - x_hat).max():.1e}; "
          f"contributions vs x_hat: "
          f"{np.abs(zero + phi.sum((1, 2)) - x_hat).max():.1e} "
          f"(|x_hat| max {scale:.2f})")

    # each head's mean write over the tail
    mean_rows = lo + np.sort(rng.choice(cfg.eval_rows, cfg.mean_rows,
                                        replace=False))
    acc = np.zeros(phi.shape[1:], np.float64)
    for i in range(0, len(mean_rows), cfg.b):
        Xm = np.asarray(mm[mean_rows[i:i + cfg.b]], dtype=np.float32)
        acc += np.asarray(con(jnp.asarray(Xm))[0]).sum(0)
    phi_bar = (acc / len(mean_rows)).astype(np.float32)

    delta = phi_u - phi                                   # (n, l, h, d)
    dnorm = whitened_norm(delta, w)                       # (n, l, h)
    forced = parse_heads(cfg.heads, l, h) if cfg.heads else None
    chosen = [forced or choose_heads(dnorm[n], cfg.heads_per_layer,
                                     cfg.heads_per_row)
              for n in range(len(rows))]
    cells = [(n, L, i) for n in range(len(rows)) for L, i in chosen[n]]
    print(f"{len(rows)} rows x {len(chosen[0])} heads: "
          f"{len(cells)} (row, head) cells")

    # live ablations
    cn, cl, ch = (np.asarray(v, np.int32) for v in zip(*cells))
    live = {}
    for mode, name in ((UNIF, "unif"), (ZERO, "zero")):
        live[name] = forward(X[cn], cl, ch, np.full(len(cells), mode,
                                                    np.int32))[-1]
    frozen = x_hat[cn] + delta[cn, cl, ch]
    tail_pool = np.setdiff1d(np.arange(lo, n_all), rows)
    ab = np.stack([rng.choice(tail_pool, 2, replace=False) for _ in cells])
    A, B = (np.asarray(mm[ab[:, j]], np.float32) for j in (0, 1))
    null = x_hat[cn] + null_steps(A, B, dnorm[cn, cl, ch], w)
    only = zero + phi[cn, cl, ch]
    heads_u = sorted(set(zip(cl.tolist(), ch.tolist())))
    only_mean = zero + np.stack([phi_bar[L, i] for L, i in heads_u])

    last = cl == l - 1
    if last.any():
        print(f"last-layer live vs frozen ablation: max |diff| "
              f"{np.abs(live['unif'][last] - frozen[last]).max():.1e} "
              f"(the same embedding up to rounding)")

    # one generation pass over everything
    nr = len(rows)
    blocks = {"ref": X, "recon": x_hat,
              "prefix": prefixes[:-1].reshape(-1, X.shape[1]),
              "unif": live["unif"], "unif_frozen": frozen,
              "zero": live["zero"], "null": null, "only": only,
              "only_mean": only_mean}
    # each block decodes and scores in batches of its own: a greedy decode
    # depends on its batch partners (kernel choice by batch shape), so
    # mixing blocks would let the head selection change the references
    dec = SonarDecoder(cfg.device, cfg.b_decode)
    print(f"decoding {sum(len(v) for v in blocks.values())} embeddings "
          f"on {cfg.device}")
    got = {k: dec.generate(v.astype(np.float32)) for k, v in blocks.items()}
    ref_seq, rec_seq = got["ref"], got["recon"]
    hidx = {hd: j for j, hd in enumerate(heads_u)}
    mean_cell = np.stack([only_mean[hidx[(L, i)]] for L, i in zip(cl, ch)])

    # one scoring pass: each condition's reference decode under the
    # condition's embedding, and both references under their own
    against_seq = {"x": lambda n: ref_seq[n], "recon": lambda n: rec_seq[n]}
    jobs = [("floor_x", X, range(nr), "x"),
            ("floor_recon", x_hat, range(nr), "recon"),
            ("recon", x_hat, range(nr), "x"),
            ("prefix", blocks["prefix"], [n for _ in range(l - 1)
                                          for n in range(nr)], "x"),
            ("only_mean", mean_cell, cn, "x")]
    jobs += [(k, blocks[k], cn, AGAINST[k])
             for k in ("unif", "unif_frozen", "zero", "null", "only")]
    print(f"scoring {sum(len(j[1]) for j in jobs)} embeddings")
    nll = {k: dec.nll(Yj.astype(np.float32),
                      [against_seq[ref](n) for n in ns])
           for k, Yj, ns, ref in jobs}
    floor = {"x": nll["floor_x"], "recon": nll["floor_recon"]}

    text = {k: [dec.text(s) for s in v] for k, v in got.items()}
    mean_txt = dict(zip(heads_u, text["only_mean"]))
    records = []
    for n in range(nr):
        ref, rec = text["ref"][n], text["recon"][n]
        against = {"x": ref, "recon": rec}

        def mk(cond, txt, nl, layer=None, head=None, dn=None):
            a = AGAINST[cond]
            return {"row": int(rows[n]), "cond": cond, "layer": layer,
                    "head": head, "text": txt, "dnorm": dn,
                    "chrf": chrf(against[a], txt), "nll": float(nl),
                    "dnll": float(nl - floor[a][n])}

        cs = [mk("prefix", text["prefix"][L * nr + n],
                 nll["prefix"][L * nr + n], layer=L) for L in range(l - 1)]
        cs.append(mk("recon", rec, nll["recon"][n], layer=l - 1))
        for j, (m, L, i) in enumerate(cells):
            if m != n:
                continue
            dn = float(dnorm[n, L, i])
            for cond in ("unif", "unif_frozen", "zero", "null", "only"):
                cs.append(mk(cond, text[cond][j], nll[cond][j], L, i, dn))
            cs.append(mk("only_mean", mean_txt[(L, i)], nll["only_mean"][j],
                         L, i, dn))
        records.append({"row": int(rows[n]), "lang": langs[n], "ref": ref,
                        "xnorm": float(whitened_norm(x_hat[n], w)),
                        "nll_x": float(floor["x"][n]),
                        "nll_recon": float(floor["recon"][n]),
                        "cells": cs})

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "strips.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta |= {"rows": len(rows), "heads_per_layer": cfg.heads_per_layer,
             "heads_per_row": cfg.heads_per_row,
             "mean_rows": cfg.mean_rows, "forced_heads": cfg.heads}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    write_outputs(out, records, title_of(meta), meta)


def title_of(meta: dict) -> str:
    return (f"text strips: {meta['run']} step {meta['step']}, "
            f"t={meta['temperature']:g}")


if __name__ == "__main__":
    main()

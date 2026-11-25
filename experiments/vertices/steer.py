"""Whether a head forced to its cell's trimmed extreme steers better than
one forced to its vertex, or to its cell's centroid, read through M2M100.

`place.py` found the softmax model's vertices far outside their cells, and
its cells' trimmed extremes in distribution. This asks whether text shows
the difference. For each layer, each of --queries held-out rows has one
head, drawn at random, forced to another entry j, drawn at random from
the head's entries with --min-rows rows. The force is frozen, as in
`place.py`: the row keeps every other head's write, and nothing
downstream reclassifies. In place of its own write the head writes

  vertex    entry j's decoded atom, as the model writes it
  extreme   the trimmed extreme of j's cell (`place.py`)
  centroid  the cell's least-squares atom, read about the head's means as
            `place.py` reads it

each at its own length. Each target's step, the target less the head's
own write, is also taken at matched lengths, --lengths times the
extreme's step, beside a random direction with the rows' covariance: the
control, collateral without intended effect.

Every row is decoded through M2M100 (`textfid.SonarDecoder`: English
forced, greedy, and rescaled to SONAR's norm, so the decoder sees a row's
direction and not its norm). Each decode is re-encoded by SONAR under the
English tag and scaled to unit norm, as the cache's rows were. Per
decode:

  p           the model's probability of entry j at the forced head, on
              the re-encoding; `hit`, whether j is that head's argmax there
  p_given     the same on the row the decoder was given, scaled to unit
              norm: what the force put in before the round trip
  effect      p less p on the re-encoding of the unforced
              reconstruction's decode, row by row
  collateral  1 - chrF(the reconstruction's decode, this decode)
  keep        the re-encoding's cosine to the source row; `keep0` the same
              against the next query row
  fidelity    the re-encoding's cosine to the row the decoder was given

The cycle's ceiling is `retain`: the share of rows whose own decode's
re-encoding keeps the row's own entry at the forced head.

  JAX_PLATFORMS=cpu uv run python experiments/vertices/steer.py \\
      --ckpt data/out/sonar/multilingual/resid_nc --device cuda \\
      --out data/out/sonar/vertices/steer

Outputs under --out: decodes.jsonl (one line per layer, row and
condition) and summary.json (per layer: each condition's means, paired
contrasts, `retain`, and the steps' lengths), rewritten after each layer.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from jaxtyping import Bool, Float, Int  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from place import (head_write, model_view, place_layer,  # noqa: E402
                   quartiles, run_temperature, shaped_random)
from textfid import chrf  # noqa: E402

TARGETS = ("vertex", "extreme", "centroid")
DIRECTIONS = TARGETS + ("random",)
METRICS = ("p", "hit", "p_given", "effect", "collateral", "keep", "keep0",
           "fidelity")
ENCODER_ID = "cointegrated/SONAR_200_text_encoder"


# ---- pure helpers -----------------------------------------------------------

def unit(V: Float[np.ndarray, "... d"]) -> Float[np.ndarray, "... d"]:
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-12)


def cosines(E: Float[np.ndarray, "n d"], V: Float[np.ndarray, "n d"]
            ) -> Float[np.ndarray, "n"]:
    return (unit(E) * unit(V)).sum(-1)


def cond(direction: str, length: float) -> str:
    """A matched-length condition's name. The extreme's step at length 1 is
    the extreme itself, named `extreme`."""
    if direction == "extreme" and length == 1:
        return "extreme"
    return f"{direction}@{length:g}"


def pick(sel: Int[np.ndarray, "n h"], ok: Bool[np.ndarray, "h k"],
         rng: np.random.Generator
         ) -> Tuple[Int[np.ndarray, "n"], Int[np.ndarray, "n"]]:
    """Per row, a head drawn uniformly from those with an eligible entry
    besides the row's own, and one of those entries, drawn uniformly."""
    n, _ = sel.shape
    k = ok.shape[1]
    heads, ents = np.empty(n, np.int64), np.empty(n, np.int64)
    for r in range(n):
        other = ok & (np.arange(k)[None, :] != sel[r][:, None])
        cand = np.flatnonzero(other.any(1))
        if not len(cand):
            raise ValueError(f"row {r}: no head has an eligible entry "
                             "besides its own")
        i = int(rng.choice(cand))
        heads[r], ents[r] = i, int(rng.choice(np.flatnonzero(other[i])))
    return heads, ents


def forced_rows(Y: Float[np.ndarray, "n d"],
                steps: Dict[str, Float[np.ndarray, "n d"]],
                rand: Float[np.ndarray, "n d"], lengths: Sequence[float]
                ) -> Dict[str, Float[np.ndarray, "n d"]]:
    """The rows the conditions decode: Y plus each target's step at its own
    length, named by the target, and along each step's direction and the
    random one at `lengths` times the extreme's step, named by `cond`."""
    ref = np.linalg.norm(steps["extreme"], axis=1, keepdims=True)
    out = {name: Y + s for name, s in steps.items()}
    for name, s in dict(steps, random=rand).items():
        u = unit(s)
        for L in lengths:
            out.setdefault(cond(name, L), Y + L * ref * u)
    return out


def measures(texts: Sequence[str], E: Float[np.ndarray, "n d"],
             given: Float[np.ndarray, "n d"], X: Float[np.ndarray, "n d"],
             P: Float[np.ndarray, "n k"], P_given: Float[np.ndarray, "n k"],
             j: Int[np.ndarray, "n"], base_texts: Sequence[str],
             p_base: Float[np.ndarray, "n"]
             ) -> Dict[str, Float[np.ndarray, "n"]]:
    """Per decode, from its re-encoding `E` and the head's assignment there
    (`P`) and on the row the decoder was given (`P_given`): the forced
    entry's probability, whether it is the argmax, that probability less
    the unforced reconstruction's (`p_base`), 1 - chrF against the
    reconstruction's decode, and the re-encoding's cosine to its source
    row, to the next row, and to the row the decoder was given."""
    rows = np.arange(len(texts))
    p = P[rows, j]
    return {"p": p, "hit": (P.argmax(1) == j).astype(np.float64),
            "p_given": P_given[rows, j], "effect": p - p_base,
            "collateral": np.array([1.0 - chrf(b, t)
                                    for b, t in zip(base_texts, texts)]),
            "keep": cosines(E, X), "keep0": cosines(E, np.roll(X, -1, 0)),
            "fidelity": cosines(E, given)}


def mean_se(x: Float[np.ndarray, "n"]) -> Dict[str, float]:
    """Mean and standard error over rows, NaNs left out."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"n": int(len(x)),
                "mean": float(x.mean()) if len(x) else float("nan"),
                "se": float("nan")}
    return {"n": int(len(x)), "mean": float(x.mean()),
            "se": float(x.std(ddof=1) / np.sqrt(len(x)))}


def contrast_pairs(lengths: Sequence[float]) -> List[Tuple[str, str]]:
    """The paired differences read: the targets against one another at
    their own lengths, and at each matched length each direction against
    the extreme's and against the random one."""
    pairs = [("extreme", "vertex"), ("extreme", "centroid"),
             ("centroid", "vertex")]
    for L in lengths:
        e, v = cond("extreme", L), cond("vertex", L)
        c, r = cond("centroid", L), cond("random", L)
        pairs += [(e, v), (e, c), (e, r), (v, r), (c, r)]
    return pairs


def summarize(per: Dict[str, Dict[str, Float[np.ndarray, "n"]]],
              lengths: Sequence[float]) -> dict:
    """Each condition's measures as means over rows, and the paired
    differences of `contrast_pairs`, row by row."""
    res = {"conditions": {c: {m: mean_se(v) for m, v in ms.items()}
                          for c, ms in per.items()}}
    res["contrasts"] = {
        f"{a} - {b}": {m: mean_se(per[a][m] - per[b][m]) for m in METRICS}
        for a, b in contrast_pairs(lengths) if a in per and b in per}
    return res


# ---- main -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="data/out/sonar/multilingual/resid_nc")
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--temperature", type=float, default=None,
                   help="default: the run's final temperature (log.jsonl)")
    p.add_argument("--eval-cache",
                   default="data/sonar_embeddings/mc4_fresh.npy")
    p.add_argument("--rows", type=int, default=32768,
                   help="held-out rows from the start of the eval cache, "
                        "which place the cells as place.py's do")
    p.add_argument("--queries", type=int, default=256,
                   help="rows forced and decoded at each layer: the first "
                        "of --rows")
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="default: every layer")
    p.add_argument("--lengths", type=float, nargs="+",
                   default=[0.5, 1.0, 2.0, 4.0, 8.0],
                   help="matched step lengths, in multiples of the "
                        "extreme's step")
    p.add_argument("--min-rows", type=int, default=32,
                   help="entries with fewer rows are not forced")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--batch", type=int, default=256, help="JAX batch")
    p.add_argument("--device", default="cpu",
                   help="torch device (M2M100 and SONAR's encoder)")
    p.add_argument("--b-decode", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    import jax.numpy as jnp
    import torch as t
    from pareto import load_onto
    from steerfid import load_steerable
    from textfid import SonarDecoder
    from ontologize.data.loaders import TokenizeTransform
    from ontologize.data.pretrained import pretrained_transformer

    T = cfg.temperature or run_temperature(cfg.ckpt)
    w_sqrt = np.sqrt(np.load(cfg.mse_weights).astype(np.float64)
                     ).astype(np.float32)
    # forced rows leave the whitened frame by dividing by sqrt(weights)
    if not (w_sqrt > 0).all():
        raise ValueError(f"{cfg.mse_weights}: every weight must be positive")
    X = np.asarray(np.load(cfg.eval_cache, mmap_mode="r")[:cfg.rows],
                   np.float32)
    model, _, _ = load_onto(cfg.ckpt, cfg.step)
    soft = model.select.lower() not in ("ste", "argmax")
    del model
    view = model_view(cfg.ckpt, cfg.step, T, X, w_sqrt, cfg.batch, soft)
    Xw = X * w_sqrt
    var = float(((Xw - Xw.mean(0)) ** 2).sum())
    L_all, h, k, _ = view["A"].shape
    layers = cfg.layers if cfg.layers else list(range(L_all))
    n = cfg.queries
    Q = slice(0, n)
    Xq = X[Q]
    ref = X[n:] - X[n:].mean(0)
    rng = np.random.default_rng(cfg.seed)
    print(f"{Path(cfg.ckpt).name} step {view['step']} T={T:g} "
          f"({'soft' if soft else 'hard'} code); cells from {len(X)} rows, "
          f"{n} forced per layer", flush=True)

    acts, _, _, _, _ = load_steerable(cfg.ckpt, argparse.Namespace(
        mode="dec", step=view["step"], temperature=T))

    def probs(V):
        """(rows, layers, heads, entries): the model's assignments on the
        rows of V scaled to unit norm, in batches of one size."""
        b = cfg.batch
        V = unit(np.asarray(V, np.float32))
        m = -(-len(V) // b) * b
        Vp = np.zeros((m, V.shape[1]), np.float32)
        Vp[:len(V)] = V
        P = np.concatenate([np.asarray(acts(jnp.asarray(Vp[i:i + b])))
                            for i in range(0, m, b)])[:len(V)]
        return P.reshape(len(V), L_all, h, k)

    dec = SonarDecoder(cfg.device, cfg.b_decode)
    dev = t.device(cfg.device)
    enc, tokenizer = pretrained_transformer(ENCODER_ID, "float32", dev=dev)
    tok = TokenizeTransform(tokenizer, maxlen=128)

    def reencode(texts):
        """SONAR's embedding of each text under the English tag, scaled to
        unit norm, pooled in torch: this process's JAX runs on the CPU."""
        E = []
        with t.no_grad():
            for i in range(0, len(texts), cfg.b_decode):
                batch = [tok.map({"text": s, "lang": "en"})
                         for s in texts[i:i + cfg.b_decode]]
                ids, mask = (t.as_tensor(np.stack([b[f] for b in batch])
                                         ).to(dev)
                             for f in ("input_ids", "attention_mask"))
                H = enc(input_ids=ids, attention_mask=mask).last_hidden_state
                w = mask.unsqueeze(-1).to(H.dtype)
                E.append(((H * w).sum(1) / w.sum(1).clamp(min=1e-9)
                          ).cpu().numpy())
        return unit(np.concatenate(E))

    def decode(V):
        texts = dec.texts(V)
        E = reencode(texts)
        return texts, E, probs(E), probs(V)

    Y = view["Y"][Q] / w_sqrt
    base = {"x": Xq, "recon": Y}
    done = {c: decode(V) for c, V in base.items()}

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"ckpt": cfg.ckpt, "step": int(view["step"]), "temperature": T,
           "soft": soft, "eval_cache": cfg.eval_cache, "rows": len(X),
           "queries": n, "min_rows": cfg.min_rows,
           "lengths": list(cfg.lengths), "seed": cfg.seed, "layers": {}}
    log = open(out / "decodes.jsonl", "w")
    rows = np.arange(n)
    for l in layers:
        _, E_l, M_l = place_layer(view, Xw, l, var, cfg.min_rows)
        ok = np.isfinite(E_l).all(-1) & np.isfinite(M_l).all(-1)
        sel = view["sel"][Q, l].astype(np.int64)
        heads, ents = pick(sel, ok, rng)
        cur = sel[rows, heads]
        own = np.stack([head_write(view, l, int(i), slice(r, r + 1))[0]
                        for r, i in enumerate(heads)]) / w_sqrt
        g = view["G"][Q, l].astype(np.float32)[:, None]
        targets = {"vertex": view["A"][l][heads, ents],
                   "extreme": E_l[heads, ents], "centroid": M_l[heads, ents]}
        steps = {name: g * a / w_sqrt - own for name, a in targets.items()}
        conds = forced_rows(Y, steps, shaped_random(ref, n, rng),
                            cfg.lengths)

        def at_head(P):
            return P[rows, l, heads]

        p_base = at_head(done["recon"][2])[rows, ents]
        per = {}
        for c, (texts, E, P, P_given) in done.items():
            per[c] = measures(texts, E, base[c], Xq, at_head(P),
                              at_head(P_given), ents, done["recon"][0],
                              p_base)
        retain = {c: mean_se(at_head(done[c][2]).argmax(1) == cur)
                  for c in done}
        for c, V in conds.items():
            texts, E, P, P_given = decode(V)
            per[c] = measures(texts, E, V, Xq, at_head(P), at_head(P_given),
                              ents, done["recon"][0], p_base)
            for r in rows:
                log.write(json.dumps(
                    {"layer": l, "row": int(r), "head": int(heads[r]),
                     "entry": int(ents[r]), "own_entry": int(cur[r]),
                     "cond": c, "text": texts[r],
                     **{m: round(float(per[c][m][r]), 4) for m in METRICS}},
                    ensure_ascii=False) + "\n")
            log.flush()
        for c in done:
            for r in rows:
                log.write(json.dumps(
                    {"layer": l, "row": int(r), "head": int(heads[r]),
                     "entry": int(ents[r]), "own_entry": int(cur[r]),
                     "cond": c, "text": done[c][0][r],
                     **{m: round(float(per[c][m][r]), 4) for m in METRICS}},
                    ensure_ascii=False) + "\n")
        lens = {name: np.linalg.norm(s, axis=1) for name, s in steps.items()}
        layer = summarize(per, cfg.lengths)
        layer["retain"] = retain
        layer["step_length"] = {name: quartiles(v) for name, v in lens.items()}
        layer["step_over_extreme"] = {
            name: quartiles(v / lens["extreme"]) for name, v in lens.items()}
        res["layers"][l] = layer
        (out / "summary.json").write_text(json.dumps(res, indent=1))

        so = layer["step_over_extreme"]
        print(f"layer {l}: steps over the extreme's, median: centroid "
              f"{so['centroid']['median']:.2f}, vertex "
              f"{so['vertex']['median']:.2f}; retain x "
              f"{retain['x']['mean']:.3f}, recon {retain['recon']['mean']:.3f}",
              flush=True)
        print(f"  {'condition':14s}" + "".join(f"{m:>11s}" for m in METRICS),
              flush=True)
        for c, ms in layer["conditions"].items():
            print(f"  {c:14s}" + "".join(f"{ms[m]['mean']:11.3f}"
                                         for m in METRICS), flush=True)
    log.close()


if __name__ == "__main__":
    main()

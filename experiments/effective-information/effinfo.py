"""Per-head effective information (EI) of a trained Ontologizer: how much
does forcing a head's entry change what the heads downstream of it choose?

The writeup's formal reading (writeup/sections/discussion.tex) treats each
head as a categorical causal variable, `withArgs` set-interventions as the
do-operator, and the residual stack as the causal graph, in which every
earlier head is a parent of every later one. EI (Hoel, Albantakis &
Tononi 2013) weighs those edges. For a source head (l, i) and an effect
head (l', i') with l' > l: intervene do[(l, i) = j] for every entry j --
uniformly, the maximum-entropy intervention distribution EI is defined
by -- on a fixed set of held-out contexts, and read the effect head's
entry. With W_j the effect distribution under do(j), pooled over contexts,

    EI = H(mean_j W_j) - mean_j H(W_j)  =  determinism - degeneracy,
    determinism = log2 k - mean_j H(W_j),
    degeneracy  = log2 k - H(mean_j W_j).

Determinism is low when what entry j does depends on the context (the
input and the rest of the code, which the intervention leaves free);
degeneracy is high when different entries lead to the same effect. EI is
at most log2 k bits (5 at k = 32).

Two properties of the Ontologizer make this exact up to finite contexts.
Heads are categorical, so the intervention distribution is enumerable: k
forward passes per head, with no binning of continuous activations. And
the effect variables are categorical too, so each W_j is a k-bin
histogram over contexts. Every W_j is measured on the same contexts, so
a pair the intervention never changes has identical W_j for every j and
EI exactly 0: there is no finite-sample floor under a null effect, and
any positive EI means some context's effect choice depends on j. The
plug-in bias applies only to the part of the effect that does change.
Heads that are not descendants of the source are such pairs by
construction; only descendant pairs are computed.

The interventions follow `Ontologizer.withArgs` exactly: the forced entry
enters the source layer's lookup (`DictBlock.hfwd`, through
`DictEnc.withStats`), so downstream layers classify the intervened
residual, and every layer's assignments are the model's own
classification at the run's final temperature.

Alongside EI the script records the observational mutual information
I(A_li; A_l'i') of every head pair on the natural forward pass.
Observational dependence includes the common cause -- both heads read the
same input -- which EI excludes, so the two together separate influence
from shared input. The plug-in MI of a k x k table is biased up by about
(k-1)^2/(2 N ln 2) bits; it is reported with the Miller-Madow correction.

The effect variable is a head's argmax entry, its value in {1, ..., k}.
--soft uses the full assignment as the effect distribution instead,
treating the softmax as the channel's output.

The intervention distribution is one-hot entries, which are the natural
states of a hard code but not of a soft one: a softmax head whose
assignments are near uniform never takes a one-hot value, and forcing one
swaps its averaged atom for a single atom, a perturbation that can swamp
the residual the next layer classifies. EI then measures the response to
off-distribution interventions -- the dependence on the intervention
distribution that critiques of EI centre on. Each source head's natural
sharpness (its mean max assignment; 1 for a hard code, 1/k at uniform) is
recorded alongside its EI so the two can be read together.

A hard code is not immune: a head that naturally uses 3 of its 32
entries is forced, 29 times in 32, onto an entry it never takes. So EI
is also computed with the intervention uniform over each source head's
*live* entries only -- those its argmax takes on at least --live-min of
the natural tail -- from the same counts (`ei_live.npy`). For a hard code
that is the on-distribution measure; for a soft one it does not remove
the one-hot-versus-soft mismatch above.

Writes to --out (default <ckpt>/effinfo): ei.npy, determinism.npy and
degeneracy.npy, each (l, h, l, h) indexed [source layer, source head,
effect layer, effect head], NaN where the effect layer is not downstream;
ei_a.npy and ei_b.npy, the same EI on two disjoint halves of the contexts
(split-half reliability); ei_live.npy, EI with the intervention uniform
over live entries; mi_obs.npy, Miller-Madow observational MI for every
pair; heads.csv, per source head; summary.json.

  uv run python experiments/effective-information/effinfo.py \\
      --ckpt data/out/sonar/multilingual/bl_ctl_full
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

REPO = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Ontologizer checkpoint dir")
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--temperature", type=float, default=None,
                   help="classification temperature (default: the run's "
                        "hyper.temperature_end from log.jsonl)")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="cache tail rows held out from training (the sae.py "
                        "split); contexts and observational rows come from it")
    p.add_argument("--contexts", type=int, default=4096,
                   help="contexts per intervention, split into two halves")
    p.add_argument("--b", type=int, default=2048, help="forward batch")
    p.add_argument("--soft", action="store_true",
                   help="effect = full assignment instead of argmax entry")
    p.add_argument("--live-min", type=float, default=1e-3,
                   help="natural argmax usage at which an entry counts as "
                        "live, for the live-entry EI")
    p.add_argument("--out", default=None)
    return p.parse_args()


def entropy(p, axis=-1):
    """Shannon entropy in bits along `axis`, with 0 log 0 = 0."""
    p = np.asarray(p, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, -p * np.log2(p), 0.0)
    return t.sum(axis)


def effective_information(W):
    """EI, determinism and degeneracy in bits of the channel
    do(source = j) -> effect, from W[..., j, :] = P(effect | do(j)) with
    rows summing to 1. The source is intervened on uniformly over its j."""
    W = np.asarray(W, np.float64)
    n = W.shape[-1]
    h_each = entropy(W).mean(-1)          # mean_j H(W_j)
    h_pool = entropy(W.mean(-2))          # H(mean_j W_j)
    return h_pool - h_each, np.log2(n) - h_each, np.log2(n) - h_pool


def mutual_information(C, miller_madow=True):
    """Mutual information in bits of joint count tables C (..., ka, kb),
    with the Miller-Madow bias correction applied to each entropy."""
    C = np.asarray(C, np.float64)
    N = C.sum((-2, -1))
    p = C / N[..., None, None]
    pa, pb = p.sum(-1), p.sum(-2)
    mi = entropy(pa) + entropy(pb) - entropy(p.reshape(p.shape[:-2] + (-1,)))
    if miller_madow:
        bins = lambda q, ax: (q > 0).sum(ax)
        corr = ((bins(pa, -1) - 1) + (bins(pb, -1) - 1)
                - (bins(p, (-2, -1)) - 1))
        mi = mi + corr / (2 * N * np.log(2))
    return mi


def assignments(module, X: Float[Array, "b d"], layer: int, head, entry,
                temperature: float):
    """Every layer's assignments (b, l, h, k) and the decoded output under
    do[(layer, head) = entry]. Follows `Ontologizer.withArgs` -- the same
    per-layer `DictEnc.withStats` call, with the set-intervention entering
    the source layer's lookup -- but keeps each layer's classification
    instead of only the last. layer = -1 is the plain forward pass; head
    and entry may be traced."""
    E, _ = module.encode(X, 0.0, None)
    R = module.resid(E)
    E = module.constinput(E)
    Ps = []
    for i, de in enumerate(module.dictencs):
        kw = (dict(h_set=jnp.reshape(head, (1,)), k_set=jnp.reshape(entry, (1,)))
              if i == layer else {})
        R, K, _, _ = de.withStats(R, E, temperature=temperature, **kw)
        Ps.append(K.reshape(K.shape[:-1] + (module.h, module.k)))
        if i < module.l - 1:
            E = module.nextinput(X, R, K)
    return jnp.stack(Ps, -3), module.decode(R)


def effect_counts_fn(model, params, layer: int, temperature: float,
                     soft: bool):
    """jitted (X, head) -> (k_src, l - layer - 1, h, k): each downstream
    head's effect histogram under do[(layer, head) = j], for every entry j,
    summed over the rows of X."""
    def one(X, head, entry):
        P, _ = model.apply(params, X, layer, head, entry, temperature,
                           method=assignments)
        P = P[:, layer + 1:]
        if not soft:
            P = jax.nn.one_hot(P.argmax(-1), P.shape[-1], dtype=P.dtype)
        return P.sum(0)

    @jax.jit
    def all_entries(X, head):
        return jax.lax.map(lambda j: one(X, head, j), jnp.arange(model.k))
    return all_entries


def run_temperature(ckpt):
    """The run's final classification temperature, from its log.jsonl:
    `temperature_end` when it annealed, else the constant `temperature`."""
    log = Path(ckpt) / "log.jsonl"
    if log.exists():
        for line in reversed(log.read_text().splitlines()):
            hyper = json.loads(line).get("hyper") if line.strip() else None
            if hyper and "temperature" in hyper:
                T = hyper.get("temperature_end")
                return float(hyper["temperature"] if T is None else T)
    return None


def main():
    cfg = parse_args()
    sys.path.insert(0, str(REPO))
    from pareto import load_onto

    model, mparams, step = load_onto(cfg.ckpt, cfg.step)
    params = {"params": mparams}
    T = cfg.temperature if cfg.temperature is not None else run_temperature(cfg.ckpt)
    assert T is not None, "no temperature_end in log.jsonl; pass --temperature"
    l, h, k = model.l, model.h, model.k
    out = Path(cfg.out) if cfg.out else Path(cfg.ckpt) / "effinfo"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{cfg.ckpt} step {step}: {l} layers x {h} heads x {k} entries, "
          f"T = {T}")

    mm = np.load(cfg.cache, mmap_mode="r")
    tail = np.asarray(mm[-cfg.eval_rows:], dtype=np.float32)
    half = cfg.contexts // 2
    b = min(cfg.b, half)
    assert half % b == 0, "--contexts/2 must be a multiple of --b"
    ctx = tail[:cfg.contexts]

    # natural forward on the whole tail: realized bits and observational MI
    natural = jax.jit(lambda X: model.apply(params, X, -1, 0, 0, T,
                                            method=assignments)[0])
    C = jnp.zeros((l * h * k, l * h * k), jnp.float32)
    sharp = np.zeros((l, h))
    n_obs = len(tail) // b * b
    for s in range(0, n_obs, b):
        P = natural(jnp.asarray(tail[s:s + b]))                # (b, l, h, k)
        sharp += np.asarray(P.max(-1).sum(0), np.float64)
        O = jax.nn.one_hot(P.argmax(-1), k, dtype=jnp.float32).reshape(b, -1)
        C = C + O.T @ O
    sharp /= n_obs
    C = np.asarray(C, np.float64).reshape(l, h, k, l, h, k)
    joint = C.transpose(0, 1, 3, 4, 2, 5)                    # (l,h,l,h,k,k)
    mi_obs = mutual_information(joint)
    usage = np.stack([[np.diag(joint[a, i, a, i]) for i in range(h)]
                      for a in range(l)]) / n_obs             # (l, h, k)
    realized = entropy(usage)

    live = usage >= cfg.live_min                              # (l, h, k)
    shape = (l, h, l, h)
    ei, det, deg, ei_a, ei_b, ei_live = (np.full(shape, np.nan)
                                         for _ in range(6))
    for a in range(l - 1):
        counts = effect_counts_fn(model, params, a, T, cfg.soft)
        for i in range(h):
            halves = []
            for lo in (0, half):
                acc = 0.0
                for s in range(lo, lo + half, b):
                    acc = acc + np.asarray(
                        counts(jnp.asarray(ctx[s:s + b]), jnp.int32(i)),
                        np.float64)
                halves.append(acc)                          # (k, L, h, k)
            # move the source-entry axis next to the effect axis
            W = np.moveaxis((halves[0] + halves[1]) / cfg.contexts, 0, -2)
            e, dt, dg = effective_information(W)
            ei[a, i, a + 1:], det[a, i, a + 1:], deg[a, i, a + 1:] = e, dt, dg
            ei_live[a, i, a + 1:] = effective_information(
                W[..., live[a, i], :])[0]
            ei_a[a, i, a + 1:] = effective_information(
                np.moveaxis(halves[0] / half, 0, -2))[0]
            ei_b[a, i, a + 1:] = effective_information(
                np.moveaxis(halves[1] / half, 0, -2))[0]
        print(f"  source layer {a} done")

    for name, arr in [("ei", ei), ("determinism", det), ("degeneracy", deg),
                      ("ei_a", ei_a), ("ei_b", ei_b), ("ei_live", ei_live),
                      ("mi_obs", mi_obs)]:
        np.save(out / f"{name}.npy", arr)
    n_live = live.sum(-1)

    ok = np.isfinite(ei)
    r = float(np.corrcoef(ei_a[ok], ei_b[ok])[0, 1])
    blocks = []
    print(f"\n{'src->eff':<9} {'pairs':>6} {'mean EI':>8} {'max EI':>7} "
          f"{'>0.1b':>6} {'det':>6} {'deg':>6} {'live EI':>8} {'MI_obs':>7}")
    for a in range(l - 1):
        for c in range(a + 1, l):
            blk = ei[a, :, c, :]
            row = dict(src=a, eff=c, pairs=blk.size,
                       mean_ei=float(blk.mean()), max_ei=float(blk.max()),
                       frac_gt_0p1=float((blk > 0.1).mean()),
                       mean_det=float(det[a, :, c, :].mean()),
                       mean_deg=float(deg[a, :, c, :].mean()),
                       mean_ei_live=float(ei_live[a, :, c, :].mean()),
                       max_ei_live=float(ei_live[a, :, c, :].max()),
                       mean_mi_obs=float(mi_obs[a, :, c, :].mean()))
            blocks.append(row)
            print(f"{f'{a}->{c}':<9} {row['pairs']:>6} {row['mean_ei']:>8.4f} "
                  f"{row['max_ei']:>7.3f} {row['frac_gt_0p1']:>6.3f} "
                  f"{row['mean_det']:>6.3f} {row['mean_deg']:>6.3f} "
                  f"{row['mean_ei_live']:>8.4f} {row['mean_mi_obs']:>7.4f}")
    print(f"\nsplit-half reliability of EI over {ok.sum()} pairs: r = {r:.3f}")
    print(f"natural sharpness by layer (mean max assignment; 1/k = "
          f"{1 / k:.3f}): " + " ".join(f"{s:.3f}" for s in sharp.mean(1)))
    print(f"live entries per head by layer (usage >= {cfg.live_min:g}): "
          + " ".join(f"{n:.1f}" for n in n_live.mean(1)))

    with open(out / "heads.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", "realized_bits", "sharpness", "n_live",
                    "ei_sum", "ei_max", "ei_max_layer", "ei_max_head",
                    "ei_live_sum", "ei_live_max", "mi_obs_sum"])
        for a in range(l):
            for i in range(h):
                if a < l - 1:
                    down = ei[a, i, a + 1:]
                    am = np.unravel_index(np.argmax(down), down.shape)
                    w.writerow([a, i, realized[a, i], sharp[a, i],
                                n_live[a, i], down.sum(), down.max(),
                                a + 1 + am[0], am[1],
                                ei_live[a, i, a + 1:].sum(),
                                ei_live[a, i, a + 1:].max(),
                                mi_obs[a, i, a + 1:].sum()])
                else:
                    w.writerow([a, i, realized[a, i], sharp[a, i],
                                n_live[a, i]] + [""] * 7)

    summary = dict(ckpt=str(cfg.ckpt), step=int(step), temperature=T,
                   contexts=cfg.contexts, obs_rows=n_obs, soft=cfg.soft,
                   live_min=cfg.live_min, split_half_r=r,
                   sharpness=sharp.mean(1).tolist(),
                   live_entries=n_live.mean(1).tolist(), blocks=blocks)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

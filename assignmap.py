"""Per-layer sample x (head, entry) assignment heatmaps.

The loss.csv columns reduce head collapse and dead entries to scalars
(`KL_m`, `entropy`, `cossim_k_hmax`, `support`). This script shows them
per sample: for each layer, rows are held-out cache rows grouped into
slices by a known label, and columns are every head's k assignment
probabilities, read off the plain forward pass (`autointerp.onto_probe`).

  rows     a stratified draw of --per-label rows for each shown label
           value, from the tail --eval-rows of the cache (the sae.py
           holdout), sliced by label. Within a slice rows are ordered by
           their argmax entries, first displayed head first, so samples
           that share an assignment sit together.
  columns  heads in index order, separated by lines; within a head,
           entries are sorted by mean probability over the shown rows,
           so dead entries collect at each head's right edge. Columns
           are ordered, never clustered, so a head stays one block.
  strip    above the map, each entry's mean probability p_bar; over each
           head two effective entry counts out of k:
             use = exp H(p_bar)       how many entries the rows spread over
             row = exp mean_i H(p_i)  how many each row hedges over
           A collapsed head -- every sample on the same entry -- has use ~
           row ~ 1 and reads as one dark column. A head with row near k is
           undecided: near-uniform on every sample, whatever its use.

Colour is probability on a fixed [0, --vmax] sequential scale shared by
every panel, white at 0. Soft heads (row near k) are pale at the default
vmax 1; lower it to see their structure, and compare runs only at the same
vmax.

Labels (--by):
  lang      the cache's .langs.npy sidecar (SONAR mC4 caches)
  tokclass  coarse class of the current token (activation caches; decodes
            the .index.npy token column with the harvest model's tokenizer)
  pos       position bucket from the .index.npy position column
  doc       document id from the .index.npy doc column
The default is lang when the sidecar exists, else tokclass.

Heads are drawn --max-heads per page; --heads picks a subset.

  uv run python assignmap.py --ckpt data/out/sonar/multilingual/resid_nc
  uv run python assignmap.py --ckpt data/out/gpt2_l8/ste_h76 \\
      --cache data/activations/gpt2_l8.npy --by tokclass --heads 0 5 9

The temperature defaults to the run's schedule at the restored step (its
log.jsonl env_config). Writes <out>/assign_l<L>_p<j>.png per layer and
page, and assign.npz (the probabilities, labels and row indices) so the
figure can be redrawn without the model (--replot, which takes --out or
--ckpt to find it, and honours --heads/--max-heads/--vmax). NOTE: Ontologizer checkpoints
restore on GPU JAX only.
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from jaxtyping import Float, Int

POS_EDGES = (1, 2, 4, 16, 64)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Ontologizer checkpoint dir")
    p.add_argument("--step", type=int, default=0, help="default: latest")
    p.add_argument("--temperature", type=float, default=None,
                   help="default: the run's schedule at --step")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="held-out tail rows of the cache (sae.py's default)")
    p.add_argument("--by", choices=["lang", "tokclass", "pos", "doc"],
                   default=None, help="default: lang if the sidecar exists, "
                   "else tokclass")
    p.add_argument("--labels", nargs="+", default=None,
                   help="label values to show, in order (default: the "
                   "--n-labels most frequent in the tail)")
    p.add_argument("--n-labels", type=int, default=12)
    p.add_argument("--per-label", type=int, default=128)
    p.add_argument("--heads", type=int, nargs="+", default=None,
                   help="head indices to draw (default: all)")
    p.add_argument("--max-heads", type=int, default=16,
                   help="heads per page")
    p.add_argument("--vmax", type=float, default=1.0,
                   help="top of the shared colour scale; lower it for "
                   "soft heads (values above saturate)")
    p.add_argument("--replot", action="store_true",
                   help="redraw from <out>/assign.npz without the model")
    p.add_argument("--b", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="default <ckpt>/assignmap")
    return p.parse_args()


# ---------- labels ----------

def token_class(s: str) -> str:
    """Coarse class of one decoded GPT-2 token. A leading space marks a
    word start, so " the" is a word and "ing" a continuation."""
    if s == "" or s.isspace():
        return "newline" if "\n" in s else "space"
    t = s[1:] if s[0] == " " else s
    lead = s[0] == " "
    if t.isdigit():
        return "digit"
    if not any(c.isalnum() for c in t):
        return "punct"
    if not lead:
        return "cont"
    return "Word" if t[0].isupper() else "word"


def pos_bucket(pos: Int[np.ndarray, "n"],
               edges: Sequence[int] = POS_EDGES) -> np.ndarray:
    """Position -> bucket name: [e_i, e_{i+1}) for consecutive edges, the
    last bucket open-ended. Names sort in position order."""
    names = [f"{a}" if b == a + 1 else f"{a}-{b - 1}"
             for a, b in zip(edges[:-1], edges[1:])] + [f"{edges[-1]}+"]
    names = [f"p{i}:{n}" for i, n in enumerate(names)]
    idx = np.searchsorted(np.asarray(edges), pos, side="right") - 1
    return np.asarray(names)[np.clip(idx, 0, len(names) - 1)]


def tail_labels(cfg, n: int, lo: int) -> np.ndarray:
    """String label for every row in [lo, n) of the cache under --by."""
    cache = Path(cfg.cache)
    if cfg.by == "lang":
        side = cache.with_name(cache.name.replace(".npy", ".langs.npy"))
        langs = np.load(side, mmap_mode="r")
        assert len(langs) >= n, f"{side} covers {len(langs)} < {n} rows"
        return np.asarray(langs[lo:n]).astype(str)
    index = np.load(cache.with_name(cache.name.replace(".npy", ".index.npy")),
                    mmap_mode="r")
    doc, pos, tok = (np.asarray(index[lo:n, j]) for j in range(3))
    if cfg.by == "pos":
        return pos_bucket(pos)
    if cfg.by == "doc":
        return np.char.add("doc", doc.astype(str))
    from transformers import AutoTokenizer
    meta = json.loads(cache.with_name(
        cache.name.replace(".npy", ".meta.json")).read_text())
    tokzr = AutoTokenizer.from_pretrained(meta["model"])
    uniq, inv = np.unique(tok, return_inverse=True)
    cls = np.asarray([token_class(tokzr.decode([int(t)])) for t in uniq])
    return cls[inv]


def pick_rows(labels: np.ndarray, show: Optional[Sequence[str]],
              n_labels: int, per_label: int, seed: int):
    """(row offsets into `labels`, the shown label values). Shown values
    are `show` in its order, else the n_labels most frequent (ties by
    name). Each contributes up to per_label rows drawn without
    replacement, kept in cache order within the slice."""
    vals, counts = np.unique(labels, return_counts=True)
    if show is None:
        order = np.lexsort((vals, -counts))
        show = [str(v) for v in vals[order[:n_labels]]]
    missing = set(show) - set(vals.tolist())
    assert not missing, f"label values not in the tail: {sorted(missing)}"
    rng = np.random.default_rng(seed)
    rows = []
    for v in show:
        pool = np.flatnonzero(labels == v)
        take = rng.choice(pool, min(per_label, len(pool)), replace=False)
        rows.append(np.sort(take))
    return rows, list(show)


# ---------- ordering ----------

def entry_order(P: Float[np.ndarray, "n h k"]) -> Int[np.ndarray, "h k"]:
    """Per head, entries by mean probability, most used first (stable)."""
    return np.argsort(-P.mean(0), axis=-1, kind="stable")


def row_order(P: Float[np.ndarray, "n h k"],
              order: Int[np.ndarray, "h k"]) -> Int[np.ndarray, "n"]:
    """Rows sorted by their argmax entries' usage ranks, head 0 primary,
    so rows sharing an assignment are contiguous."""
    rank = np.argsort(order, axis=-1)  # entry -> its column within the head
    A = np.take_along_axis(rank[None], P.argmax(-1)[..., None],
                           -1)[..., 0]  # (n, h)
    return np.lexsort(A.T[::-1])


def eff_entries(P: Float[np.ndarray, "n h k"]) -> Float[np.ndarray, "h"]:
    """exp(H(p_bar)) per head: k when usage is uniform, 1 when every row
    lands on one entry."""
    pb = P.mean(0)
    pb = pb / pb.sum(-1, keepdims=True)
    H = -(pb * np.log(np.where(pb > 0, pb, 1.0))).sum(-1)
    return np.exp(H)


def row_eff(P: Float[np.ndarray, "n h k"]) -> Float[np.ndarray, "h"]:
    """exp(mean_i H(P_i)) per head: 1 when every row is one-hot, k when
    every row is uniform. Low eff_entries with low row_eff is collapse;
    high row_eff is an undecided head whatever its usage."""
    H = -(P * np.log(np.where(P > 0, P, 1.0))).sum(-1)
    return np.exp(H.mean(0))


def temperature_at(hyper: dict, step: int) -> float:
    """The training temperature at `step` under Hyperparams' geometric
    anneal (`temperature` -> `temperature_end` over `anneal_steps`)."""
    T0, T1, n = (hyper["temperature"], hyper.get("temperature_end"),
                 hyper.get("anneal_steps") or 0)
    if T1 is None or n == 0:
        return T0
    return T0 * (T1 / T0) ** min(step / n, 1.0)


def read_hyper(ckpt) -> Optional[dict]:
    """The run's recorded hyperparameters: the last `env_config`."""
    log = Path(ckpt) / "log.jsonl"
    if not log.exists():
        return None
    hyper = None
    with open(log) as f:
        for line in f:
            if '"env_config"' in line:
                hyper = json.loads(line)["hyper"]
    return hyper


# ---------- figure ----------

def draw_page(P: Float[np.ndarray, "n h k"], heads: Sequence[int],
              slices: Sequence[int], names: Sequence[str], title: str, path,
              vmax: float = 1.0):
    """One page: the rows of P (already in slice-then-argmax order) by
    the given heads' entries (each head usage-sorted). `slices` are the
    row counts of consecutive slices, `names` their labels. The colour
    scale is [0, vmax], the same on every page of a run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, _, k = P.shape
    Q = P[:, heads]
    order = entry_order(Q)
    Q = np.take_along_axis(Q, order[None], -1)
    M = Q.reshape(n, -1)
    usage = M.mean(0)
    eff, reff = eff_entries(Q), row_eff(Q)

    w = min(4 + M.shape[1] / 40, 40)
    fig, (ax_u, ax) = plt.subplots(
        2, 1, figsize=(w, 3 + n / 120), sharex=True,
        gridspec_kw={"height_ratios": [1, 8], "hspace": 0.04})
    x = np.arange(M.shape[1])
    ax_u.bar(x, usage, width=1.0, color="0.35")
    top = 1.35 * usage.max()  # headroom for the annotations
    ax_u.set_ylim(0, top)
    ax_u.set_ylabel("p̄", rotation=0, labelpad=10)
    ax_u.tick_params(labelbottom=False)
    for i, hd in enumerate(heads):
        ax_u.text((i + 0.5) * k - 0.5, 0.98 * top,
                  f"use {eff[i]:.1f}\nrow {reff[i]:.1f}",
                  ha="center", va="top", fontsize=6)

    im = ax.imshow(M, aspect="auto", interpolation="nearest",
                   cmap="Blues", vmin=0.0, vmax=vmax)
    for a in (ax, ax_u):
        for i in range(1, len(heads)):
            a.axvline(i * k - 0.5, color="0.2", lw=0.6)
    edges = np.cumsum(slices)
    for e in edges[:-1]:
        ax.axhline(e - 0.5, color="crimson", lw=0.6)
    mids = edges - np.asarray(slices) / 2 - 0.5
    ax.set_yticks(mids, names, fontsize=7)
    ax.set_xticks([(i + 0.5) * k - 0.5 for i in range(len(heads))],
                  [f"h{hd}" for hd in heads], fontsize=7)
    ax.set_xlim(-0.5, M.shape[1] - 0.5)
    ax.set_xlabel("head (entries sorted by usage within each head)")
    ax_u.set_title(title + f"   [per head, of k={k}: use = exp H(p̄), "
                   "row = exp mean H(p_i)]", fontsize=9)
    cb = fig.colorbar(im, ax=[ax_u, ax], fraction=0.015, pad=0.01,
                      extend="max" if vmax < 1 else "neither")
    cb.set_label("assignment probability")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_all(cfg, P: Float[np.ndarray, "n l h k"], slices: Sequence[int],
             shown: Sequence[str], step: int, out: Path):
    """Every layer's pages, with a one-line summary per layer."""
    _, l, h, k = P.shape
    heads = cfg.heads if cfg.heads is not None else list(range(h))
    pages = [heads[i:i + cfg.max_heads]
             for i in range(0, len(heads), cfg.max_heads)]
    starts = np.cumsum([0] + list(slices))
    for L in range(l):
        PL = P[:, L]
        eff, reff = eff_entries(PL), row_eff(PL)
        print(f"layer {L}: median use {np.median(eff):.1f}/{k}, median row "
              f"{np.median(reff):.1f}/{k}; {(eff < 1.5).sum()} heads with "
              f"use < 1.5")
        for j, page in enumerate(pages):
            # row order within each slice, from this page's heads
            order = entry_order(PL[:, page])
            perm = np.concatenate([
                starts[s] + row_order(PL[starts[s]:starts[s + 1]][:, page],
                                      order)
                for s in range(len(slices))])
            path = out / f"assign_l{L}_p{j}.png"
            draw_page(PL[perm], page, slices, shown,
                      f"{Path(cfg.ckpt).name} step {step} layer {L}"
                      f" (by {cfg.by})", path, cfg.vmax)
            print(f"-> {path}")


def main():
    cfg = parse_args()
    out = Path(cfg.out) if cfg.out else Path(cfg.ckpt) / "assignmap"
    if cfg.replot:
        z = np.load(out / "assign.npz")
        cfg.by, cfg.ckpt = str(z["by"]), str(z["run"])
        draw_all(cfg, z["P"], z["slices"].tolist(), z["shown"].tolist(),
                 int(z["step"]), out)
        return
    cache = Path(cfg.cache)
    if cfg.by is None:
        side = cache.with_name(cache.name.replace(".npy", ".langs.npy"))
        cfg.by = "lang" if side.exists() else "tokclass"

    mm = np.load(cache, mmap_mode="r")
    n = mm.shape[0]
    lo = n - cfg.eval_rows
    labels = tail_labels(cfg, n, lo)
    groups, shown = pick_rows(labels, cfg.labels, cfg.n_labels,
                              cfg.per_label, cfg.seed)
    rows = np.concatenate(groups)
    slices = [len(g) for g in groups]
    print(f"{len(rows)} tail rows in {len(shown)} slices by {cfg.by}")

    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from autointerp import onto_acts_fn

    T = cfg.temperature
    if T is None:
        hyper = read_hyper(cfg.ckpt)
        step = cfg.step or ocp.CheckpointManager(
            Path(cfg.ckpt).resolve(),
            checkpointers={'state': ocp.PyTreeCheckpointer(),
                           'spec': ocp.PyTreeCheckpointer()}).latest_step()
        assert hyper is not None, \
            "no env_config in log.jsonl; pass --temperature"
        T = temperature_at(hyper, step)
    acts, _, _, meta = onto_acts_fn(cfg.ckpt, cfg.step, T)
    l, h, k = meta["l"], meta["h"], meta["k"]
    print(f"step {meta['step']}: {l} layers x {h} heads x {k} entries, T={T:g}")

    X = np.asarray(mm[lo + rows], dtype=np.float32)
    P = np.concatenate([np.asarray(acts(jnp.asarray(X[i:i + cfg.b])))
                        for i in range(0, len(X), cfg.b)])
    P = P.reshape(len(rows), l, h, k)

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "assign.npz", P=P, rows=lo + rows,
             labels=labels[rows], slices=np.asarray(slices),
             shown=np.asarray(shown), by=cfg.by, run=Path(cfg.ckpt).name,
             step=meta["step"],
             temperature=T)
    draw_all(cfg, P, slices, shown, meta["step"], out)


if __name__ == "__main__":
    main()

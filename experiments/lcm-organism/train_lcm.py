"""Train a Base-LCM (lcm.py) on a document-ordered sentence cache.

    uv run python experiments/lcm-organism/train_lcm.py --seed 42 --epochs 1
    uv run python experiments/lcm-organism/train_lcm.py --max-steps 200 --b 64 \\
        --layers 2 --d-model 256 --heads 4 --d-ff 704 --out <dir>      # a pilot

**Data.** Every document before the cache's last `--tail-rows` rows (snapped
up to whole documents) is a training document, so harvests of the last
262,144 rows and every score on the 32,768-row tail see text the LCM never
trained on. Each epoch visits the documents in a fresh order (seeded by
`--seed` and the epoch), `--b` documents per step, each padded to `--t`
rows. A prediction is scored only where the next row is the next segment
of the text (`--targets adjacent`, the pairs `eval_next.py` scores), since
runs skip unusable segments; `--targets all` scores every next row.

**The scaler** is fitted on `--scaler-rows` training rows drawn with a
fixed seed, so every run on one cache shares it.

**Checkpoints** (orbax, under `<out>/ckpt`) hold the parameters, the
optimizer state and the scaler, every `--save-every` steps; the last two
are kept, and every `--keep-every`-th step permanently. Step 0, the
untrained init, is always saved and kept: `harvest_lcm.py --step 0`
harvests it. A run resumes from its latest checkpoint and continues the
same data order. On SIGTERM it saves the current step and exits 75, so a
preempted TPU job loses nothing that reached the bucket.

**Logs** go to `<out>/log.jsonl`:
- every `--log-every` steps, the training loss, gradient norm, learning
  rate and throughput;
- at step 0, every `--eval-every` steps and at the end, the held-out FVU_w
  of the next-sentence prediction on `eval_next.py`'s held-out pairs, under
  its weights, at matmul precision `highest`;
- once, unless `--no-baselines`, the same FVU_w for the training mean,
  copying x_t, a ridge from x_t (eval_next.py's 0.896), and a ridge from x_t
  together with the mean of the run's earlier rows (the context baseline of
  Gate A), fitted on the adjacent pairs of `--fit-docs` training documents.
"""
import os

# before jax is imported: JAX otherwise reserves 75% of the card at its first
# touch. setdefault, so an exported choice wins
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import queue  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, Iterator, Optional, Tuple  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "next-sentence"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402
from jaxtyping import Bool, Float, Int  # noqa: E402

from eval_next import fvu  # noqa: E402
from lcm import (BaseLCM, batches, doc_spans, fit_scaler, gather,  # noqa: E402
                 next_loss, next_targets)
from ontologize.data.loaders import doc_boundaries, doc_holdout  # noqa: E402
from pairs import adjacent  # noqa: E402

CACHE = ROOT / "data" / "sonar_embeddings" / "mc4_sent16.npy"
WEIGHTS = ROOT / "data" / "out" / "sonar" / "sentcache" / "mse_weights_no837.npy"
OUT = ROOT / "data" / "out" / "lcm-organism"
STOPPED = 75                     # the exit status after a SIGTERM save
MODEL_KEYS = ("d_model", "layers", "heads", "d_ff", "dropout", "gate")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--out", default=None, help="default: data/out/lcm-organism/<tag>")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=0, help="0: --epochs decides")
    p.add_argument("--b", type=int, default=512, help="documents per step")
    p.add_argument("--t", type=int, default=16, help="rows per document, padded")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=2816)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gate", default="silu",
                   help="the MLP's gate, an activation key: silu is SwiGLU, none a bilinear MLP")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, default=0.1, help="final lr, as a share of --lr")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--wd", type=float, default=0.1, help="AdamW decay on weight matrices")
    p.add_argument("--b1", type=float, default=0.9)
    p.add_argument("--b2", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=1.0, help="global gradient-norm clip")
    p.add_argument("--targets", choices=("adjacent", "all"), default="adjacent")
    p.add_argument("--tail-rows", type=int, default=262144,
                   help="rows at the end of the cache the LCM never trains on")
    p.add_argument("--holdout", type=int, default=32768, help="the scored tail")
    p.add_argument("--scaler-rows", type=int, default=1 << 17)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--keep-every", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-b", type=int, default=256)
    p.add_argument("--mse-weights", default=str(WEIGHTS),
                   help="weights of the held-out FVU_w (eval_next.py's default)")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--fit-docs", type=int, default=20000)
    p.add_argument("--alpha", type=float, default=1e-3, help="the ridges' penalty")
    p.add_argument("--prefetch", type=int, default=4)
    return p


def tag(cfg: argparse.Namespace) -> str:
    """The run's name; a gate other than SwiGLU's is named in it."""
    gate = "" if cfg.gate == "silu" else ("_bilinear" if cfg.gate == "none" else f"_g{cfg.gate}")
    return f"lcm_l{cfg.layers}_d{cfg.d_model}{gate}_s{cfg.seed}"


def build(cfg, d_sonar: int) -> BaseLCM:
    return BaseLCM(d_sonar=d_sonar, d_model=cfg.d_model, layers=cfg.layers,
                   heads=cfg.heads, d_ff=cfg.d_ff, dropout=cfg.dropout, gate=cfg.gate)


# ---------------------------------------------------------------- the split

def split(cache: Path, tail_rows: int, holdout: int) -> Dict[str, int]:
    """Row boundaries: training documents end at `train_hi`, the scored
    tail starts at `eval_lo`. Both tails are snapped up to whole
    documents, as `doc_holdout` snaps."""
    n = len(np.load(cache, mmap_mode="r"))
    docstart = np.load(cache.with_name(cache.stem + ".docstart.npy"))
    lo, hi = doc_boundaries(docstart, n, tail_rows)
    tail = lo if lo == tail_rows else hi
    return {"n": n, "train_hi": n - tail, "eval_lo": n - doc_holdout(cache, holdout)}


# ---------------------------------------------------------------- scoring

def forward(model: BaseLCM):
    """The deterministic forward, jitted; call it inside
    `jax.default_matmul_precision` to choose the precision."""
    return jax.jit(lambda v, x, pos, mask: model.apply(v, x, pos, mask))


def predict_rows(fwd, variables, arr, index, starts, lengths, b: int, t: int
                 ) -> Float[np.ndarray, "rows d"]:
    """The prediction made at every real row of these documents, in order."""
    out = []
    for s, n in batches(starts, lengths, b):
        x, pos, mask, _ = gather(arr, index, s, n, t)
        pred = np.asarray(fwd(variables, x, pos, mask))
        out.append(pred[mask])
    return np.concatenate(out)


def score_next(fwd, variables, arr, index, docstart, sp, held, w, b, t) -> float:
    """Held-out FVU_w of the prediction at row r against row r + 1, over
    the adjacent pairs `held` of the scored tail."""
    starts, lengths = doc_spans(docstart, sp["n"], sp["eval_lo"], sp["n"])
    with jax.default_matmul_precision("highest"):
        pred = predict_rows(fwd, variables, arr, index, starts, lengths, b, t)
    target = np.asarray(arr[held + 1], np.float64)
    return fvu(pred[held - sp["eval_lo"]].astype(np.float64), target, w)


class Ridge:
    """Least squares Y ~ X W + b with eval_next.py's penalty (`alpha` times
    the mean input variance), from sufficient statistics accumulated by
    chunk, so the fit never holds its pairs at once."""

    def __init__(self, dx: int, dy: int):
        self.n = 0
        self.sx, self.sy = np.zeros(dx), np.zeros(dy)
        self.xx, self.xy = np.zeros((dx, dx)), np.zeros((dx, dy))

    def add(self, X: Float[np.ndarray, "m dx"], Y: Float[np.ndarray, "m dy"]) -> None:
        X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        self.n += len(X)
        self.sx += X.sum(0)
        self.sy += Y.sum(0)
        self.xx += X.T @ X
        self.xy += X.T @ Y

    def solve(self, alpha: float) -> Tuple[Float[np.ndarray, "dx dy"], Float[np.ndarray, "dy"]]:
        mx, my = self.sx / self.n, self.sy / self.n
        G = self.xx / self.n - np.outer(mx, mx)
        C = self.xy / self.n - np.outer(mx, my)
        lam = alpha * float(np.trace(G)) / len(G)
        W = np.linalg.solve(G + lam * np.eye(len(G)), C)
        return W, my - mx @ W


def pair_features(x: Float[np.ndarray, "b t d"], mask: Bool[np.ndarray, "b t"],
                  adj: Bool[np.ndarray, "b t-1"]
                  ) -> Tuple[Float[np.ndarray, "p d"], Float[np.ndarray, "p e"],
                             Float[np.ndarray, "p d"]]:
    """For each adjacent pair (n, n + 1) of these documents, in row order:
    x_n; x_n with the mean of the run's rows before n and an indicator of
    n = 0 (the context baseline's input); and x_{n+1}."""
    t = x.shape[1]
    xm = x * mask[..., None]
    before = np.cumsum(xm, axis=1) - xm
    count = np.arange(t, dtype=np.float32)[None, :, None]
    mean = before / np.maximum(count, 1)
    first = np.broadcast_to((count == 0).astype(np.float32), x.shape[:2] + (1,))
    sel = adj
    xn = x[:, :-1][sel]
    ctx = np.concatenate([xn, mean[:, :-1][sel], first[:, :-1][sel]], -1)
    return xn, ctx, x[:, 1:][sel]


def baselines(arr, index, docstart, sp, w, fit_docs: int, alpha: float,
              t: int, b: int = 512) -> Dict[str, float]:
    """Held-out FVU_w of the next-sentence baselines (see the docstring),
    on the same pairs as `score_next`."""
    starts, lengths = doc_spans(docstart, sp["n"], 0, sp["train_hi"], min_len=2)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(starts), min(fit_docs, len(starts)), replace=False))
    d = arr.shape[1]
    plain, ctx = Ridge(d, d), Ridge(2 * d + 1, d)
    for s, n in batches(starts[pick], lengths[pick], b):
        x, _, mask, adj = gather(arr, index, s, n, t)
        xn, c, y = pair_features(x, mask, adj)
        plain.add(xn, y)
        ctx.add(c, y)
    es, el = doc_spans(docstart, sp["n"], sp["eval_lo"], sp["n"])
    X, Cx, Y = [], [], []
    for s, n in batches(es, el, b):
        x, _, mask, adj = gather(arr, index, s, n, t)
        xn, c, y = pair_features(x, mask, adj)
        X.append(xn), Cx.append(c), Y.append(y)
    X, Cx, Y = (np.concatenate(v).astype(np.float64) for v in (X, Cx, Y))
    W, b0 = plain.solve(alpha)
    Wc, bc = ctx.solve(alpha)
    mean = plain.sy / plain.n
    return {"fit_pairs": int(plain.n), "held_pairs": int(len(Y)),
            "train mean": fvu(np.broadcast_to(mean, Y.shape), Y, w),
            "copy x_t": fvu(X, Y, w),
            "ridge x_t": fvu(X @ W + b0, Y, w),
            "ridge x_t + context mean": fvu(Cx @ Wc + bc, Y, w)}


# ---------------------------------------------------------------- training

def make_tx(cfg, total: int) -> optax.GradientTransformation:
    sched = optax.warmup_cosine_decay_schedule(
        0.0, cfg.lr, min(cfg.warmup, max(total - 1, 1)), max(total, 2),
        end_value=cfg.lr * cfg.lr_end)
    decay = lambda params: jax.tree.map(lambda p: p.ndim >= 2, params)  # noqa: E731
    return optax.chain(optax.clip_by_global_norm(cfg.clip),
                       optax.adamw(sched, b1=cfg.b1, b2=cfg.b2,
                                   weight_decay=cfg.wd, mask=decay))


def init_state(model: BaseLCM, tx, seed: int, t: int,
               scaler: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> dict:
    x = jnp.zeros((1, t, model.d_sonar))
    v = model.init(jax.random.PRNGKey(seed), x, jnp.zeros((1, t), jnp.int32),
                   jnp.ones((1, t), bool))
    sc = v["scaler"] if scaler is None else {"median": jnp.asarray(scaler[0]),
                                            "iqr": jnp.asarray(scaler[1])}
    return {"step": jnp.zeros((), jnp.int32), "params": v["params"],
            "opt": tx.init(v["params"]), "scaler": sc}


def abstract(state: dict) -> dict:
    """The restore target: shapes, dtypes and this process's devices."""
    return jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=a.sharding),
                        state)


def manager(out: Path, cfg) -> ocp.CheckpointManager:
    """The run's checkpoints. orbax keeps every step divisible by
    `keep_period`, which step 0 always is; with `--keep-every 0` the period
    is one no later step reaches, so step 0 is still kept."""
    return ocp.CheckpointManager(
        (out / "ckpt").resolve(),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=cfg.save_every, max_to_keep=2,
            keep_period=cfg.keep_every or 10 ** 12, create=True))


def load_run(run: Path, step: int = -1):
    """`(model, variables, step, spec)` from a run directory, at `step`
    (-1: the latest checkpoint)."""
    run = Path(run)
    spec = json.loads((run / "spec.json").read_text())
    cfg = argparse.Namespace(**spec["train"])
    model = BaseLCM(**spec["model"])
    tx = make_tx(cfg, spec["total_steps"])
    mgr = manager(run, cfg)
    step = mgr.latest_step() if step < 0 else step
    if step is None:
        raise FileNotFoundError(f"{run} has no checkpoint")
    state = mgr.restore(step, args=ocp.args.StandardRestore(
        abstract(init_state(model, tx, 0, cfg.t))))
    return model, {"params": state["params"], "scaler": state["scaler"]}, step, spec


def doc_order(n_docs: int, seed: int, epoch: int) -> Int[np.ndarray, "n_docs"]:
    return np.random.default_rng([seed, epoch]).permutation(n_docs)


def stream(arr, index, starts, lengths, cfg, first: int, total: int
           ) -> Iterator[Tuple[int, tuple]]:
    """`(step, batch)` from `first` to `total`, in the seeded order."""
    per = math.ceil(len(starts) / cfg.b)
    epoch, order = -1, None
    for step in range(first, total):
        e, k = divmod(step, per)
        if e != epoch:
            epoch, order = e, doc_order(len(starts), cfg.seed, e)
        pick = np.sort(order[k * cfg.b:(k + 1) * cfg.b])
        (s, n), = batches(starts[pick], lengths[pick], cfg.b)
        yield step, gather(arr, index, s, n, cfg.t)


def prefetch(it: Iterator, depth: int) -> Iterator:
    """`it` run ahead by `depth` items on a thread."""
    q: queue.Queue = queue.Queue(maxsize=max(depth, 1))
    done = object()

    def work():
        try:
            for item in it:
                q.put(item)
            q.put(done)
        except BaseException as e:          # surfaced on the main thread
            q.put(e)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def train_step(model: BaseLCM, tx, adjacent_only: bool):
    def step(state, x, pos, mask, adj, key):
        target = next_targets(mask, adj, adjacent_only)

        def loss_fn(params):
            pred = model.apply({"params": params, "scaler": state["scaler"]},
                               x, pos, mask, train=True, rngs={"dropout": key})
            return next_loss(pred, x, target)

        (loss, n), g = jax.value_and_grad(loss_fn, has_aux=True)(state["params"])
        upd, opt = tx.update(g, state["opt"], state["params"])
        new = {**state, "params": optax.apply_updates(state["params"], upd),
               "opt": opt, "step": state["step"] + 1}
        return new, {"loss": loss, "targets": n, "grad_norm": optax.tree.norm(g)}
    return jax.jit(step, donate_argnums=(0,))


class Log:
    """`<out>/log.jsonl`. On a resume at step s, records past s are
    dropped, since the run repeats those steps."""

    def __init__(self, path: Path, resume: Optional[int]):
        self.path = path
        if resume is not None and path.exists():
            keep = [ln for ln in path.read_text().splitlines()
                    if ln.strip() and json.loads(ln).get("step", -1) <= resume]
            path.write_text("".join(ln + "\n" for ln in keep))

    def write(self, **rec) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def main(argv=None) -> int:
    cfg = parser().parse_args(argv)
    cache = Path(cfg.cache)
    out = Path(cfg.out) if cfg.out else OUT / tag(cfg)
    out.mkdir(parents=True, exist_ok=True)
    arr = np.load(cache, mmap_mode="r")
    index = np.load(cache.with_name(cache.stem + ".index.npy"), mmap_mode="r")
    docstart = np.load(cache.with_name(cache.stem + ".docstart.npy"))
    sp = split(cache, cfg.tail_rows, cfg.holdout)
    starts, lengths = doc_spans(docstart, sp["n"], 0, sp["train_hi"], min_len=2)
    per = math.ceil(len(starts) / cfg.b)
    total = cfg.max_steps or max(1, round(cfg.epochs * per))
    w = np.load(cfg.mse_weights).astype(np.float64)
    held = adjacent(index, sp["eval_lo"], sp["n"])

    model = build(cfg, arr.shape[1])
    spec = {"model": {"d_sonar": arr.shape[1], **{k: getattr(cfg, k) for k in MODEL_KEYS}},
            "train": {k: v for k, v in vars(cfg).items() if k != "out"},
            "split": sp, "train_docs": int(len(starts)), "steps_per_epoch": per,
            "total_steps": total}
    sfile = out / "spec.json"
    if sfile.exists():
        old = json.loads(sfile.read_text())
        if old["model"] != spec["model"]:
            raise SystemExit(f"{out} holds a different model: {old['model']}")
    sfile.write_text(json.dumps(spec, indent=2) + "\n")

    tx = make_tx(cfg, total)
    mgr = manager(out, cfg)
    latest = mgr.latest_step()
    if latest is None:
        rows = np.sort(np.random.default_rng(0).choice(sp["train_hi"], cfg.scaler_rows,
                                                       replace=False))
        state = init_state(model, tx, cfg.seed, cfg.t, fit_scaler(arr[rows]))
        mgr.save(0, args=ocp.args.StandardSave(state), force=True)
        mgr.wait_until_finished()
    else:
        state = mgr.restore(latest, args=ocp.args.StandardRestore(
            abstract(init_state(model, tx, cfg.seed, cfg.t))))
    first = int(state["step"])
    log = Log(out / "log.jsonl", None if latest is None else first)
    log.write(event="start", at=first, total=total, split=sp, train_docs=int(len(starts)),
              held_pairs=int(len(held)), devices=[str(d) for d in jax.devices()],
              matmul_precision=str(jax.config.jax_default_matmul_precision))
    print(f"{out.name}: {len(starts):,} training documents, {per:,} steps per epoch, "
          f"steps {first:,} to {total:,}; {len(held):,} held-out pairs")

    if latest is None and not cfg.no_baselines:
        base = baselines(arr, index, docstart, sp, w, cfg.fit_docs, cfg.alpha, cfg.t)
        log.write(step=0, baselines=base)
        print("baselines: " + ", ".join(f"{k} {v:.4f}" for k, v in base.items()
                                       if isinstance(v, float)))
    fwd = forward(model)

    def evaluate(step: int) -> None:
        v = {"params": state["params"], "scaler": state["scaler"]}
        f = score_next(fwd, v, arr, index, docstart, sp, held, w, cfg.eval_b, cfg.t)
        log.write(step=step, eval={"fvu_next": f})
        print(f"step {step:,}: held-out next-sentence FVU_w {f:.4f}")

    if first == 0:
        evaluate(0)
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    step_fn = train_step(model, tx, cfg.targets == "adjacent")
    key = jax.random.PRNGKey(cfg.seed + 1)
    t0, seen, acc = time.time(), 0, []
    step = first
    for step, (x, pos, mask, adj) in prefetch(stream(arr, index, starts, lengths, cfg,
                                                     first, total), cfg.prefetch):
        state, m = step_fn(state, x, pos, mask, adj, jax.random.fold_in(key, step))
        acc.append(m)
        seen += int(mask.sum())
        done = step + 1
        if done % cfg.log_every == 0 or done == total:
            ms = jax.device_get(acc)
            dt = time.time() - t0
            log.write(step=done, loss=float(np.mean([a["loss"] for a in ms])),
                      grad_norm=float(np.mean([a["grad_norm"] for a in ms])),
                      targets=int(np.sum([a["targets"] for a in ms])),
                      rows_per_s=seen / dt)
            print(f"step {done:,}: loss {np.mean([a['loss'] for a in ms]):.5f}, "
                  f"{seen / dt:,.0f} rows/s")
            t0, seen, acc = time.time(), 0, []
        mgr.save(done, args=ocp.args.StandardSave(state), force=stop["now"])
        if cfg.eval_every and done % cfg.eval_every == 0 and done != total:
            evaluate(done)
        if stop["now"]:
            mgr.wait_until_finished()
            log.write(step=done, event="stopped")
            print(f"SIGTERM: saved step {done:,}")
            return STOPPED
    mgr.wait_until_finished()
    if first < total and total not in mgr.all_steps():
        mgr.save(total, args=ocp.args.StandardSave(state), force=True)
        mgr.wait_until_finished()
    evaluate(total)
    log.write(step=total, event="done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

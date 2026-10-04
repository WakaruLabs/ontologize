"""Thin transcoder harness: train an Ontologizer to map x from one
embedding cache to y from another, using the stock training step.

This is the buildable part of DESIGN.md (section 3): a paired data
source plus a training loop that mirrors ontologize.training.ontostate
.train but unpacks (x, y) pairs -- the stock loop's single autoencoder
hardcode (`Y = X`) implemented outside the package. The jitted step
(`ontostate.update`), the schedules, checkpointing (`OntoState.save` +
`Metadata.manager`) and the loss (`Hyperparams.loss`, including target-
space mse_weights) are all the stock ones.

Constraints honored (DESIGN.md 2d): forward="labels" only. resid
forwarding compares the model INPUT to the output space (invalid for
d_x != d_y, target leakage if patched with y) and the inference-safe
"input" mode needs package changes, so neither is offered here.

Alignment contract: row i of --cache-x and --cache-y must come from the
same underlying sample (two encode_corpus.py passes over the same
corpus in the same order). The harness can only check lengths.

Also reports, on the cache tail (--eval-rows, the eval-split
convention): the trained model's FVU and a closed-form ridge baseline
x -> y fit on disjoint head rows -- the floor a transcoder must beat
before its dictionary structure means anything.

Run (from the repo root):

  uv run python experiments/transcoder/transcode_train.py \\
      --cache-x data/sonar_embeddings/mc4_4M.npy \\
      --cache-y /path/to/target_space.npy \\
      --mse-weights /path/to/target_space_mse_weights.npy

  # baseline + eval only (needs an existing run dir)
  uv run python experiments/transcoder/transcode_train.py \\
      --cache-x ... --cache-y ... --eval-only
"""
# disable preallocation so jax and torch can share VRAM (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import numpy as np
from pathlib import Path

import grain.python as gp

EXP_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-x", required=True,
                   help="input-space .npy cache (encode_corpus.py format); "
                        "e.g. data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--cache-y", required=True,
                   help="target-space .npy cache, row-aligned with "
                        "--cache-x")
    p.add_argument("--out", default=str(EXP_DIR / "runs" / "transcoder"),
                   help="run dir (orbax checkpoints, loss.csv, log.jsonl)")
    p.add_argument("--mse-weights", default="",
                   help="(d_y,) per-dim weights over the TARGET cache "
                        "(make_mse_weights.py on --cache-y). The shipped "
                        "SONAR weights are wrong for any other target "
                        "space. '' = unweighted")
    # model (sonar.py-sized defaults)
    p.add_argument("--e-dec", type=int, default=2048)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--h", type=int, default=32)
    p.add_argument("--l", type=int, default=5)
    p.add_argument("--deepsup", action="store_true",
                   help="per-prefix losses (works with labels forwarding)")
    # optimization (sonar.py defaults)
    p.add_argument("--b", type=int, default=256)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature-end", type=float, default=0.03)
    p.add_argument("--anneal-steps", type=int, default=50000)
    p.add_argument("--p-drop", type=float, default=0.1)
    p.add_argument("--sd-k", type=float, default=0.02)
    p.add_argument("--sd-f", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-each", type=int, default=100)
    p.add_argument("--checkpoint-each", type=int, default=10000)
    # regularizers (sonar.py live values)
    p.add_argument("--s-l1f", type=float, default=1e-9)
    p.add_argument("--s-bcossim", type=float, default=1e-5)
    p.add_argument("--s-hcossim", type=float, default=1e-6)
    p.add_argument("--s-hm", type=float, default=1e-6)
    # eval
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="cache-tail rows for FVU (eval-split convention)")
    p.add_argument("--baseline-rows", type=int, default=65536,
                   help="head rows for the ridge baseline fit (disjoint "
                        "from the tail)")
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--eval-only", action="store_true",
                   help="skip training; evaluate the latest checkpoint in "
                        "--out plus the ridge baseline")
    return p.parse_args()


class PairedNpySource(gp.RandomAccessDataSource):
    """Row-aligned pair of memory-mapped .npy caches -> {"x", "y"} dicts.
    Lazy-memmap pickling pattern follows loaders.NpyDataSource so the
    source survives grain worker processes."""

    def __init__(self, file_x, file_y):
        self.file_x, self.file_y = str(file_x), str(file_y)
        self._x = np.load(self.file_x, mmap_mode="r")
        self._y = np.load(self.file_y, mmap_mode="r")
        if self._x.shape[0] != self._y.shape[0]:
            raise SystemExit(
                f"cache length mismatch: {self.file_x} has "
                f"{self._x.shape[0]} rows, {self.file_y} has "
                f"{self._y.shape[0]} -- the pairing contract is broken")
        self._n = self._x.shape[0]
        self.d_x = int(self._x.shape[1])
        self.d_y = int(self._y.shape[1])

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        if self._x is None:
            self._x = np.load(self.file_x, mmap_mode="r")
            self._y = np.load(self.file_y, mmap_mode="r")
        i = int(idx)
        # copy rows out so batches don't pin the memmaps
        return {"x": np.array(self._x[i]), "y": np.array(self._y[i])}

    def __getstate__(self):
        return {"file_x": self.file_x, "file_y": self.file_y,
                "_n": self._n, "d_x": self.d_x, "d_y": self.d_y,
                "_x": None, "_y": None}


def fvu(pred, Y, w):
    num = (((pred - Y) ** 2) * w).mean()
    den = (((Y - Y.mean(0)) ** 2) * w).mean()
    return float(num / den)


def ridge_baseline(cfg, src, w):
    """Closed-form ridge x -> y (with intercept) on head rows, FVU on the
    tail. The floor a transcoder must beat."""
    n_fit = min(cfg.baseline_rows, len(src) - cfg.eval_rows)
    Xf = np.asarray(src._x[:n_fit], np.float64)
    Yf = np.asarray(src._y[:n_fit], np.float64)
    mx, my = Xf.mean(0), Yf.mean(0)
    Xc, Yc = Xf - mx, Yf - my
    G = Xc.T @ Xc + cfg.ridge * n_fit * np.eye(src.d_x)
    W = np.linalg.solve(G, Xc.T @ Yc)                  # (d_x, d_y)
    Xe = np.asarray(src._x[-cfg.eval_rows:], np.float64)
    Ye = np.asarray(src._y[-cfg.eval_rows:], np.float64)
    return fvu((Xe - mx) @ W + my, Ye, w)


def model_fvu(cfg, model, params, src, w):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def fwd(X):
        return model.apply(params, X)

    preds, Ys = [], []
    for i in range(len(src) - cfg.eval_rows, len(src), cfg.b):
        X = jnp.asarray(np.asarray(src._x[i:i + cfg.b], np.float32))
        preds.append(np.asarray(fwd(X), np.float64))
        Ys.append(np.asarray(src._y[i:i + cfg.b], np.float64))
    return fvu(np.concatenate(preds), np.concatenate(Ys), w)


def train(cfg, src):
    from tqdm import tqdm
    from ontologize.training.config import Hyperparams, Metadata, log_env
    from ontologize.training.ontostate import (update, schedules,
                                               load_params)
    from ontologize.data.loaders import SampleLoader

    out = Path(cfg.out)
    d_x, d_y = src.d_x, src.d_y
    hyper = Hyperparams(
        d_x, d_y, cfg.b, cfg.epochs, cfg.lr, 0.0, cfg.temperature,
        "batchnorm", "normal", "featvar", 0.0, cfg.sd_k, cfg.sd_f,
        s_g=0.0, s_L1K=0.0, s_L1F=cfg.s_l1f, s_H=0.0,
        s_bcossim=cfg.s_bcossim, s_hcossim=cfg.s_hcossim, s_Hm=cfg.s_hm,
        p_drop=cfg.p_drop, p_drop_start=0.0,
        mse_weights=cfg.mse_weights or None,
        temperature_end=cfg.temperature_end, anneal_steps=cfg.anneal_steps,
        sd_K_end=0.2 * cfg.temperature_end,
        ghost=False, seed=cfg.seed)

    # labels forwarding only; see the module docstring / DESIGN.md 2d
    model = hyper.ontologizer(
        d_x, d_y, cfg.e_dec, cfg.k, cfg.h, cfg.l, n=2, gate="none",
        scaled=False, encoded=False,
        forward="labels", deepsup=cfg.deepsup,
        dtype_str="float32", dtype_p_str="float32")

    meta = Metadata("embedding", None, out_path=out, threads=0,
                    save_each=cfg.save_each,
                    checkpoint_each=cfg.checkpoint_each)
    manager = meta.manager()
    state = hyper.init(model, save_each=cfg.save_each)

    resume = manager.latest_step()
    if resume is not None:
        state = load_params(state, manager, resume)
        # truncate loss.csv to the resumed step (TrainingEnv convention)
        loss_path = manager.directory / "loss.csv"
        if loss_path.exists():
            lines = loss_path.read_text().splitlines(keepends=True)
            loss_path.write_text("".join(lines[:resume]))
        print(f"resumed from step {resume}")
    elif cfg.eval_only:
        print("WARNING: --eval-only with no checkpoint in "
              f"{out} -- the model FVU below is a random-init model")

    if not cfg.eval_only:
        log_env(out / "log.jsonl", hyper, meta)
        loader = SampleLoader(cfg.b, cfg.epochs, 0, src, [],
                              threads=0, shuffle=True, seed=cfg.seed)
        rng = hyper.rng()
        with tqdm(loader, desc=f"Transcoding {cfg.epochs} epochs") as pbar:
            for batch in pbar:
                X, Y = batch["x"], batch["y"]
                T, pd, sk = schedules(
                    int(state.step), cfg.anneal_steps,
                    cfg.temperature, cfg.temperature_end,
                    cfg.p_drop, 0.0, cfg.sd_k, 0.2 * cfg.temperature_end)
                state, L, rng = update(
                    state, hyper.loss, rng, X, Y,
                    temperature=T, p_drop=pd, sd_K=sk,
                    sd_in=0.0, sd_F=cfg.sd_f, grad_clip=hyper.grad_clip)
                pbar.set_postfix({"loss": float(L),
                                  "step": int(state.step),
                                  "T": round(T, 4)})
                if cfg.save_each and not (state.step % cfg.save_each):
                    if state.save(manager):
                        if state.writestats(manager):
                            state = state.replace(stats=state.newstats())
        print("Training Complete!")

    return model, state


def main():
    cfg = parse_args()
    src = PairedNpySource(cfg.cache_x, cfg.cache_y)
    print(f"pairs: {len(src)} rows, d_x={src.d_x} -> d_y={src.d_y}")
    if cfg.eval_rows + cfg.baseline_rows > len(src):
        raise SystemExit("eval tail and baseline head overlap; shrink "
                         "--eval-rows/--baseline-rows")

    w = (np.load(cfg.mse_weights).astype(np.float64) if cfg.mse_weights
         else np.ones(src.d_y))
    if len(w) != src.d_y:
        raise SystemExit(f"--mse-weights dim {len(w)} != d_y {src.d_y} "
                         "(weights must be computed over the TARGET cache)")

    model, state = train(cfg, src)

    base = ridge_baseline(cfg, src, w)
    mf = model_fvu(cfg, model, state.params, src, w)
    print(f"\ncache-tail FVU ({cfg.eval_rows} rows, "
          f"{'weighted' if cfg.mse_weights else 'unweighted'}):")
    print(f"  ridge baseline (lambda={cfg.ridge}): {base:.4f}")
    print(f"  ontologizer transcoder:              {mf:.4f}")
    if mf >= base:
        print("  NOTE: the transcoder does not beat the linear floor -- "
              "its dictionary structure explains nothing the ridge map "
              "doesn't. Treat downstream analyses accordingly.")


if __name__ == "__main__":
    main()

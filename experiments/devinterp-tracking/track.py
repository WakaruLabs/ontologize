"""Developmental tracking over saved checkpoints: do classifications
sharpen at stage boundaries, and does head-level semantic structure
(headcoh's coherence z-scores) emerge gradually or in jumps?

Sweeps every checkpoint the run's orbax CheckpointManager retained
(save_interval_steps=save_each with max_to_keep, plus keep_period=
checkpoint_each kept indefinitely; see ontologize/training/config.py
Metadata.manager) and computes, per checkpoint step:

  geometry   headcoh.group_zscores over the model's exact end-to-end
             direction matrix G (refit.onto_linear_model: Y = P_flat @ G),
             per-head z_cos / z_sv against size-matched within-layer nulls.
  code shape activations P over a fixed cache slice: per-head mean
             per-sample entropy (bits), sharpness (mean max p), and usage
             KL(E[p] || uniform) in bits; KL summed per layer and over
             layers reproduces the loss.csv KL_m stat's definition
             (DictBlock.hmean_kl) on a larger, noise-free sample.
  training   the loss.csv block covering (step - save_each, step], averaged
             (loss / MSE / KL_m columns), so eval-time metrics line up with
             what training saw.

Writes two CSVs ready for plotting: steps.csv (one aggregate row per
checkpoint) and heads.csv (one row per head per checkpoint, for spotting
individual heads snapping into place at a boundary). Both are appended;
already-recorded steps are skipped, so the sweep is resumable and can be
re-run as training deposits new checkpoints.

Temperature: post-hoc analyses in this repo evaluate at the anneal floor
(0.03), but early checkpoints trained at much softer temperatures, so a
fixed eval temperature confounds "the schedule hardened" with "the model
learned". --schedule evaluates each checkpoint at its own training-time
temperature (recovered from the run's log.jsonl env_config, falling back
to --anneal-* flags), via ontologize.training.ontostate.schedules. Run
both and difference them: sharpening visible at FIXED temperature is
learning, not schedule.

  uv run python experiments/devinterp-tracking/track.py \\
      --ckpt data/out/sonar/multilingual/resid_nc_hm
  uv run python experiments/devinterp-tracking/track.py \\
      --ckpt data/out/sonar/multilingual/resid_nc_hm --schedule \\
      --out experiments/devinterp-tracking/out/resid_nc_hm_sched
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import json
import sys
import numpy as np
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

# imported, not copied: this list used to be a local 9-wide copy that put
# KL_m at index 8, which is `cossim_k` in the real layout -- so every
# tracked run recorded cossim_k under the name train_KL_m
from ontologize.visualize.loss import COLUMNS as LOSS_COLS
DEFAULT_SAVE_EACH = 100  # sonar.py's live save cadence


# ---------- pure helpers ----------

def entropy_bits(P, eps=1e-12):
    """Shannon entropy in bits over the last axis (numpy mirror of
    ontologize.fns.loss.entropy)."""
    P = P + eps
    P = P / P.sum(-1, keepdims=True)
    return -(P * np.log2(P)).sum(-1)


def head_code_stats(P_sum, H_sum, mp_sum, nb, k):
    """Finalize per-head accumulators: (mean per-sample entropy, mean max
    p, usage KL from uniform in bits)."""
    Ep = P_sum / nb
    Ep = Ep / Ep.sum(-1, keepdims=True)
    kl = np.log2(k) - entropy_bits(Ep)
    return H_sum / nb, mp_sum / nb, kl


def loss_block(loss, step, save_each):
    """Mean of the loss.csv rows covering steps (step - save_each, step].
    Row order inside a block is rotated (stats_insert writes row
    step % save_each), so a block MEAN is the robust join. Returns None
    when the block isn't fully on file."""
    if loss is None or step < save_each or step % save_each:
        return None
    j = step // save_each
    lo, hi = (j - 1) * save_each, j * save_each
    if hi > len(loss):
        return None
    return loss[lo:hi].mean(0)


def read_schedule(ckpt):
    """Recover (temperature, temperature_end, anneal_steps) from the run's
    log.jsonl env_config record; None when unavailable."""
    log = Path(ckpt) / "log.jsonl"
    if not log.exists():
        return None
    with open(log) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "env_config":
                hy = rec["hyper"]
                return (hy["temperature"], hy["temperature_end"],
                        hy["anneal_steps"])
    return None


def done_steps(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {int(r["step"]) for r in csv.DictReader(f)}


# ---------- per-checkpoint measurement ----------

def measure_step(cfg, step, T, mm):
    """All metrics for one checkpoint step. Restores the checkpoint via
    refit.onto_linear_model (which reuses autointerp.onto_acts_fn), so the
    acts probe and direction matrix G are byte-identical to the ones the
    post-hoc analyses use."""
    import jax.numpy as jnp
    from refit import onto_linear_model
    from headcoh import group_zscores

    acts, G, meta = onto_linear_model(cfg.ckpt, step, T)
    l, h, k = meta["l"], meta["h"], meta["k"]
    F = l * h * k
    f = np.arange(F)
    labels, pools = f // k, f // (h * k)  # head id / layer id (headcoh conv.)
    geo = group_zscores(np.asarray(G, np.float32), labels, pools,
                        cfg.n_null, cfg.seed)
    geo = {r["group"]: r for r in geo}

    n = min(cfg.rows, mm.shape[0]) // cfg.b * cfg.b
    if n == 0:
        raise SystemExit("--rows must cover at least one batch (--b)")
    P_sum = np.zeros((l * h, k))
    H_sum = np.zeros(l * h)
    mp_sum = np.zeros(l * h)
    nb = 0
    for i in range(0, n, cfg.b):
        A = np.asarray(acts(jnp.asarray(
            np.asarray(mm[i:i + cfg.b], np.float32))))
        P = A.reshape(-1, l * h, k).astype(np.float64)
        P_sum += P.mean(0)
        H_sum += entropy_bits(P).mean(0)
        mp_sum += P.max(-1).mean(0)
        nb += 1
    H_s, mp, kl = head_code_stats(P_sum, H_sum, mp_sum, nb, k)
    return geo, H_s, mp, kl, meta


def head_rows(step, T, geo, H_s, mp, kl, h):
    for g in sorted(geo):
        r = geo[g]
        yield {"step": step, "T_eval": f"{T:.5f}", "layer": r["pool"],
               "head": g % h, "head_global": g, "size": r["size"],
               "mean_cos": f"{r['mean_cos']:.4f}",
               "z_cos": f"{r['z_cos']:.3f}", "sv1": f"{r['sv1']:.4f}",
               "z_sv": f"{r['z_sv']:.3f}", "H_sample": f"{H_s[g]:.4f}",
               "sharpness": f"{mp[g]:.4f}", "KL_usage": f"{kl[g]:.4f}"}


def step_row(step, T, geo, H_s, mp, kl, meta, train):
    l, h, k = meta["l"], meta["h"], meta["k"]
    z_cos = np.array([geo[g]["z_cos"] for g in sorted(geo)])
    z_sv = np.array([geo[g]["z_sv"] for g in sorted(geo)])
    kl_layers = kl.reshape(l, h).mean(1)  # per-layer mean over heads
    row = {"step": step, "T_eval": f"{T:.5f}",
           "mean_z_cos": f"{z_cos.mean():.3f}",
           "median_z_cos": f"{np.median(z_cos):.3f}",
           "frac_z_cos_gt2": f"{(z_cos > 2).mean():.3f}",
           "frac_z_cos_lt_neg2": f"{(z_cos < -2).mean():.3f}",
           "mean_z_sv": f"{z_sv.mean():.3f}",
           "mean_H_sample": f"{H_s.mean():.4f}",
           "mean_sharpness": f"{mp.mean():.4f}",
           "KL_m_eval": f"{kl_layers.sum():.4f}"}
    for li in range(l):
        sl = slice(li * h, (li + 1) * h)
        row[f"H_sample_l{li}"] = f"{H_s[sl].mean():.4f}"
        row[f"sharpness_l{li}"] = f"{mp[sl].mean():.4f}"
        row[f"KL_usage_l{li}"] = f"{kl_layers[li]:.4f}"
    for name in ("loss", "MSE", "KL_m"):
        row[f"train_{name}"] = ""
    if train is not None:
        cols = LOSS_COLS[:len(train)]
        idx = {c: i for i, c in enumerate(cols)}
        for name in ("loss", "MSE", "KL_m"):
            if name in idx:
                row[f"train_{name}"] = f"{train[idx[name]]:.6g}"
    return row


def append_csv(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row[0].keys())
                           if isinstance(row, list) else list(row.keys()))
        if new:
            w.writeheader()
        w.writerows(row if isinstance(row, list) else [row])


# ---------- main ----------

def list_steps(ckpt):
    import orbax.checkpoint as ocp
    manager = ocp.CheckpointManager(
        Path(ckpt).resolve(),
        checkpointers={"state": ocp.PyTreeCheckpointer(),
                       "spec": ocp.PyTreeCheckpointer()})
    steps = sorted(int(s) for s in manager.all_steps())
    if not steps:
        raise SystemExit(f"no checkpoints under {ckpt}")
    return steps


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="data/out/sonar/multilingual/resid_nc_hm",
                   help="run dir (orbax manager layout); resid_nc is the "
                        "other live run")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--rows", type=int, default=16384,
                   help="cache rows for the activation statistics (a fixed "
                        "slice from the cache head, same for every step)")
    p.add_argument("--b", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.03,
                   help="fixed eval temperature (the post-hoc convention)")
    p.add_argument("--schedule", action="store_true",
                   help="evaluate each checkpoint at its training-time "
                        "annealed temperature instead (log.jsonl, or the "
                        "--anneal-* fallbacks)")
    p.add_argument("--anneal-temperature", type=float, default=1.0)
    p.add_argument("--anneal-temperature-end", type=float, default=0.03)
    p.add_argument("--anneal-steps", type=int, default=50000)
    p.add_argument("--save-each", type=int, default=DEFAULT_SAVE_EACH,
                   help="the run's loss-flush cadence, for the loss.csv join")
    p.add_argument("--n-null", type=int, default=200,
                   help="size-matched null draws per pool (headcoh uses 500; "
                        "200 keeps the sweep fast, z resolution ~0.07)")
    p.add_argument("--stride", type=int, default=1,
                   help="keep every stride-th retained checkpoint")
    p.add_argument("--min-step", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="default experiments/devinterp-tracking/out/<run name>")
    cfg = p.parse_args()

    out = Path(cfg.out) if cfg.out else EXP / "out" / Path(cfg.ckpt).name
    out.mkdir(parents=True, exist_ok=True)
    steps_csv, heads_csv = out / "steps.csv", out / "heads.csv"

    steps = [s for s in list_steps(cfg.ckpt) if s >= cfg.min_step]
    steps = steps[::cfg.stride]
    done = done_steps(steps_csv)
    todo = [s for s in steps if s not in done]
    print(f"{len(steps)} retained checkpoints, {len(todo)} to measure "
          f"-> {out}")

    sched = read_schedule(cfg.ckpt) if cfg.schedule else None
    if cfg.schedule and sched is None:
        sched = (cfg.anneal_temperature, cfg.anneal_temperature_end,
                 cfg.anneal_steps)
        print("schedule: no log.jsonl env_config; using --anneal-* flags")

    loss = None
    loss_path = Path(cfg.ckpt) / "loss.csv"
    if loss_path.exists():
        try:
            loss = np.loadtxt(loss_path, delimiter=",", ndmin=2)
        except ValueError as e:
            print(f"warning: could not parse {loss_path}: {e}")

    mm = np.load(cfg.cache, mmap_mode="r")
    from ontologize.training.ontostate import schedules
    for s in todo:
        if sched is not None:
            T = schedules(s, sched[2], sched[0], sched[1],
                          0.0, None, 0.0, None)[0]
        else:
            T = cfg.temperature
        geo, H_s, mp, kl, meta = measure_step(cfg, s, T, mm)
        train = loss_block(loss, s, cfg.save_each)
        append_csv(heads_csv,
                   list(head_rows(s, T, geo, H_s, mp, kl, meta["h"])))
        row = step_row(s, T, geo, H_s, mp, kl, meta, train)
        append_csv(steps_csv, row)
        print(f"step {s}: T {T:.4f} mean_z_cos {row['mean_z_cos']} "
              f"sharpness {row['mean_sharpness']} "
              f"KL_m_eval {row['KL_m_eval']}", flush=True)
    print(f"-> {steps_csv}\n-> {heads_csv}")


if __name__ == "__main__":
    main()

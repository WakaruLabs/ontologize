"""Train an Ontologizer on the GPT-2 activation cache (train rows only), then
run diagnostics and splice CE on the held-out eval rows.

Defaults mirror sonar.py's live config (resid forwarding, joint deep
supervision, resid_norm + resid_const, bilinear classifiers, T 1 -> 0.03
over 50k steps, sd_K 0.02 -> 0.2*T_end, winner dropout ramp 0 -> 0.1, featvar
F-noise, the same regularizer weights), except: plain (unweighted) MSE on
the normalized cache, ghost off, e_dec = 2*d.

  uv run python experiments/gpt2/train_onto.py --name base
  uv run python experiments/gpt2/train_onto.py --name first --resid-first
  uv run python experiments/gpt2/train_onto.py --name lnorm --resid-first --logit-norm
  uv run python experiments/gpt2/train_onto.py --name gain --resid-first --resid-gain
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# JAX never returns its pool to the device; cap it so two concurrent runs
# plus GPT-2 (torch, for the splice eval) fit in 12 GB
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
# not "platform": it cudaMalloc/cudaFree-s every buffer each step (~1.7x slower here)

import argparse
import dataclasses
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2]))
from common import Cache  # noqa: E402
from ontologize.training.config import Hyperparams, Metadata  # noqa: E402
import diagnose  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--cache", default="data/gpt2_l8")
    p.add_argument("--out-root", default="data/out/gpt2")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--b", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--l", type=int, default=5)
    p.add_argument("--h", type=int, default=32)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--e-dec", type=int, default=0, help="default 2*d")
    p.add_argument("--forward", default="resid", choices=["resid", "labels"])
    p.add_argument("--no-deepsup", action="store_true")
    p.add_argument("--deepsup-sg", action="store_true")
    p.add_argument("--no-resid-norm", action="store_true")
    p.add_argument("--no-resid-const", action="store_true")
    p.add_argument("--resid-first", action="store_true")
    p.add_argument("--resid-gain", action="store_true")
    p.add_argument("--logit-norm", action="store_true")
    p.add_argument("--select", default="softmax", choices=["softmax", "ste", "anneal", "topm"],
                   help="ste: hard one-hot forward, softmax gradient backward; anneal: "
                        "scheduled mix from softmax to ste over --anneal-steps; topm: "
                        "normalized top-m with m: k -> 1 over --anneal-steps")
    p.add_argument("--private-heads", action="store_true",
                   help="head-private dictionary blocks of e_dec/h dims, concatenated")
    p.add_argument("--signed-dict", action="store_true", help="no abs() on entries")
    p.add_argument("--direct", action="store_true",
                   help="entries in output space, no decoder (sets e_dec = d)")
    p.add_argument("--p-head-drop", type=float, default=0.0)
    p.add_argument("--fiber-rank", type=int, default=0,
                   help="per-entry local basis of this rank (tangent-bundle fiber)")
    p.add_argument("--fiber-drop", type=float, default=0.0,
                   help="per-(sample, head) probability of zeroing the fiber in training")
    p.add_argument("--fiber-bound", type=float, default=0.0,
                   help="tanh bound on fiber coordinates (0 = unbounded)")
    p.add_argument("--fiber-eps", type=float, default=0.0,
                   help="cap per-head fiber output norm (fraction of the residual under gain)")
    p.add_argument("--fiber-radial", action="store_true",
                   help="rank-1 fiber tied to the entry direction (per-head gain)")
    p.add_argument("--fiber-shared", action="store_true",
                   help="one fiber coordinate vector per layer shared by all heads")
    p.add_argument("--fiber-eps-layer", type=float, default=0.0,
                   help="cap the sum over heads of fiber norms (group-lasso ball)")
    p.add_argument("--base-aux", type=int, default=0,
                   help="add N copies of a fiber-free decode to the deep-supervision loss "
                        "(weight N/(l+N))")
    p.add_argument("--init-from", default=None,
                   help="run dir whose latest checkpoint initializes all matching params")
    p.add_argument("--train-only", default="all", choices=["all", "fiber"],
                   help="fiber: freeze everything except fiber bases and fiber reads "
                        "(error correction on top of a trained, frozen base)")
    p.add_argument("--gain-clip", action="store_true",
                   help="with --resid-gain: clip each layer's gain at the input norm")
    p.add_argument("--head-sparse", default="none", choices=["none", "jumprelu", "topk"],
                   help="per-head sparse encoder before the classifier; its L0 goes in the "
                        "L1_K stat slot, weighted by --s-L1K")
    p.add_argument("--m-h", type=int, default=32, help="sparse features per head")
    p.add_argument("--hs-shared", action="store_true",
                   help="one sparse encoder per layer shared by all heads")
    p.add_argument("--k-z", type=int, default=4, help="active features per head (topk)")
    p.add_argument("--hs-bandwidth", type=float, default=1e-3)
    p.add_argument("--hs-init-threshold", type=float, default=1e-3)
    p.add_argument("--hs-lr-mult", type=float, default=1.0,
                   help="learning-rate multiplier for the sparse-encoder parameters")
    p.add_argument("--resample-period", type=int, default=0,
                   help="resample dead sparse-encoder latents every N steps (0 = off; see resample.py)")
    p.add_argument("--resample-count-every", type=int, default=50)
    p.add_argument("--resample-scale", type=float, default=1.1,
                   help="new row preact on its target input = scale x that input's current top-k threshold")
    p.add_argument("--hs-decoder", action="store_true",
                   help="auxiliary SAE decoder on the sparse encoder (loss in the aux stat slot)")
    p.add_argument("--hs-auxk", type=int, default=0, help="AuxK dead-latent term (inactive latents used)")
    p.add_argument("--s-aux", type=float, default=0.0, help="weight of the aux SAE loss (0.5 in the SAE recipe)")
    p.add_argument("--s-L1K", type=float, default=0.0,
                   help="weight of the L1_K stat (= L0 sparsity coefficient with --head-sparse)")
    p.add_argument("--ref-stats", action="store_true",
                   help="use the reference (materializing) stats path")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature-end", type=float, default=0.03)
    p.add_argument("--anneal-steps", type=int, default=50000)
    p.add_argument("--sd-K", type=float, default=0.02)
    p.add_argument("--sd-K-end", type=float, default=None,
                   help="default 0.2 * temperature_end (sonar.py); <0 disables")
    p.add_argument("--sd-F", type=float, default=0.1)
    p.add_argument("--p-drop", type=float, default=0.1)
    p.add_argument("--p-drop-start", type=float, default=0.0)
    p.add_argument("--s-L1F", type=float, default=1e-9)
    p.add_argument("--s-H", type=float, default=0.0)
    p.add_argument("--s-bcossim", type=float, default=1e-5)
    p.add_argument("--s-hcossim", type=float, default=1e-6)
    p.add_argument("--s-Hm", type=float, default=1e-6)
    p.add_argument("--save-each", type=int, default=1000)
    p.add_argument("--checkpoint-each", type=int, default=10000)
    p.add_argument("--max-to-keep", type=int, default=3, help="recent checkpoints kept (besides every checkpoint-each); wide runs write ~1 GB each")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-ce", action="store_true")
    p.add_argument("--diagnose", action="store_true",
                   help="run diagnostics in-process after training (default: run "
                        "diagnose.py separately, one job at a time)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run from its latest checkpoint (same flags; full state incl. Adam and step)")
    return p.parse_args()


def main():
    cfg = parse_args()
    cache = Cache(cfg.cache)
    d = cache.d
    e_dec = d if cfg.direct else (cfg.e_dec or 2 * d)
    sd_K_end = 0.2 * cfg.temperature_end if cfg.sd_K_end is None else (
        None if cfg.sd_K_end < 0 else cfg.sd_K_end)
    out = Path(cfg.out_root) / cfg.name
    if cfg.resume:
        assert out.exists(), f"{out} does not exist; nothing to resume"
        (out / "config_resume.json").write_text(json.dumps(vars(cfg) | {
            "e_dec": e_dec, "sd_K_end": sd_K_end}, indent=2))
    else:
        if out.exists() and cfg.overwrite:
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=False)
        (out / "config.json").write_text(json.dumps(vars(cfg) | {
            "e_dec": e_dec, "sd_K_end": sd_K_end, "cache_meta": cache.meta}, indent=2))

    hyper = Hyperparams(
        d, d, cfg.b, cfg.epochs, cfg.lr, 0.0, cfg.temperature,
        "none", "normal", "featvar", 0.0, cfg.sd_K, cfg.sd_F,
        s_g=0.0, s_L1K=cfg.s_L1K, s_L1F=cfg.s_L1F, s_H=cfg.s_H,
        s_bcossim=cfg.s_bcossim, s_hcossim=cfg.s_hcossim, s_Hm=cfg.s_Hm, s_aux=cfg.s_aux,
        p_drop=cfg.p_drop, mse_weights=None,
        temperature_end=cfg.temperature_end, anneal_steps=cfg.anneal_steps,
        p_drop_start=cfg.p_drop_start, sd_K_end=sd_K_end,
        ghost=False, seed=cfg.seed)
    model = hyper.ontologizer(
        d, d, e_dec, cfg.k, cfg.h, cfg.l, n=2, gate="none", scaled=False,
        encoded=False, forward=cfg.forward, deepsup=not cfg.no_deepsup,
        deepsup_sg=cfg.deepsup_sg, resid_norm=not cfg.no_resid_norm,
        resid_const=not cfg.no_resid_const, resid_first=cfg.resid_first,
        resid_gain=cfg.resid_gain, logit_norm=cfg.logit_norm,
        select=cfg.select, fast_stats=not cfg.ref_stats,
        private_heads=cfg.private_heads, signed_dict=cfg.signed_dict,
        direct=cfg.direct, p_head_drop=cfg.p_head_drop,
        fiber_rank=cfg.fiber_rank, gain_clip=cfg.gain_clip,
        fiber_drop=cfg.fiber_drop, fiber_bound=cfg.fiber_bound,
        fiber_eps=cfg.fiber_eps, fiber_radial=cfg.fiber_radial,
        fiber_shared=cfg.fiber_shared, fiber_eps_layer=cfg.fiber_eps_layer,
        base_aux=cfg.base_aux,
        head_sparse=cfg.head_sparse, m_h=cfg.m_h, k_z=cfg.k_z, hs_shared=cfg.hs_shared,
        hs_bandwidth=cfg.hs_bandwidth, hs_init_threshold=cfg.hs_init_threshold,
        hs_decoder=cfg.hs_decoder, hs_auxk=cfg.hs_auxk,
        dtype_str="float32", dtype_p_str="float32")

    if cfg.train_only == "fiber":
        # optimizer that leaves every non-fiber parameter untouched
        import optax
        from flax import traverse_util
        from ontologize.training.ontostate import state_init

        def labels(params):
            flat = traverse_util.flatten_dict(params)
            lab = {k: ("fiber" if any(part in ("fiber", "fiber_read") for part in k)
                       else "freeze") for k in flat}
            return traverse_util.unflatten_dict(lab)
        tx = optax.multi_transform({"fiber": hyper.opt(), "freeze": optax.set_to_zero()},
                                   labels)
        state = state_init(model, cfg.b, tx, hyper.rng(), cfg.save_each,
                           hyper.n_stats, ghost=False)
    elif cfg.hs_lr_mult != 1.0:
        # same optimizer, scaled learning rate for the sparse encoders
        import optax
        from flax import traverse_util
        from ontologize.training.ontostate import state_init

        def labels(params):
            flat = traverse_util.flatten_dict(params)
            lab = {k: ("enc" if "hs_encoder" in k else "rest") for k in flat}
            return traverse_util.unflatten_dict(lab)
        hyper_enc = dataclasses.replace(hyper, lr=hyper.lr * cfg.hs_lr_mult)
        tx = optax.multi_transform({"enc": hyper_enc.opt(), "rest": hyper.opt()}, labels)
        state = state_init(model, cfg.b, tx, hyper.rng(), cfg.save_each,
                           hyper.n_stats, ghost=False)
    else:
        state = hyper.init(model, save_each=cfg.save_each)
    if cfg.init_from:
        # copy every parameter the checkpoint has into the (possibly larger) new tree
        from flax import traverse_util
        _, src, src_step = diagnose.load(cfg.init_from)
        new = traverse_util.flatten_dict(state.params)
        old = traverse_util.flatten_dict(src)
        copied = 0
        for k, v in old.items():
            if k in new and new[k].shape == v.shape:
                new[k] = v
                copied += 1
        state = state.replace(params=traverse_util.unflatten_dict(new))
        print(f"initialized {copied}/{len(new)} params from {cfg.init_from} step {src_step}"
              f" ({len(new) - copied} fresh)", flush=True)
    manager = Metadata(out_path=out, save_each=cfg.save_each,
                       checkpoint_each=cfg.checkpoint_each, max_to_keep=cfg.max_to_keep).manager()
    steps = cfg.epochs * cache.steps_per_epoch(cfg.b)
    if cfg.max_steps:
        steps = min(steps, cfg.max_steps)
    start = 0
    if cfg.resume:
        from ontologize.training.ontostate import load_params
        step = manager.latest_step()
        state = load_params(state, manager, step)      # params, optimizer state and step
        start = int(state.step)
        print(f"resumed {cfg.name} from step {start}", flush=True)
    # the loader is deterministic in (b, epochs, seed): skip the consumed batches
    dat = itertools.islice(cache.train_batches(cfg.b, cfg.epochs, cfg.seed), start, steps)
    print(f"{cfg.name}: {steps - start} steps of b={cfg.b} "
          f"({(steps - start) * cfg.b / cache.n_train:.2f} epochs) from step {start}", flush=True)

    hook = None
    if cfg.resample_period and cfg.head_sparse != "none":
        from ontologize.training.ontostate import schedules
        from resample import Resampler
        T_of = lambda step: schedules(step, hyper.anneal_steps, hyper.temperature, hyper.temperature_end,
                                      hyper.p_drop, hyper.p_drop_start, hyper.sd_K, hyper.sd_K_end)[0]
        hook = Resampler(model, T_of, period=cfg.resample_period, count_every=cfg.resample_count_every,
                         scale=cfg.resample_scale, seed=cfg.seed,
                         log=lambda m: print(m, flush=True))
    t0 = time.time()
    state = hyper.train(state, dat, manager, save_each=cfg.save_each, hook=hook)
    train_s = time.time() - t0
    tail = int(state.step) % cfg.save_each
    if tail:  # train() only flushes whole save_each blocks
        # force: the manager otherwise skips steps off its save interval
        manager.save(int(state.step), items={"state": state, "spec": state.spec()},
                     force=True)
        manager.wait_until_finished()
        import csv
        import numpy as np
        with open(out / "loss.csv", "a", newline="") as f:
            csv.writer(f).writerows(np.asarray(state.stats)[:tail])

    (out / "train_seconds.txt").write_text(f"{train_s:.1f}\n")
    if cfg.diagnose:
        # in-process diagnostics: only safe when no other GPU job is running
        # (JAX keeps its pool; the splice eval then needs room for GPT-2)
        X = cache.eval_rows(65536)
        s = diagnose.summarize(model, state.params, X, cfg.temperature_end)
        s["step"] = int(state.step)
        s["train_seconds"] = train_s
        if not cfg.no_ce:
            s["ce"] = diagnose.splice_ce(model, state.params, cache, cfg.temperature_end)
        diagnose.print_summary(s, s.get("ce"))
        (out / f"diag_{int(state.step)}.json").write_text(json.dumps(s, indent=2))
    print(f"train {train_s / 60:.1f} min")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

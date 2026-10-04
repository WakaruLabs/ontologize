"""Planted factorial-code recovery for a single DictEnc layer.

The layer's forward pass is itself a generative model: one categorical
choice per head, one non-negative dictionary row per choice, summed over
heads and pushed through the shared linear decoder. This harness plants
exactly that structure -- h_true independent factors with k_true levels
each, one random atom per level, x = sum_h M_h[z_h] + noise -- trains a
one-layer Ontologizer on it (l=1: no residual forwarding, no deep
supervision, passthrough encoder) with the stock training step and
schedules, and scores the result against the planted truth:

  fvu        held-out MSE / Var(x), next to the oracle (the planted
             decomposition's own FVU, i.e. the noise floor) and to the
             one-hot ("hard") code's FVU
  acc        per true factor: the best-matched model head's accuracy
             under the best entry bijection (Hungarian on the k x k
             contingency; chance ~ 1/k_true)
  nmi        that pairing's normalized mutual information: the
             head-matching statistic (symmetric, defined for
             k != k_true, and unlike acc it does not punish a head that
             splits one level over two entries)
  atom_cos   cosine between the matched decoded entries and the planted
             atoms, both centered within the head (a per-head shift is
             a symmetry of the sum, so only centered atoms are
             identifiable)
  extras     model heads left unmatched (h > h_true): best NMI to any
             factor (redundant if high) and normalized argmax-usage
             entropy (frozen if ~0)
  dead       entries never selected on the eval set; usage KL_m,
             per-sample entropy and the within-head row cosine as the
             training stats define them

Every score is also reported for the untrained model and under a
shuffled-label null (Z rows permuted, matching redone), which is the
chance level of the Hungarian statistics at this sample size.

Defaults put the data exactly in the model class (raw magnitudes,
resid_gain off, e_dec >= h_true*k_true so an exact solution exists:
decoder columns = atoms, one-hot rows). Arms are single-knob flags:

  uv run python experiments/dictenc-toy/toy_dictenc.py   # h, k = true
  ... --h 8                  # extra heads: die, duplicate, or split?
  ... --k 16                 # extra entries per head
  ... --select top1          # hard selection (pair with --p-revive)
  ... --antipodal --no-const # bilinear sign-blindness: levels planted in
                             #   (+m, -m) pairs, which an even classifier
                             #   cannot separate; compare against --antipodal
  ... --no-anneal            # T fixed at 1: soft codes
  ... --unit                 # SONAR regime: unit-norm inputs, gain-shape
                             #   split; the model class is then approximate
  ... --sigma 1.0            # harder noise floor
  ... --seeds 5              # replicates (model and data seeds advance
                             #   together) with a tally at the end

CPU by default (JAX_PLATFORMS=cpu unless already set), so it can run
beside a live GPU training run; a default run takes about half a
minute per seed. Writes <out>/summary.json, <out>/heads.csv and
<out>/loss_s<seed>.csv.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import linear_sum_assignment

LOSS_COLS = ["loss", "MSE", "MSE_ghost", "L1_K", "L1_F", "entropy",
             "cossim_b", "cossim_h", "cossim_k", "cossim_k_max", "KL_m",
             "KL_pwak", "L2_pwak", "s_L1F", "s_kcossim", "kcos_max"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("planted code")
    g.add_argument("--h-true", type=int, default=4)
    g.add_argument("--k-true", type=int, default=8)
    g.add_argument("--d", type=int, default=64)
    g.add_argument("--sigma", type=float, default=0.3,
                   help="noise sd per coordinate; atoms are N(0, 1/h_true) "
                        "per coordinate, so the clean signal has about unit "
                        "variance per coordinate")
    g.add_argument("--center", action="store_true",
                   help="zero-mean atoms within each factor, removing the "
                        "corpus mean an unbiased bilinear classifier leans on")
    g.add_argument("--antipodal", action="store_true",
                   help="plant levels in (+m, -m) pairs within each factor "
                        "(k_true must be even). An even classifier cannot "
                        "separate a pair, so this is the sharp test of the "
                        "constant coordinate; every factor is then exactly "
                        "zero-mean")
    g.add_argument("--unit", action="store_true",
                   help="unit-normalize samples and classify with the "
                        "gain-shape split (the SONAR input regime)")
    g.add_argument("--n-train", type=int, default=32768)
    g.add_argument("--n-eval", type=int, default=8192)
    g.add_argument("--data-seed", type=int, default=0)
    m = p.add_argument_group("model")
    m.add_argument("--h", type=int, default=None, help="heads (h_true)")
    m.add_argument("--k", type=int, default=None,
                   help="entries per head (k_true)")
    m.add_argument("--e-dec", type=int, default=128)
    m.add_argument("--select", default="softmax",
                   help="DictBlock.select: softmax, ste, or top<k>")
    m.add_argument("--n", type=int, default=2, help="classifier order")
    m.add_argument("--gate", default="none")
    m.add_argument("--no-const", action="store_true",
                   help="drop resid_const's constant classifier coordinate")
    t = p.add_argument_group("training")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--anneal-steps", type=int, default=3000)
    t.add_argument("--b", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument("--temperature-end", type=float, default=0.03)
    t.add_argument("--no-anneal", action="store_true",
                   help="hold every schedule at its starting value")
    t.add_argument("--p-drop", type=float, default=0.1)
    t.add_argument("--p-revive", type=float, default=0.0)
    t.add_argument("--sd-k", type=float, default=0.02)
    t.add_argument("--sd-f", type=float, default=0.1)
    t.add_argument("--s-h", type=float, default=0.0,
                   help="per-sample classification entropy penalty")
    t.add_argument("--entropy-target", type=float, default=0.0,
                   help="setpoint control for s_H, the counterpart of the "
                        "package's L1F_target: hold the per-head code "
                        "entropy (bits) at a setpoint ramped down from "
                        "wherever the run starts to this value, instead of "
                        "fixing --s-h. 0 = off. A fixed weight has no "
                        "equilibrium here -- the MSE gradient shrinks "
                        "toward the noise floor while the entropy gradient "
                        "does not -- so the same argument that motivates "
                        "L1F_target applies")
    t.add_argument("--entropy-eta", type=float, default=1e-2,
                   help="controller gain, a step in log(s) per unit "
                        "log-error. The package uses 1e-3 against a "
                        "50k-step anneal; this toy anneals in 3k")
    t.add_argument("--entropy-ema", type=float, default=0.99,
                   help="EMA horizon for the measured entropy")
    t.add_argument("--entropy-ramp", type=int, default=0,
                   help="steps over which the setpoint ramps to the "
                        "target (0 = --anneal-steps)")
    t.add_argument("--entropy-min", type=float, default=1e-6)
    t.add_argument("--entropy-max", type=float, default=1.0)
    t.add_argument("--entropy-no-project", action="store_true",
                   help="apply the unprojected dual variable instead of "
                        "`dual_apply`'s projection to zero. The projection "
                        "is complementary slackness, right for a one-way "
                        "plant like L1_F where every overshoot is "
                        "permanent; entropy is driven down by the "
                        "temperature anneal for free, so the constraint is "
                        "slack through most of training and the projected "
                        "penalty never acts while the partition is still "
                        "forming")
    t.add_argument("--s-hm", type=float, default=0.0)
    t.add_argument("--s-bcossim", type=float, default=0.0)
    t.add_argument("--s-hcossim", type=float, default=0.0)
    t.add_argument("--s-kcossim", type=float, default=0.0)
    t.add_argument("--s-l1f", type=float, default=0.0)
    t.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, default=1,
                   help="replicates; the model seed and the data seed "
                        "advance together from --seed and --data-seed")
    p.add_argument("--nulls", type=int, default=3,
                   help="shuffled-label null draws")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default=None,
                   help="default experiments/dictenc-toy/out/<name>")
    cfg = p.parse_args()
    cfg.h = cfg.h or cfg.h_true
    cfg.k = cfg.k or cfg.k_true
    if cfg.antipodal and cfg.k_true % 2:
        p.error("--antipodal needs an even --k-true")
    if cfg.seeds < 1:
        p.error("--seeds must be at least 1")
    if cfg.entropy_target > 0:
        if cfg.s_kcossim:
            p.error("--entropy-target borrows the s_kcossim coefficient "
                    "slot (see hyper_class), so the two cannot both be "
                    "applied in one run")
        # the gradient gate is static: `Hyperparams.s_loss` turns
        # `entropy_loss` on only when `s_H` is nonzero at construction,
        # so the controller needs a nonzero starting value
        cfg.s_h = cfg.s_h or 1e-3
    if cfg.name is None:
        cfg.name = (f"t{cfg.h_true}x{cfg.k_true}_m{cfg.h}x{cfg.k}_{cfg.select}"
                    + ("_noconst" if cfg.no_const else "")
                    + ("_center" if cfg.center else "")
                    + ("_antipodal" if cfg.antipodal else "")
                    + ("_unit" if cfg.unit else "")
                    + ("_noanneal" if cfg.no_anneal else "")
                    + (f"_eH{cfg.entropy_target:g}"
                       if cfg.entropy_target else "")
                    + (f"_x{cfg.seeds}" if cfg.seeds > 1 else ""))
    return cfg


# ---------- planted code ----------

def plant(rng, h, k, d, center=False, antipodal=False):
    """One atom per (factor, level), N(0, 1/h) per coordinate so the sum
    over factors has about unit variance per coordinate. Antipodal
    planting draws half the levels and takes the other half as their
    negatives, so level j and level j + k/2 differ only in sign and every
    factor is exactly zero-mean."""
    if antipodal:
        half = rng.normal(0.0, 1.0 / np.sqrt(h), (h, k // 2, d))
        M = np.concatenate([half, -half], axis=1)
    else:
        M = rng.normal(0.0, 1.0 / np.sqrt(h), (h, k, d))
    if center:
        M = M - M.mean(1, keepdims=True)
    return M.astype(np.float32)


def sample(rng, M, n, sigma):
    """n samples of x = sum_h M_h[z_h] + sigma * eps with uniform
    independent levels. Returns (X (n, d), Z (n, h), X_clean (n, d))."""
    h, k, d = M.shape
    Z = rng.integers(0, k, (n, h))
    X_clean = M[np.arange(h)[None, :], Z].sum(1)
    X = X_clean + sigma * rng.normal(size=(n, d))
    return X.astype(np.float32), Z, X_clean.astype(np.float32)


def unit_rows(A, eps=1e-9):
    return A / (np.linalg.norm(A, axis=-1, keepdims=True) + eps)


# ---------- matching ----------

def contingency(a, b, ka, kb):
    C = np.zeros((ka, kb), int)
    np.add.at(C, (a, b), 1)
    return C


def bijection_acc(C):
    """Accuracy under the best one-to-one relabeling (rectangular allowed:
    unmatched rows or columns count as errors)."""
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / C.sum(), (r, c)


def purity(C):
    """Fraction of samples whose model entry's majority level is theirs:
    1 when the model partition refines the true one."""
    return C.max(1).sum() / C.sum()


def nmi(C):
    N = C.sum()
    pa, pb, pab = C.sum(1) / N, C.sum(0) / N, C / N

    def H(p):
        p = p[p > 0]
        return -(p * np.log(p)).sum()

    nz = pab > 0
    I = (pab[nz] * np.log(pab[nz] / np.outer(pa, pb)[nz])).sum()
    denom = 0.5 * (H(pa) + H(pb))
    return float(I / denom) if denom > 0 else 0.0


def head_entropy(P):
    """Per-head mean per-sample entropy of the classification, bits."""
    eps = np.finfo(P.dtype).eps
    return -(P * np.log2(P + eps)).sum(-1).mean(0)


def score(A, Z, entries, atoms, H_head=None):
    """A (n, h_model) argmax entries, Z (n, h_true) true levels,
    entries (h_model, k_model, d) decoded dictionary rows, atoms
    (h_true, k_true, d), H_head (h_model,) per-head code entropy.
    Matches heads to factors by Hungarian on NMI, then entries within
    each pair; returns per-factor rows and the unmatched heads."""
    h_m, h_t = A.shape[1], Z.shape[1]
    if H_head is None:
        H_head = np.full(h_m, np.nan)
    k_m, k_t = entries.shape[1], atoms.shape[1]
    S = np.zeros((h_m, h_t))
    Cs = {}
    for a in range(h_m):
        for b in range(h_t):
            C = contingency(A[:, a], Z[:, b], k_m, k_t)
            Cs[a, b] = C
            S[a, b] = nmi(C)
    ra, rb = linear_sum_assignment(-S)
    rows = []
    for a, b in zip(ra, rb):
        C = Cs[a, b]
        acc, (ei, ej) = bijection_acc(C)
        Ea = entries[a][ei]
        Mb = atoms[b][ej]
        cos = (unit_rows(Ea - Ea.mean(0)) * unit_rows(Mb - Mb.mean(0))).sum(-1)
        rows.append({"factor": int(b), "head": int(a), "nmi": float(S[a, b]),
                     "acc": float(acc), "purity": float(purity(C)),
                     "atom_cos": float(cos.mean()),
                     "entropy": float(H_head[a])})
    rows.sort(key=lambda r: r["factor"])
    n = A.shape[0]
    extras = []
    for a in sorted(set(range(h_m)) - set(ra.tolist())):
        u = np.bincount(A[:, a], minlength=k_m) / n
        u = u[u > 0]
        ent = float(-(u * np.log2(u)).sum() / np.log2(k_m))
        extras.append({"head": int(a), "max_nmi": float(S[a].max()),
                       "usage_entropy": ent, "entropy": float(H_head[a])})
    return rows, extras


def summarize(rows, extras):
    accs = np.array([r["acc"] for r in rows])
    return {"acc_mean": float(accs.mean()), "acc_min": float(accs.min()),
            "nmi_mean": float(np.mean([r["nmi"] for r in rows])),
            "purity_mean": float(np.mean([r["purity"] for r in rows])),
            "atom_cos_mean": float(np.mean([r["atom_cos"] for r in rows])),
            "factors_recovered": int((accs >= 0.9).sum()),
            "extras": extras}


# ---------- model ----------

_HYPER_CLS = None


def hyper_class():
    """`Hyperparams` whose second traced-coefficient slot carries `s_H`
    rather than `s_kcossim`.

    `update` threads exactly two per-step coefficients into the jitted
    loss, because those are the two the package controls with a
    `DualLoop`. An entropy setpoint needs a third. `s_kcossim` is
    unidentified on this toy (a free decoder makes dense and one-hot
    dictionary rows equally exact, so the recovered solutions sit at
    row cosine 0.45 to 0.6), so the slot is borrowed rather than the
    package changed -- the move `experiments/hsic-bottleneck` makes
    with the inert ghost slot. The objective is linear in the
    coefficient, so exchanging the static weight for the controlled one
    is a single correction term, and the applied value is logged in the
    column the slot owns.

    Built once and cached: the class is the jit-static `lossfn`'s owner,
    and it is defined lazily only to keep `--help` free of JAX."""
    global _HYPER_CLS
    if _HYPER_CLS is None:
        import jax.numpy as jnp
        from ontologize.training.config import Hyperparams

        class EntropyHyperparams(Hyperparams):
            def loss(self, X, Y, X_g, stats, s_L1F=None, s_H=None):
                L, row = super().loss(X, Y, X_g, stats, s_L1F, None)
                if s_H is None:
                    return L, row
                s_H = jnp.asarray(s_H, L.dtype)
                # row[5] is the entropy stat summed over layers, which is
                # exactly what s[3] multiplies inside the parent
                L = L + (s_H - self.s_H) * row[5]
                return L, row.at[0].set(L).at[14].set(s_H)

        _HYPER_CLS = EntropyHyperparams
    return _HYPER_CLS


def build(cfg, seed):
    Hyperparams = hyper_class()
    hyper = Hyperparams(
        cfg.d, cfg.d, cfg.b, epochs=1, lr=cfg.lr, wd=0.0,
        temperature=cfg.temperature,
        noise_in="batchnorm", noise_K="normal", noise_F="featvar",
        sd_in=0.0, sd_K=cfg.sd_k, sd_F=cfg.sd_f,
        p_drop=cfg.p_drop, p_revive=cfg.p_revive,
        s_L1F=cfg.s_l1f, s_bcossim=cfg.s_bcossim, s_hcossim=cfg.s_hcossim,
        s_H=cfg.s_h, s_kcossim=cfg.s_kcossim, s_Hm=cfg.s_hm,
        seed=seed, ghost=False,
        temperature_end=None if cfg.no_anneal else cfg.temperature_end,
        anneal_steps=0 if cfg.no_anneal else cfg.anneal_steps,
        p_drop_start=None if cfg.no_anneal else 0.0,
        sd_K_end=None if cfg.no_anneal else 0.2 * cfg.temperature_end)
    model = hyper.ontologizer(
        cfg.d, cfg.d, cfg.e_dec, cfg.k, cfg.h, 1,
        n=cfg.n, gate=cfg.gate, select=cfg.select,
        forward="resid", deepsup=False,
        resid_norm=False, resid_const=not cfg.no_const, resid_gain=cfg.unit,
        dtype_str="float32", dtype_p_str="float32")
    state = hyper.init(model, save_each=cfg.steps)
    return hyper, model, state


def run_model(model, params, X, T):
    """Soft decode at temperature T, the one-hot decode of the same
    classification, the classification itself (b, h, k), and the raw
    classifier logits (b, h, k) before temperature."""
    import jax
    import jax.numpy as jnp

    def probe(module, X):
        E, _ = module.encode(X, 0.0, None)
        E = module.constinput(E)
        de = module.dictencs[0]
        U, G = de.gainshape_in(E)
        K = de.classifier(U)
        P = de.dict.cluster(K, T)
        Y = module(X, temperature=T)
        P_hard = jax.nn.one_hot(jnp.argmax(P, -1), P.shape[-1], dtype=P.dtype)
        Y_hard = module.decode(de.gained(de.dict.fwd(P_hard), G))
        return Y, Y_hard, P, K

    Y, Y_hard, P, K = model.apply(params, jnp.asarray(X), method=probe)
    return np.asarray(Y), np.asarray(Y_hard), np.asarray(P), np.asarray(K)


def decoded_entries(model, params):
    from ontologize.ontologizer import Ontologizer
    R, _ = model.apply(params, method=Ontologizer.decodeEntries)
    return np.asarray(R).reshape(model.h, model.k, model.d_out)


def row_cos(model, params):
    return np.asarray(model.apply(
        params, method=lambda m: m.dictencs[0].dict.rowcos()))


def code_stats(P):
    """Per-sample entropy (bits, mean over heads), KL(E[p] || uniform)
    (bits, mean over heads), and the count of never-selected entries."""
    k = P.shape[-1]
    eps = np.finfo(P.dtype).eps
    H = -(P * np.log2(P + eps)).sum(-1).mean()
    Pm = P.mean(0)
    KLm = (np.log2(k) + (Pm * np.log2(Pm + eps)).sum(-1)).mean()
    A = P.argmax(-1)
    used = np.zeros(P.shape[1:], bool)
    for a in range(P.shape[1]):
        used[a, np.unique(A[:, a])] = True
    return float(H), float(KLm), int((~used).sum())


def evaluate(cfg, model, params, X, Z, X_clean, atoms, T, nulls_rng):
    Y, Y_hard, P, K = run_model(model, params, X, T)
    var = ((X - X.mean(0)) ** 2).mean()
    A = P.argmax(-1)
    entries = decoded_entries(model, params)
    H_head = head_entropy(P)
    rows, extras = score(A, Z, entries, atoms, H_head)
    H, KLm, dead = code_stats(P)
    # the within-head logit spread, in units of the eval temperature: the
    # softmax is hard only when this is large, and a scale-free
    # classifier can shrink it as fast as the temperature falls
    spread = float(K.std(-1).mean() / T)
    res = {"fvu": float(((Y - X) ** 2).mean() / var),
           "fvu_hard": float(((Y_hard - X) ** 2).mean() / var),
           "fvu_oracle": float(((X_clean - X) ** 2).mean() / var),
           "entropy_bits": H, "KL_m": KLm, "dead_entries": dead,
           "logit_spread_over_T": spread,
           "cossim_k_max": float(row_cos(model, params).max()),
           "heads": rows, **summarize(rows, extras)}
    null = []
    for _ in range(cfg.nulls):
        Zs = Z[nulls_rng.permutation(len(Z))]
        null.append(summarize(*score(A, Zs, entries, atoms, H_head)))
    if null:
        res["null"] = {key: float(np.mean([n[key] for n in null]))
                       for key in ("acc_mean", "nmi_mean", "atom_cos_mean")}
    return res


def sh_now(cfg, sh):
    """The entropy coefficient to apply this step: the projected dual
    variable, or the raw one under `--entropy-no-project`. `None` when
    the controller is off, which keeps the traced/untraced argument
    shape constant across steps."""
    if not sh.on:
        return None
    return sh.s if cfg.entropy_no_project else sh.applied


def train(cfg, hyper, state, X_tr, seed):
    import jax
    import jax.numpy as jnp
    from ontologize.training.ontostate import update, schedules, DualLoop
    rng = hyper.rng()
    order = np.random.default_rng(seed)
    n_batches = len(X_tr) // cfg.b
    perm = order.permutation(len(X_tr))
    # the package's own setpoint controller, watching the code entropy
    # (stats row 5) and driving the borrowed coefficient slot. `dual_apply`
    # holds it at exactly 0 while the measurement is under the ramped
    # setpoint, so the penalty is a thermostat: it engages only where the
    # temperature anneal fails to deliver hardness on its own.
    sh = DualLoop(hyper.s_H, cfg.entropy_target, cfg.entropy_eta,
                  cfg.entropy_ema, cfg.entropy_min, cfg.entropy_max,
                  cfg.entropy_ramp or hyper.anneal_steps or cfg.steps, 0)
    t0 = time.time()
    for step in range(cfg.steps):
        if step and step % n_batches == 0:
            perm = order.permutation(len(X_tr))
        i = (step % n_batches) * cfg.b
        Xb = jnp.asarray(X_tr[perm[i:i + cfg.b]])
        T, pd, sk, _ = schedules(
            step, hyper.anneal_steps, hyper.temperature,
            hyper.temperature_end, hyper.p_drop, hyper.p_drop_start,
            hyper.sd_K, hyper.sd_K_end)
        rng, r = jax.random.split(rng)
        state, L, _ = update(
            state, hyper.loss, r, Xb, Xb, temperature=T, p_drop=pd,
            sd_K=sk, sd_in=0.0, sd_F=hyper.sd_F, grad_clip=hyper.grad_clip,
            s_kcossim=sh_now(cfg, sh),
            p_revive=hyper.p_revive, revive_frac=hyper.revive_frac)
        row = state.last_stats()
        if sh.on:
            sh.step(int(state.step), float(row[5]))
        if step % 500 == 0 or step == cfg.steps - 1:
            msg = (f"step {step:5d}  loss {float(L):.4f}  "
                   f"MSE {float(row[1]):.4f}  H {float(row[5]):.2f}b  "
                   f"KL_m {float(row[10]):.3f}b  T {T:.3f}")
            if sh.on:
                msg += f"  H_tgt {sh.tgt:.3f}b  s_H {sh_now(cfg, sh):.2e}"
            print(msg + f"  ({time.time() - t0:.0f}s)", flush=True)
    return state


def print_report(tag, res):
    print(f"\n== {tag} ==")
    print(f"fvu {res['fvu']:.4f}  hard {res['fvu_hard']:.4f}  "
          f"oracle {res['fvu_oracle']:.4f}  |  entropy {res['entropy_bits']:.3f}b  "
          f"KL_m {res['KL_m']:.3f}b  dead {res['dead_entries']}  "
          f"logit spread/T {res['logit_spread_over_T']:.2f}  "
          f"cossim_k_max {res['cossim_k_max']:.3f}")
    print(f"{'factor':>6} {'head':>4} {'nmi':>6} {'acc':>6} {'purity':>6} "
          f"{'atom_cos':>8} {'H_bits':>6}")
    for r in res["heads"]:
        print(f"{r['factor']:>6} {r['head']:>4} {r['nmi']:>6.3f} "
              f"{r['acc']:>6.3f} {r['purity']:>6.3f} {r['atom_cos']:>8.3f} "
              f"{r['entropy']:>6.2f}")
    for e in res["extras"]:
        print(f"  extra head {e['head']}: max nmi {e['max_nmi']:.3f}, "
              f"usage entropy {e['usage_entropy']:.3f}, "
              f"H {e['entropy']:.2f}b")
    line = (f"recovered {res['factors_recovered']}/{len(res['heads'])} "
            f"(acc >= 0.9)  acc {res['acc_mean']:.3f}  nmi {res['nmi_mean']:.3f}  "
            f"atom_cos {res['atom_cos_mean']:.3f}")
    if "null" in res:
        n = res["null"]
        line += (f"  |  null: acc {n['acc_mean']:.3f}  nmi {n['nmi_mean']:.3f}  "
                 f"atom_cos {n['atom_cos_mean']:.3f}")
    print(line)


def replicate(cfg, seed, data_seed, out, first):
    """One planted code, one model, trained and scored. The untrained
    report is printed for the first replicate only."""
    data = np.random.default_rng(data_seed)
    atoms = plant(data, cfg.h_true, cfg.k_true, cfg.d, cfg.center,
                  cfg.antipodal)
    X_tr, _, _ = sample(data, atoms, cfg.n_train, cfg.sigma)
    X_ev, Z_ev, C_ev = sample(data, atoms, cfg.n_eval, cfg.sigma)
    if cfg.unit:
        n_tr = np.linalg.norm(X_tr, axis=-1, keepdims=True)
        n_ev = np.linalg.norm(X_ev, axis=-1, keepdims=True)
        X_tr, X_ev, C_ev = X_tr / n_tr, X_ev / n_ev, C_ev / n_ev

    hyper, model, state = build(cfg, seed)
    T_eval = hyper.temperature if cfg.no_anneal else hyper.temperature_end
    nulls = np.random.default_rng(seed + 1)
    before = evaluate(cfg, model, state.params, X_ev, Z_ev, C_ev, atoms,
                      T_eval, nulls)
    if first:
        print_report(f"untrained (seed {seed}, data seed {data_seed})",
                     before)

    state = train(cfg, hyper, state, X_tr, seed)
    after = evaluate(cfg, model, state.params, X_ev, Z_ev, C_ev, atoms,
                     T_eval, nulls)
    print_report(f"trained (seed {seed}, data seed {data_seed})", after)

    with open(out / f"loss_s{seed}.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = list(LOSS_COLS)
        if cfg.entropy_target:
            cols[14] = "s_H_applied"   # the borrowed slot; see hyper_class
        w.writerow(cols)
        w.writerows(np.asarray(state.stats)[:cfg.steps])
    return {"seed": seed, "data_seed": data_seed,
            "untrained": before, "trained": after}


TALLY_KEYS = ("acc_mean", "nmi_mean", "purity_mean", "atom_cos_mean",
              "entropy_bits", "fvu", "fvu_hard", "fvu_oracle", "KL_m",
              "dead_entries", "logit_spread_over_T")


def tally(results):
    """Across replicates: the per-seed recovered counts, how many seeds
    recovered every factor, and mean and std of each trained scalar."""
    T = [r["trained"] for r in results]
    n_factors = len(T[0]["heads"])
    rec = [t["factors_recovered"] for t in T]
    out = {"n": len(T), "n_factors": n_factors, "recovered": rec,
           "exact": int(sum(r == n_factors for r in rec))}
    for key in TALLY_KEYS:
        v = np.array([t[key] for t in T], float)
        out[key] = {"mean": float(v.mean()), "std": float(v.std())}
    if "null" in T[0]:
        out["null"] = {key: float(np.mean([t["null"][key] for t in T]))
                       for key in T[0]["null"]}
    return out


def print_tally(tl):
    def f(key):
        return f"{tl[key]['mean']:.3f}+-{tl[key]['std']:.3f}"

    print(f"\n== tally over {tl['n']} seeds ==")
    print(f"recovered {' '.join(str(r) for r in tl['recovered'])} "
          f"of {tl['n_factors']};  exact {tl['exact']}/{tl['n']}")
    print(f"acc {f('acc_mean')}  nmi {f('nmi_mean')}  "
          f"purity {f('purity_mean')}  atom_cos {f('atom_cos_mean')}  "
          f"H {f('entropy_bits')}b  logit spread/T {f('logit_spread_over_T')}")
    print(f"fvu {f('fvu')}  hard {f('fvu_hard')}  oracle {f('fvu_oracle')}  "
          f"KL_m {f('KL_m')}b  dead {f('dead_entries')}")
    if "null" in tl:
        n = tl["null"]
        print(f"null: acc {n['acc_mean']:.3f}  nmi {n['nmi_mean']:.3f}  "
              f"atom_cos {n['atom_cos_mean']:.3f}")


def main():
    cfg = parse_args()
    out = Path(cfg.out) if cfg.out else ROOT / "experiments/dictenc-toy/out" / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    if cfg.e_dec < cfg.h_true * cfg.k_true:
        print(f"note: e_dec {cfg.e_dec} < h_true*k_true "
              f"{cfg.h_true * cfg.k_true}: an exact solution is not "
              f"guaranteed to exist")
    print(f"{cfg.name}: planted {cfg.h_true} factors x {cfg.k_true} levels "
          f"in d={cfg.d}, sigma={cfg.sigma}"
          f"{', antipodal' if cfg.antipodal else ''}; model h={cfg.h} "
          f"k={cfg.k} e_dec={cfg.e_dec} select={cfg.select}; "
          f"{cfg.steps} steps x {cfg.seeds} seed(s)")

    results = [replicate(cfg, cfg.seed + i, cfg.data_seed + i, out, i == 0)
               for i in range(cfg.seeds)]
    tl = tally(results)
    if cfg.seeds > 1:
        print_tally(tl)

    with open(out / "heads.csv", "w", newline="") as f:
        fields = ["seed", "data_seed"] + list(results[0]["trained"]["heads"][0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            for row in r["trained"]["heads"]:
                w.writerow({"seed": r["seed"], "data_seed": r["data_seed"],
                            **row})
    (out / "summary.json").write_text(json.dumps(
        {"config": vars(cfg), "seeds": results, "tally": tl}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

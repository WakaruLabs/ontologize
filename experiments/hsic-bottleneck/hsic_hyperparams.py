"""HSICHyperparams: an Hyperparams subclass that swaps the four live
auxiliary loss terms (s_L1F, s_bcossim, s_hcossim, s_Hm) for HSIC-based
penalties WITHOUT modifying the ontologize package.

Mechanism. The stock training step (ontostate.update) calls
`state.apply_fn(params, X, ...) -> (Y, Y_g, stats, rng)` and then
`lossfn(Y, Y_0, Y_g, stats)`. The ghost slot `Y_g` is unused on the live
runs (ghost=False wires it to None), so this subclass:

  1. builds `apply_fn` from a custom probe (`probe_with_codes`) that
     replicates `Ontologizer.withStats` exactly -- same encode, same
     `DictEnc.withStats` calls (noise, winner dropout, temperature all
     flow through kwargs unchanged), same deepsup prefix decodes -- but
     returns the stacked per-layer classifications (l, b, h*k) in the
     ghost slot;
  2. overrides `loss` to consume that tensor: base (whitened) MSE, plus
     `s_hsic_heads` * mean pairwise-CKA between heads (per layer,
     summed over layers -- the layer-summing convention of the stock
     aux stats) and optionally `s_hsic_res` * HSIC(residual, target);
  3. keeps the 9-column stats layout: the raw (unweighted) HSIC penalty
     is recorded in the third column, the one the stock pipeline labels
     MSE_ghost. loss.csv/plot_loss keep working; that column just
     changes meaning for these runs.

Everything else (schedules, checkpointing, TrainingEnv, resume) is
inherited: `TrainingEnv(model, HSICHyperparams(...), meta)` works as a
drop-in because `init`/`loss` are the only overridden entry points.

The stock aux terms remain available through the inherited s_* fields,
so one class expresses all three ablation arms (aux / hsic / both).
"""
import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import OntoState
from ontologize.fns.keys import get_dtype, get_loss

try:
    import hsic
except ImportError:  # imported from outside this directory
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hsic

_W_CACHE = {}


def _mse_w(path):
    """Per-dim MSE weight vector, cached (constant under jit); local twin
    of config._mse_weights so we don't lean on a private helper."""
    if path not in _W_CACHE:
        _W_CACHE[path] = jnp.asarray(np.load(path))
    return _W_CACHE[path]


def probe_with_codes(module, X, sd_in=0.0, rng=None, *args, **kwargs):
    """`Ontologizer.withStats` with the per-layer classifications kept.

    Returns (Y, P_all, stats, rng): Y as withStats (per-prefix stack
    under deepsup), P_all (l, b, h*k) the flattened post-noise/dropout
    classifications each layer actually decoded with, stats (l, 6).
    The body mirrors ontologize/ontologizer.py::withStats line for line;
    kwargs carry temperature / sd_K / sd_F / p_drop exactly as there."""
    E, rng_K = module.encode(X, sd_in, rng)
    R = module.resid(E)
    stats, Ys, Ks = [], [], []
    for i, dictenc in enumerate(module.dictencs):
        R_0 = R
        R, K, stat, rng_K = dictenc.withStats(R, E, rng=rng_K,
                                              *args, **kwargs)
        stats.append(stat)
        Ks.append(K)
        if module.deepsup:
            Ys.append(module.decode(module.prefix(R, R_0)))
        if i < module.l - 1:
            E = module.nextinput(X, R, K)

    Y = jnp.stack(Ys) if module.deepsup else module.decode(R)
    return Y, jnp.stack(Ks), jnp.stack(stats), rng_K


@dataclass
class HSICHyperparams(Hyperparams):
    """Hyperparams + HSIC penalties. New fields:

    s_hsic_heads   scale on the mean pairwise-CKA between heads (per
                   layer, summed over layers). Replaces the head/sample
                   decorrelation aux terms (s_hcossim / s_bcossim) with
                   one dependence measure; note it does NOT punish
                   frozen heads (constant code -> ~0 centered gram).
    s_hsic_res     scale on HSIC(target - decode, target): dependence
                   left in the reconstruction residual. 0 disables.
    hsic_h         heads per layer (needed to unflatten P; must match
                   the model's h when s_hsic_heads > 0).
    hsic_sigma2    fixed RBF bandwidth^2 for the head grams; 0 = median
                   heuristic per head per batch.
    hsic_estimator biased | unbiased, for the head term and the residual
                   term alike. The biased estimator carries a floor that
                   independent variables alone produce, and at a
                   training batch that floor can exceed the dependence
                   being measured -- which leaves the penalty nothing to
                   descend but the bias. Unbiased is the default.
    """
    s_hsic_heads: float = 0.0
    s_hsic_res: float = 0.0
    hsic_h: int = 0
    hsic_sigma2: float = 0.0
    hsic_estimator: str = "unbiased"

    def init(self, model, save_each: int = 1000) -> OntoState:
        """As Hyperparams.init/state_init, but apply_fn runs
        `probe_with_codes` so the loss sees the per-layer codes."""
        x = jnp.zeros((self.b, model.d_in), get_dtype(model.dtype_str))
        params = model.init(self.rng(), x)
        f = functools.partial(model.apply, method=probe_with_codes)
        state = OntoState.create(
            model=model, b=self.b, save_each=save_each,
            n_stats=self.n_stats, apply_fn=f, params=params, tx=self.opt())
        return state.replace(step=0, stats=state.newstats())

    def loss(self, X, Y, P_all, stats):
        """X: decode(s) (b, d) or (l, b, d) under deepsup. Y: target.
        P_all: (l, b, h*k) from the probe (rides in the ghost slot).
        Returns (L, stats9) with the raw HSIC penalty in column 3."""
        f = get_loss(self.lossfn)
        s, _ = self.s_loss()                      # 7: [s_g, ...6 aux...]
        stats6 = jnp.einsum("...s -> s", stats)   # sum stats over layers
        if self.mse_weights:
            rw = jnp.sqrt(_mse_w(self.mse_weights)).astype(X.dtype)
            X, Y = X * rw, Y * rw
        L2 = f(X, Y)

        heads_raw = jnp.zeros((), X.dtype)
        res_raw = jnp.zeros((), X.dtype)
        if self.s_hsic_heads:
            assert self.hsic_h > 0, "set hsic_h to the model's h"
            l, b, hk = P_all.shape
            k = hk // self.hsic_h
            for i in range(l):                    # l is static under jit
                Pl = P_all[i].reshape(b, self.hsic_h, k)
                heads_raw = heads_raw + hsic.pairwise_head_cka(
                    Pl, self.hsic_sigma2, self.hsic_estimator)
        if self.s_hsic_res:
            Xf = X[-1] if X.ndim == Y.ndim + 1 else X   # final prefix
            res_raw = hsic.hsic(Y - Xf, Y, estimator=self.hsic_estimator)

        # raw (unweighted) penalty sum for the stats column
        pen_raw = heads_raw + res_raw

        # inherited aux terms: s[1:] pairs with the 6 layer-summed stats
        # [L1_K, L1_F, entropy, cossim_b, cossim_h, KL_m] (ghost dropped)
        L = (L2 + self.s_hsic_heads * heads_raw
             + self.s_hsic_res * res_raw + jnp.dot(stats6, s[1:]))
        stats9 = jnp.concatenate(
            [jnp.stack([L, L2, pen_raw]), stats6])
        return L, stats9

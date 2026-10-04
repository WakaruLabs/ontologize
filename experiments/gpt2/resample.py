"""Dead-latent resampling for the head-sparse encoders (decoder-free).

A latent of a top-k encoder that never enters the top-k gets no gradient and
stays dead. Following Bricken et al. (2023, "resampling"), every `period`
steps we (1) find latents that fired on none of the tokens counted since the
last resample, (2) point their encoder rows at the layer inputs the model
reconstructs worst (sampled with probability proportional to squared
error), scaled so that the row's preactivation on its target input is
`scale` times that input's current top-k threshold (Bricken et al.'s
0.2x-live-norm rule is for ReLU encoders, where any positive preactivation
fires; under top-k it never re-enters the top k (19% -> 38% dead), and a
full-norm row fires on nearly every token and starves the trained latents
(19% -> 81% dead); observed 2026-09-30), with zero bias, and
(3) reset their Adam moments. Activity is counted on the current batch every
`count_every` steps (one extra forward pass), so the criterion is "silent
over ~period/count_every batches", not over the whole epoch.

Works on the layer inputs the encoders actually see (unit residual plus the
constant coordinate), with no auxiliary decoder or reconstruction loss: what
the latents are *trained* on is unchanged.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def _probe(module, X, T):
    """Per-layer (encoder input E, active mask (b, h_enc, m), per-token squared
    error of the layer's correction) with the training forward's hard/soft
    selection at temperature T and no noise."""
    E, _ = module.encode(X)
    R = module.resid(E)
    E = module.first_input(X, E, R)
    out = []
    for i, de in enumerate(module.dictencs):
        R0 = R
        g = module.gain(X, R0, i)
        n = X - module.decode(R0)
        z = de.hs_encoder(E)
        active = z > 0
        pre = de.hs_encoder.preacts(E)
        kk = min(de.hs_encoder.k_z, de.hs_encoder.m)
        thr = jax.lax.top_k(pre, kk)[0][..., -1]           # (b, h_enc): k-th largest preact
        K = de.logits(E)
        P = de.dict.cluster(K, T)
        S = de.scale(E) if de.scaled else None
        C = de.fiber_coords(E)
        R = module.apply_gain(R0, R0 + de.dict.combine(de.dict.hfwd(P, S, C=C)), g)
        c = module.decode(R) - module.decode(R0)
        err = jnp.sum((n - c) ** 2, -1)
        out.append((E, active, err, thr))
        if i < module.l - 1:
            E = module.nextinput(X, R, P.reshape(*P.shape[:-2], -1))
    return out


class Resampler:
    def __init__(self, model, T_of_step, period=5000, count_every=50, scale=1.1, seed=0, log=print):
        self.model, self.T_of_step = model, T_of_step
        self.period, self.count_every, self.scale = period, count_every, scale
        self.rng = np.random.default_rng(seed)
        self.log = log
        self.counts = None
        self.last_batch = None

        @jax.jit
        def probe(params, X, T):
            layers = model.apply(params, X, T, method=_probe)
            return [(E, a.sum(0), err, thr) for E, a, err, thr in layers]
        self._probe = probe

    def __call__(self, state, X):
        step = int(state.step)
        if step % self.count_every == 0 or step % self.period == 0:
            T = self.T_of_step(step)
            layers = self._probe(state.params, X, T)
            counts = [np.asarray(c) for _, c, _, _ in layers]
            self.counts = counts if self.counts is None else [a + b for a, b in zip(self.counts, counts)]
            self.last = [(np.asarray(E), np.asarray(err), np.asarray(thr)) for E, _, err, thr in layers]
        if step % self.period == 0 and self.counts is not None:
            state = self.resample(state)
            self.counts = None
        return state

    def resample(self, state):
        """Re-point dead rows. Only the sparse-encoder leaves (params and their
        Adam moments) are copied to the host, one at a time; everything else
        stays on the device (a full-state device_get of a 500M-parameter model
        is a 6 GB host spike, and doubled by the moments' reset)."""
        total_dead = 0
        new_rows = {}                                   # layer -> (W, b) host arrays
        for i, (dead_counts, (E, err, thr)) in enumerate(zip(self.counts, self.last)):
            dead = dead_counts == 0                                 # (h_enc, m)
            n_dead = int(dead.sum())
            total_dead += n_dead
            if n_dead == 0:
                continue
            enc = state.params["params"][f"dictencs_{i}"]["hs_encoder"]
            W, b = np.array(enc["W_enc"]), np.array(enc["b_enc"])   # (h_enc, m, d_in), (h_enc, m)
            h_enc, m, d = W.shape
            p = err.astype(np.float64) ** 2
            p /= p.sum()
            for hh in range(h_enc):
                idx = np.flatnonzero(dead[hh])
                if len(idx) == 0:
                    continue
                rows = self.rng.choice(len(E), size=len(idx), replace=True, p=p)
                v = E[rows]                                          # target inputs
                # new row w = c * x* with w.x* = scale * (current top-k threshold of
                # this head on x*): it enters the top-k on its target with a small
                # margin and fires elsewhere only where inputs resemble x*. (A row
                # at the live norm fires on nearly everything through the constant
                # coordinate and crowds out the trained latents: 19% -> 81% dead.)
                t = np.maximum(thr[rows, hh], 1e-6)
                c = self.scale * t / (np.sum(v * v, -1) + 1e-8)
                W[hh, idx] = (c[:, None] * v).astype(W.dtype)
                b[hh, idx] = 0.0
            new_rows[i] = (dead, W, b)
            self.log(f"resample step {int(state.step)} layer {i}: {n_dead}/{dead.size} dead latents "
                     f"({100 * n_dead / dead.size:.1f}%)")
        if not new_rows:
            self.log(f"resample step {int(state.step)}: no dead latents")
            return state

        def layer_of(kp):
            keys = [getattr(k, "key", None) for k in kp]
            if "hs_encoder" not in keys:
                return None
            for k in keys:
                if isinstance(k, str) and k.startswith("dictencs_"):
                    return int(k.split("_")[1])
            return None

        def new_param(kp, leaf):
            i = layer_of(kp)
            if i is None or i not in new_rows:
                return leaf
            dead, W, b = new_rows[i]
            name = getattr(kp[-1], "key", None)
            return W if name == "W_enc" else b if name == "b_enc" else leaf

        def reset_moment(kp, leaf):
            # Adam's mu/nu mirror the param tree: zero the resampled rows of the
            # encoder leaves, touching nothing else (device arrays pass through)
            i = layer_of(kp)
            if i is None or i not in new_rows or not hasattr(leaf, "shape"):
                return leaf
            dead, W, b = new_rows[i]
            if leaf.shape == W.shape:
                return np.where(dead[..., None], 0.0, np.asarray(leaf)).astype(W.dtype)
            if leaf.shape == b.shape:
                return np.where(dead, 0.0, np.asarray(leaf)).astype(b.dtype)
            return leaf
        params = jax.tree_util.tree_map_with_path(new_param, state.params)
        opt = jax.tree_util.tree_map_with_path(reset_moment, state.opt_state)
        # host arrays go back into the state; the next (donating) update step
        # moves them to the device after the old buffers are released
        return state.replace(params=params, opt_state=opt)

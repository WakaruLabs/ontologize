import functools
import math
import os
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import csv
import numpy as np
import einops
import torch as t

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from jaxtyping import Array, Float, PRNGKeyArray
from flax import struct
from flax.training.train_state import TrainState
from tqdm import tqdm

from ontologize.fns.keys import get_dtype, get_noise
from ontologize.fns.loss import identity
from ontologize.data.pretrained import encode
from ontologize.ontologizer import Ontologizer


def fsync_tree(root: Path) -> None:
    """Force everything under `root`, and the directory entries naming it,
    to stable storage. Syncing the files alone is not enough: a crash
    could leave the data written but unreachable, so each directory is
    synced after its contents. Missing paths are ignored, and an
    unsyncable one is skipped rather than failing a save that has already
    written its data."""
    if not root.exists():
        return
    for path, _, files in os.walk(root, topdown=False):
        for name in files:
            try:
                fd = os.open(os.path.join(path, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.fsync(fd)
            except OSError:
                pass
            finally:
                os.close(fd)
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

@struct.dataclass
class OntoState(TrainState):
    """`TrainState` for an `Ontologizer`."""
    model: Ontologizer = struct.field(pytree_node=False)
    b: int = struct.field(pytree_node=False)
    save_each: int = struct.field(pytree_node=False, default=1000)
    n_stats: int = struct.field(pytree_node=False, default=18)
    noise_in: str = struct.field(pytree_node=False, default="none")
    noise_K: str = struct.field(pytree_node=False, default="none")
    noise_F: str = struct.field(pytree_node=False, default="none")
    stats: Optional[Float[Array, "save_each n_stats"]] = None

    def spec(self) -> dict:
        """Converts `self.model` to a `dict` for serialization."""
        return asdict(self.model)

    def save(self, manager: ocp.CheckpointManager,
             keep_period: int = 0) -> bool:
        """Attempts to write `self` and `self.spec()` to `self.step`.
        Returns `True` if successful.

        `manager.wait_until_finished` waits for orbax's writes to complete,
        not for them to reach stable storage. `keep_period` syncs the
        checkpoints the manager retains and only those: the rest are
        unlinked within `max_to_keep` saves, and one unlinked while its
        pages are still dirty never reaches the device at all."""
        step = int(self.step)
        manager.save(step, items={'state': self, 'spec': self.spec()})
        manager.wait_until_finished()
        if keep_period and step % keep_period == 0:
            fsync_tree(Path(manager.directory) / str(step))
        return True

    def newstats(self) -> Float[Array, "save_each n_stats"]:
        """Initialize tensor of zeros with shape `(save_each, n_stats)`."""
        return jnp.zeros((self.save_each, self.n_stats),
                         get_dtype(self.model.dtype_p_str))

    def writestats(self, manager: ocp.CheckpointManager, file: str="loss.csv"
                    ) -> bool:
        """ Attempts to append `self.stats` to `file`. Returns `True` if successful.

        Synced before returning: a lost span of `loss.csv` cannot be
        reconstructed from anything, where a checkpoint can be reloaded.
        Unsynced, a hard stop leaves the file extended to its full size
        with the unflushed tail reading as NUL, so the row count still
        looks right."""
        path = manager.directory / file
        dat = np.array(self.stats)
        with open(path, 'a', newline='') as f:
            csv.writer(f).writerows(dat)
            f.flush()
            os.fsync(f.fileno())
            return True
        return False

    def last_stats(self) -> Float[Array, "n_stats"]:
        """The stats row most recently written by `stats_insert`. `update`
        inserts before `apply_gradients` increments the step, so after it
        returns the row sits one behind `self.step`."""
        return self.stats[(self.step - 1) % self.save_each]

def state_init(model: Ontologizer, b: int, tx, rng: PRNGKeyArray,
               save_each: int = 1000, n_stats: int=18, step: int=0,
               ghost: bool=True) -> OntoState:
    """Initialize an `OntoState` from an `Ontologizer`, optimiser, and `PRNGKeyArray`.
    When `ghost=False`, `apply_fn` runs `withStats` instead of `withGhost`
    (no ghost-gradient forward; the ghost output is `None` and
    `Hyperparams.loss` reports `L2_g = 0`)."""
    x = jnp.zeros((b, model.d_in), get_dtype(model.dtype_str))
    params = model.init(rng, x)
    if ghost:
        f = functools.partial(model.apply, method=Ontologizer.withGhost)
    else:
        f_stats = functools.partial(model.apply, method=Ontologizer.withStats)
        def f(params, X, *args, **kwargs):
            Y, stats, rng_next = f_stats(params, X, *args, **kwargs)
            return Y, None, stats, rng_next
    state = OntoState.create(model=model, b=b, save_each=save_each, n_stats=n_stats, 
                             apply_fn=f, params=params, tx=tx)
    stats_0 = state.newstats()
    return state.replace(step=step, stats=stats_0)

def stats_insert(state: OntoState, s: Float[Array, "n_stats"]) -> OntoState:
    """Inserts an array of statistics into `state.stats` at row index
    `state.step % state.save_each` Returns a copy of `state` with updated `stats`."""
    i = state.step % state.save_each
    stats = state.stats.at[i, :].set(s)
    return state.replace(stats=stats)

def load_params(state: OntoState, manager: ocp.CheckpointManager, step):
    """Load the model parameters (and optimizer state if available) from a specified `step` into `state`.
    Restores the step count to prevent overwriting old checkpoints and resetting the loss log.
    Does not restore `stats` to prevent shape mismatches if `save_each` changed."""
    cpt_raw = manager.restore(step, items={'state': None})
    
    if 'opt_state' in cpt_raw['state']:
        # New format: full state was saved. We must use `state` as the target to 
        # properly reconstruct `opt_state`'s NamedTuples.
        try:
            cpt = manager.restore(step, items={'state': state})
            restored = cpt['state']
        except Exception:
            # If shape mismatch occurs (e.g. `save_each` changed causing `stats` shape mismatch)
            old_stats_shape = cpt_raw['state']['stats'].shape
            target_state = state.replace(stats=jnp.zeros(old_stats_shape, dtype=state.stats.dtype))
            cpt = manager.restore(step, items={'state': target_state})
            restored = cpt['state']
            
        return restored.replace(stats=state.stats) # Keep newly initialized stats
    else:
        # Old format: only params were saved
        params = cpt_raw['state']
        
        # Orbax sometimes flattens single-key dictionaries when restoring without a target.
        # If the 'params' collection key is missing, re-wrap it so model.apply works.
        if 'params' not in params:
            params = {'params': params}
            
        return state.replace(params=params, step=step)

# Prevent recompilation by specifying these hyperparameters are immutable.
# `temperature`, `sd_K`, and `p_drop` are deliberately not static so they can
# be annealed per step without retracing (traced as weakly-typed scalars;
# the model code is branch-free in all three). `pwak_s` is the exception:
# it is a loop trip count, and unrolling it as a Python loop is what keeps
# the diffusion reverse-differentiable (a traced bound would lower to a
# `while_loop`, which has no reverse-mode rule). Its ramp is a handful of
# integers, so that is a handful of compilations over a run.
@functools.partial(jax.jit, static_argnames=[
    'sd_in', 'sd_F', 'lossfn', 'pwak_s'])
def update(state: OntoState, lossfn: Callable, rng: PRNGKeyArray,
           X: Float[Array, "... d_in"], Y_0: Float[Array, "... d_out"],
           grad_clip: Optional[float] = None,
           s_L1F: Optional[Float[Array, ""]] = None,
           s_kcossim: Optional[Float[Array, ""]] = None, *args, **kwargs
           ) -> Tuple[OntoState, Float[Array, ""],
                      Float[Array, "n_stats"], PRNGKeyArray]:
    """Core function updating an `OntoState` via gradient descent.
    `s_L1F` and `s_kcossim` pass through to `lossfn` as traced scalars
    (`lossfn` is static, so a partial closing over them would retrace every
    step)."""
    def f(params):
        Y, Y_g, stats, rng_next = state.apply_fn(
                params, X, *args, **kwargs, rng=rng)
        L, stats = lossfn(Y, Y_0, Y_g, stats, s_L1F, s_kcossim)
        return L, (Y, stats, rng_next)

    gradfn = jax.value_and_grad(f, has_aux=True)
    (L, (Y, stats, rng_next)), grad = gradfn(state.params)
    
    if grad_clip is not None:
        # Global gradient norm clipping
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grad)))
        scaling = jnp.minimum(1.0, grad_clip / (grad_norm + 1e-6))
        grad = jax.tree_util.tree_map(lambda x: x * scaling, grad)
        
    state = stats_insert(state, stats)
    return state.apply_gradients(grads=grad), L, rng_next

def dual_target(step: int, step_0: int, ramp: int,
                start: float, target: float) -> float:
    """Geometric ramp of a setpoint from `start` to `target`,
    reaching it at step `ramp` and held there after.

    The loop is a pure integrator against a plant that takes thousands of
    steps to respond, so demanding the final setpoint immediately is what
    winds it up: the measurement begins well above target, the error stays
    large and one-signed while the plant lags, and the multiplier
    integrates to its bound and overshoots badly before it changes sign.
    Ramping keeps the error near zero throughout, so the multiplier
    tracks the value that sustains the *current* target instead of
    integrating against an unreachable one -- interpolation rather than
    extrapolation.

    `start` is measured when the loop begins rather than configured, so the
    ramp starts at zero error by construction. `step_0` is where that
    measurement was taken, which makes this resume-safe: a resumed run
    measures wherever it left off and covers the remaining steps."""
    if ramp <= step_0:
        return target
    f = min(max((step - step_0) / (ramp - step_0), 0.0), 1.0)
    if start <= 0.0 or target <= 0.0:
        return start + (target - start) * f
    return start * (target / start) ** f

def dual_control(s: float, measured: float, target: float, eta: float,
                 lo: float, hi: float) -> float:
    """One step of multiplicative dual ascent on a penalty coefficient,
    holding `measured` at `target`.

    `log s <- log s + eta * log(measured / target)`, clipped to `[lo, hi]`.
    Multiplicative because the multiplier spans orders of magnitude and the
    sensible step is relative; the error is a log-ratio so overshoot and
    undershoot are treated symmetrically and a far-from-target measurement
    produces a large but not explosive correction.

    This is the standard constrained-optimization dual step, with `s` as
    the Lagrange multiplier on `measured <= target`: it rises
    while the constraint is violated and decays once it is satisfied. It
    supplies the equilibrium a fixed coefficient does not have -- see
    `Hyperparams.L1F_target`."""
    if target <= 0.0:
        return s
    err = math.log(max(measured, 1e-12) / target)
    return min(max(s * math.exp(eta * err), lo), hi)

def dual_apply(s: float, measured: float, target: float) -> float:
    """Project the multiplier onto the constraint: `s` is the dual variable
    of `measured <= target`, so complementary slackness sends it to exactly
    zero while the constraint is slack.

    It matters most against a one-way plant, where the penalty can lower
    `measured` but nothing raises it back: every overshoot is then
    permanent, and a multiplier that merely decays while the constraint is
    slack keeps pushing past the target forever. Against a two-way plant
    it instead acts as a thermostat -- off while healthy, costing nothing,
    and re-engaging on drift.

    `dual_control` keeps evolving the unprojected variable underneath, so
    the penalty resumes from a sensible level rather than from zero (which
    a multiplicative update could never leave) when `measured` comes back
    above target -- as it does on every step of a descending ramp."""
    return s if target <= 0.0 or measured > target else 0.0

class DualLoop:
    """Per-step state for one setpoint controller: the unprojected dual
    variable, the EMA of what it watches, and the ramped setpoint.

    Two of these run in `train` over different measurements, which is why
    the pieces are parameterized rather than written twice. `on` is False
    when `target` is 0, and every method is then a no-op returning the
    configured coefficient unchanged."""

    def __init__(self, s: float, target: float, eta: float, ema: float,
                 lo: float, hi: float, ramp: int, step_0: int):
        self.s, self.target, self.eta, self.ema = s, target, eta, ema
        self.lo, self.hi, self.ramp, self.step_0 = lo, hi, ramp, step_0
        self.on = target > 0.0
        self.avg = None
        self.tgt_0 = None
        self.tgt = target
        self.applied = s

    def step(self, step: int, measured: float) -> float:
        """Fold in one measurement and return the coefficient to apply
        next. `measured` sets the ramp's start the first time it is seen,
        so the loop begins at zero error however far from target it is."""
        if not self.on:
            return self.s
        self.avg = measured if self.avg is None \
            else self.ema * self.avg + (1.0 - self.ema) * measured
        if self.tgt_0 is None:
            self.tgt_0 = self.avg
        self.tgt = dual_target(step, self.step_0, self.ramp,
                               self.tgt_0, self.target)
        self.s = dual_control(self.s, self.avg, self.tgt, self.eta,
                              self.lo, self.hi)
        self.applied = dual_apply(self.s, self.avg, self.tgt)
        return self.applied

def schedules(step: int, anneal_steps: int,
              temperature: float, temperature_end: Optional[float],
              p_drop: float, p_drop_start: Optional[float],
              sd_K: float, sd_K_end: Optional[float],
              pwak_s: int = 0, pwak_s_start: int = 0
              ) -> Tuple[float, float, float]:
    """Per-step annealed `(temperature, p_drop, sd_K, pwak_s)`, all sharing the
    `anneal_steps` horizon and held at their end values afterwards.
    Temperature and `sd_K` anneal geometrically (linearly if an endpoint is
    0); `p_drop` ramps linearly from `p_drop_start` (dropout is redundant
    while labels are soft -- its value is in the hardening window). Each
    schedule is active only when its companion endpoint is not `None`."""
    if not anneal_steps:
        return temperature, p_drop, sd_K, pwak_s
    f = min(step / anneal_steps, 1.0)

    def geo(a: float, b: float) -> float:
        if a == 0.0 or b == 0.0:
            return a + (b - a) * f
        return a * (b / a) ** f

    T = temperature if temperature_end is None \
        else geo(temperature, temperature_end)
    pd = p_drop if p_drop_start is None \
        else p_drop_start + (p_drop - p_drop_start) * f
    sk = sd_K if sd_K_end is None else geo(sd_K, sd_K_end)
    # diffusion depth ramps linearly and lands on `pwak_s`. Integer by
    # construction: G^s is s matmuls. Starting above 0 would diffuse while
    # the partition is still noise, which locks the noise in.
    ps = int(pwak_s_start + round((pwak_s - pwak_s_start) * f))
    return T, pd, sk, ps

def train(state: OntoState, dat, lossfn: Callable,
          rng: PRNGKeyArray, epochs: int,
          manager: ocp.CheckpointManager, save_each: int=0,
          checkpoint_each: int=0,
          encoder=None, decoder: Callable=None, dev=t.device("cuda"),
          temperature: float=1.0, temperature_end: Optional[float]=None,
          anneal_steps: int=0,
          p_drop: float=0.0, p_drop_start: Optional[float]=None,
          p_revive: float=0.0, revive_frac: float=0.5,
          sd_K: float=0.0, sd_K_end: Optional[float]=None,
          pwak_s: int=0, pwak_s_start: int=0, pwak_tau: float=0.2,
          s_L1F: float=0.0, L1F_target: float=0.0, L1F_eta: float=1e-3,
          L1F_ema: float=0.99, L1F_min: float=1e-12, L1F_max: float=1e-3,
          L1F_ramp: int=0,
          s_kcossim: float=0.0, KCOS_target: float=0.0,
          KCOS_eta: float=1e-3, KCOS_ema: float=0.99,
          KCOS_min: float=1e-12, KCOS_max: float=1e-2, KCOS_ramp: int=0,
          *args, **kwargs) -> OntoState:
    """Training loop for an `OntoState`. Saves checkpoint and appends `state.stats`
    to `loss.csv` every `save_each`, then reinitializes `state.stats`.
    Temperature, winner dropout, and classifier noise follow `schedules`
    (indexed by `state.step`, so annealing resumes correctly from
    checkpoints). Note classifier noise enters the softmax as `sd_K / T`:
    constant `sd_K` under falling temperature means growing effective
    exploration; set `sd_K_end` to control the endpoint instead.

    A `<X>_target` above 0 makes the matching coefficient a controlled
    variable (`DualLoop`) instead of a constant: it is adjusted per step to
    hold the measurement at a setpoint that ramps from wherever the run
    starts, and `s_L1F` / `s_kcossim` become starting values. Both loops
    run in Python because each is per-step scalar arithmetic on a value the
    loop already reads back.

    `KCOS_target` watches the per-layer MAX of `cossim_k`, not the summed
    stat: the layers that stay healthy improve faster than a collapsing one
    degrades, so the sum falls throughout a collapse while the max rises."""
    step_0 = int(state.step)
    l1f = DualLoop(s_L1F, L1F_target, L1F_eta, L1F_ema, L1F_min, L1F_max,
                   L1F_ramp or anneal_steps, step_0)
    kcos = DualLoop(s_kcossim, KCOS_target, KCOS_eta, KCOS_ema, KCOS_min,
                    KCOS_max, KCOS_ramp or anneal_steps, step_0)

    # `dat` yields only the batches after `step_0` (see `SampleLoader`), so
    # the loop ends at the run's final step whether or not it resumed
    left = getattr(dat, "steps", None)
    with tqdm(dat, desc=f"Training {epochs} epochs", initial=step_0,
              total=None if left is None else step_0 + left) as pbar:
        for batch in pbar:
            # Assuming autoencoder where target Y_0 is exactly input X
            X = batch
            if encoder is not None:
                X = encode(encoder, X, dev)
            Y = X
            if decoder is not None:
                Y = decoder(Y)
            T, pd, sk, ps = schedules(int(state.step), anneal_steps,
                                      temperature, temperature_end,
                                      p_drop, p_drop_start, sd_K, sd_K_end,
                                      pwak_s, pwak_s_start)
            state, L, rng = update(
                    state, lossfn, rng, X, Y,
                    *args, **kwargs, temperature=T, p_drop=pd, sd_K=sk,
                    s_L1F=l1f.applied if l1f.on else None,
                    s_kcossim=kcos.applied if kcos.on else None,
                    p_revive=p_revive, revive_frac=revive_frac,
                    pwak_s=ps, pwak_tau=pwak_tau)
            post = {"loss": float(L), "step": int(state.step),
                    "T": round(T, 4)}
            # stats row columns; see Hyperparams.loss for the layout
            row = state.last_stats()
            if l1f.on:
                l1f.step(int(state.step), float(row[4]))     # L1_F
                post["L1_F"] = round(l1f.avg, 1)
                post["L1_tgt"] = round(l1f.tgt, 1)
                post["s_L1F"] = f"{l1f.applied:.2e}"
            if kcos.on:
                kcos.step(int(state.step), float(row[15]))   # max cossim_k
                post["ck"] = round(kcos.avg, 3)
                post["ck_tgt"] = round(kcos.tgt, 3)
                post["s_ck"] = f"{kcos.applied:.2e}"
            if p_drop_start is not None:
                post["p_drop"] = round(pd, 4)
            if sd_K_end is not None:
                post["sd_K"] = round(sk, 5)
            if pwak_s:
                post["pwak_s"] = ps
            pbar.set_postfix(post)

            if save_each and not (state.step % save_each):
                if state.save(manager, checkpoint_each):
                    if state.writestats(manager):
                        state = state.replace(stats=state.newstats())

    return state


"""Training state management and optimization execution for Ontologizer models.

This module defines `OntoState` (extending Flax's `TrainState`), initialization
routines, checkpoint restoration, JIT-compiled optimization steps (`update`),
annealing schedules for exploration hyperparameters (`schedules`), and the
primary dataset training loop (`train`).
"""

import functools
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import csv
import numpy as np
import einops
import torch as t

from dataclasses import asdict
from typing import Any, Callable, List, Optional, Tuple
from jaxtyping import Array, Float, PRNGKeyArray
from flax import struct
from flax.training.train_state import TrainState
from tqdm import tqdm

from ontologize.fns.keys import get_dtype, get_noise
from ontologize.fns.loss import identity
from ontologize.data.pretrained import encode
from ontologize.ontologizer import Ontologizer


@struct.dataclass
class OntoState(TrainState):
    """`TrainState` for an `Ontologizer`."""
    model: Ontologizer = struct.field(pytree_node=False)
    b: int = struct.field(pytree_node=False)
    save_each: int = struct.field(pytree_node=False, default=1000)
    n_stats: int = struct.field(pytree_node=False, default=10)
    noise_in: str = struct.field(pytree_node=False, default="none")
    noise_K: str = struct.field(pytree_node=False, default="none")
    noise_F: str = struct.field(pytree_node=False, default="none")
    stats: Optional[Float[Array, "save_each n_stats"]] = None

    def spec(self) -> dict:
        """Converts `self.model` to a `dict` for serialization."""
        return asdict(self.model)

    def save(self, manager: ocp.CheckpointManager) -> bool:
        """Attempts to write `self` and `self.spec()` to `self.step`.
        Returns `True` if successful."""
        manager.save(int(self.step),
                            items={
                                'state': self,
                                'spec': self.spec()
                                })
        manager.wait_until_finished()
        return True

    def newstats(self) -> Float[Array, "save_each n_stats"]:
        """Initialize tensor of zeros with shape `(save_each, n_stats)`."""
        return jnp.zeros((self.save_each, self.n_stats),
                         get_dtype(self.model.dtype_p_str))

    def writestats(self, manager: ocp.CheckpointManager, file: str="loss.csv"
                    ) -> bool:
        """ Attempts to append `self.stats` to `file`. Returns `True` if successful."""
        path = manager.directory / file
        dat = np.array(self.stats)
        with open(path, 'a', newline='') as f:
            csv.writer(f).writerows(dat)
            return True
        return False

def state_init(model: Ontologizer, b: int, tx, rng: PRNGKeyArray,
               save_each: int = 1000, n_stats: int=10, step: int=0,
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
# the model code is branch-free in all three).
# The incoming state is donated: its params / optimizer moments are updated in
# place instead of coexisting with the new ones (halves the peak memory of a
# step, which matters for 500M-parameter encoders). Callers must not reuse
# the state they passed in.
@functools.partial(jax.jit, static_argnames=[
    'sd_in', 'sd_F', 'lossfn'], donate_argnums=(0,))
def update(state: OntoState, lossfn: Callable, rng: PRNGKeyArray,
           X: Float[Array, "... d_in"], Y_0: Float[Array, "... d_out"], 
           grad_clip: Optional[float] = None, *args, **kwargs
           ) -> Tuple[OntoState, Float[Array, ""], 
                      Float[Array, "n_stats"], PRNGKeyArray]:
    """Core function updating an `OntoState` via gradient descent."""
    def f(params):
        Y, Y_g, stats, rng_next = state.apply_fn(
                params, X, *args, **kwargs, rng=rng)
        L, stats = lossfn(Y, Y_0, Y_g, stats)
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

def anneal_fraction(step: int, anneal_steps: int) -> float:
    """Fraction of the anneal horizon elapsed (1.0 when there is none)."""
    return min(step / anneal_steps, 1.0) if anneal_steps else 1.0


def schedules(step: int, anneal_steps: int,
              temperature: float, temperature_end: Optional[float],
              p_drop: float, p_drop_start: Optional[float],
              sd_K: float, sd_K_end: Optional[float]
              ) -> Tuple[float, float, float]:
    """Per-step annealed `(temperature, p_drop, sd_K)`, all sharing the
    `anneal_steps` horizon and held at their end values afterwards.
    Temperature and `sd_K` anneal geometrically (linearly if an endpoint is
    0); `p_drop` ramps linearly from `p_drop_start` (dropout is redundant
    while labels are soft -- its value is in the hardening window). Each
    schedule is active only when its companion endpoint is not `None`."""
    if not anneal_steps:
        return temperature, p_drop, sd_K
    f = anneal_fraction(step, anneal_steps)

    def geo(a: float, b: float) -> float:
        if a == 0.0 or b == 0.0:
            return a + (b - a) * f
        return a * (b / a) ** f

    T = temperature if temperature_end is None \
        else geo(temperature, temperature_end)
    pd = p_drop if p_drop_start is None \
        else p_drop_start + (p_drop - p_drop_start) * f
    sk = sd_K if sd_K_end is None else geo(sd_K, sd_K_end)
    return T, pd, sk

def train(state: OntoState, dat, lossfn: Callable,
          rng: PRNGKeyArray, epochs: int,
          manager: ocp.CheckpointManager, save_each: int=0,
          encoder=None, decoder: Callable=None, dev=t.device("cuda"),
          temperature: float=1.0, temperature_end: Optional[float]=None,
          anneal_steps: int=0,
          p_drop: float=0.0, p_drop_start: Optional[float]=None,
          sd_K: float=0.0, sd_K_end: Optional[float]=None,
          hook: Optional[Callable[[OntoState, Array], OntoState]]=None,
          *args, **kwargs) -> OntoState:
    """Training loop for an `OntoState`. Saves checkpoint and appends `state.stats`
    to `loss.csv` every `save_each`, then reinitializes `state.stats`.
    `hook(state, X)`, if given, runs after every update and may return a
    modified state (used for host-side interventions such as dead-latent
    resampling; it decides itself on which steps to act).
    Temperature, winner dropout, and classifier noise follow `schedules`
    (indexed by `state.step`, so annealing resumes correctly from
    checkpoints). Note classifier noise enters the softmax as `sd_K / T`:
    constant `sd_K` under falling temperature means growing effective
    exploration; set `sd_K_end` to control the endpoint instead."""

    # dat iterator already repeats for `epochs` because of num_epochs in SampleLoader.
    with tqdm(dat, desc=f"Training {epochs} epochs") as pbar:
        for batch in pbar:
            # Assuming autoencoder where target Y_0 is exactly input X
            X = batch
            if encoder is not None:
                X = encode(encoder, X, dev)
            Y = X
            if decoder is not None:
                Y = decoder(Y)
            T, pd, sk = schedules(int(state.step), anneal_steps,
                                  temperature, temperature_end,
                                  p_drop, p_drop_start, sd_K, sd_K_end)
            # `hardness` (the anneal fraction) only matters for select="anneal"
            state, L, rng = update(
                    state, lossfn, rng, X, Y,
                    *args, **kwargs, temperature=T, p_drop=pd, sd_K=sk,
                    hardness=anneal_fraction(int(state.step), anneal_steps))
            if hook is not None:
                state = hook(state, X)
            post = {"loss": float(L), "step": int(state.step),
                    "T": round(T, 4)}
            if p_drop_start is not None:
                post["p_drop"] = round(pd, 4)
            if sd_K_end is not None:
                post["sd_K"] = round(sk, 5)
            pbar.set_postfix(post)

            if save_each and not (state.step % save_each):
                if state.save(manager):
                    if state.writestats(manager):
                        state = state.replace(stats=state.newstats())

    return state


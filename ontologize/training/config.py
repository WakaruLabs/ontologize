"""Configuration dataclasses, hyperparameter specifications, and execution environment.

This module defines configuration containers used throughout the Ontologizer training
pipeline:
- `Hyperparams`: Optimizer settings, loss weights, noise injection parameters,
  and parameter annealing schedules.
- `Metadata`: Serialization targets, checkpoint retention policies, and data loader
  dispatch.
- `TrainingEnv`: Execution wrapper coordinating initialization, dataset loading,
  and model optimization.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import orbax.checkpoint as ocp
import grain.python as gp
import csv
import numpy as np
import os
import json

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from jaxtyping import Array, Bool, Float, UInt, PRNGKeyArray
from flax.training.train_state import TrainState

from ontologize.ontologizer import Ontologizer
from .ontostate import OntoState, state_init, load_params, update, train
from ontologize.fns.keys import get_loss, get_srctype
from ontologize.data.loaders import SampleLoader
from ontologize.data.pretrained import pretrained_transformer

_MSE_W_CACHE = {}

def _mse_weights(path):
    """Load and cache a per-dim MSE weight vector (constant under jit)."""
    if path not in _MSE_W_CACHE:
        _MSE_W_CACHE[path] = jnp.asarray(np.load(path))
    return _MSE_W_CACHE[path]

@dataclass
class Hyperparams:
    """Training hyperparameters."""
    d_in: int
    d_out: int
    b: int

    epochs: int = 10

    lr: float = 1e-3
    wd: float = 0.0

    temperature: float = 1.0
    noise_in: str = "none"
    noise_K: str = "none"
    noise_F: str = "none"
    sd_in: float = 0.0
    sd_K: float = 0.0
    sd_F: float = 0.0
    # winner dropout: probability per head per sample of masking the argmax
    # dict entry so the runner-up wins and receives gradient (anti-collapse)
    p_drop: float = 0.0

    n_stats: int = 10  # [L, L2, L2_g, L1_K, L1_F, entropy, cossim_b, cossim_h, KL_m, aux]
    lossfn: str = "mse"
    # optional path to a (d_out,) npy of per-dim MSE weights (e.g.
    # inverse-variance for target whitening, normalized to mean 1). Applied
    # as sqrt-weights to output/target/ghost before the loss, so MSE
    # equal-weights dims instead of letting high-variance dims dominate.
    mse_weights: Optional[str] = None

    # scaling values for loss components
    s_g: float = 0.0
    s_L1K: float = 0.0 #disrecommended; tends to paradoxically cause exploding L1_K
    s_L1F: float = 0.0
    s_H: float = 0.0
    s_bcossim: float = 0.0
    s_hcossim: float = 0.0
    # batch mean-entropy bonus: penalizes KL(E_batch[p] ‖ uniform) per head
    # (bits, mean over heads, summed over layers). Maximizes the entropy of
    # the *mean* classification, not the mean of the entropies: per-sample
    # hardness is untouched, only usage imbalance is taxed. Anchors the
    # corpus-generic code point at the uniform mixture and opposes
    # frozen-head collapse (a frozen head pays the full log2(k) bits, with
    # pressure applied before its softmax saturates and gradients vanish).
    s_Hm: float = 0.0
    # weight of the sparse encoders' auxiliary SAE loss (`aux` stat; see
    # `DictEnc.hs_decoder`)
    s_aux: float = 0.0

    seed: int = 42
    grad_clip: Optional[float] = 1.0

    # ghost-gradient dead-feature resurrection. When False, training runs
    # `Ontologizer.withStats` instead of `withGhost`, skipping the parallel
    # ghost forward (~a full extra model pass per step) and reporting the
    # MSE_ghost stat as 0. Note ghostgrad only fires for features that are
    # exactly 0 across the whole batch: reachable with relu-gated
    # classifiers, structurally impossible with bilinear (gate="none")
    # classifiers + softmax selection + abs()'d dictionaries -- in that
    # regime the ghost path is inert (MSE_ghost == MSE) and pure overhead.
    ghost: bool = True

    # geometric temperature annealing: anneal from `temperature` to
    # `temperature_end` over the first `anneal_steps` steps, then hold.
    # Disabled when `temperature_end` is None or `anneal_steps` is 0.
    temperature_end: Optional[float] = None
    anneal_steps: int = 0

    # companion schedules sharing the `anneal_steps` horizon (each active
    # only when its endpoint is not None):
    # winner dropout ramps LINEARLY p_drop_start -> p_drop. Dropout is
    # redundant while labels are soft (every entry already gets gradient
    # through the soft mixture); its value is in the hardening window.
    p_drop_start: Optional[float] = None
    # classifier logit noise anneals GEOMETRICALLY sd_K -> sd_K_end. The
    # noise enters the softmax as sd_K / T, so constant sd_K under falling
    # temperature means growing effective exploration; choose sd_K_end
    # relative to temperature_end to set the late-training level instead
    # (e.g. sd_K_end = 0.2 * temperature_end reproduces the historically
    # calibrated hard-phase regime of sd_K=0.02 at T=0.1).
    sd_K_end: Optional[float] = None

    def s_loss(self, *args, **kwargs) -> Tuple[Float[Array, "8"],
                                               Tuple[bool, bool, bool, bool, bool, bool, bool, bool]]:
        """Returns a vector of scaling values for loss components and whether each is nonzero.
        This prevents backpropagation through functions used for evaluation but not training."""
        s = jnp.array([self.s_g, self.s_L1K, self.s_L1F, self.s_H,
                       self.s_bcossim, self.s_hcossim, self.s_Hm, self.s_aux],
                      *args, **kwargs)
        isloss = (self.s_g != 0.0, self.s_L1K != 0.0, self.s_L1F != 0.0,
                  self.s_H != 0.0, self.s_bcossim != 0.0, self.s_hcossim != 0.0,
                  self.s_Hm != 0.0, self.s_aux != 0.0)
        return s, isloss

    def rng(self) -> PRNGKeyArray:
        """Creates PRNGKey from `self.seed`."""
        return jax.random.PRNGKey(self.seed)

    def loss(self, X: Float[Array, "... b d_out"], 
             Y: Float[Array, "... b d_out"], X_g: Float[Array, "... b d_out"],
             stats: Float[Array, "4"]
             ) -> Tuple[Float[Array, ""], Float[Array, "n_stats"]]:
        """Accepts model output `X`, target `Y`, ghost gradient output `X_g`,
        and validation statistics `stats`. `L2` is defined as `self.lossfn(X, Y)`.
        `L2_g` is defined as `self.lossfn(X_g, Y - X)`. Total loss `L` is defined as
        `L2 + s_g * L2_g + sum([s_L1K, s_L1F, s_H, s_bcossim, s_hcossim, s_Hm, s_aux] * stats)`."""
        f = get_loss(self.lossfn)
        s, _ = self.s_loss() # scaling vector; ignore isloss
        stats = jnp.einsum("...s -> s", stats) # sum stats over layers
        if self.mse_weights:
            rw = jnp.sqrt(_mse_weights(self.mse_weights)).astype(X.dtype)
            X, Y = X * rw, Y * rw
            X_g = X_g * rw if X_g is not None else None
        L2 = f(X, Y)
        # X_g is None when the ghost path is disabled (see `ghost`)
        L2_g = f(X_g, Y - X) if X_g is not None else jnp.zeros_like(L2)
        stats = jnp.append(L2_g, stats)
        L = L2 + jnp.dot(stats, s)
        stats = jnp.append(jnp.stack([L, L2]), stats)
        return L, stats

    def opt(self, *args, **kwargs):
        """Adam optimizer with learning rate `self.lr`. If `self.wd` is nonzero, returns AdamW.
        """
        if self.wd:
            return optax.adamw(self.lr, weight_decay=self.wd, *args, **kwargs)
        return optax.adam(self.lr, *args, **kwargs)

    def ontologizer(self, *args, **kwargs):
        """Instantiates an `Ontologizer` model configured with loss gating and noise settings.

        Passes active loss flags determined by `self.s_loss()` (`sparse_K`, `sparse_F`,
        `entropy_loss`, `cossim_loss`, `bcossim_loss`, `hmean_loss`) and noise type
        strings (`noise_in`, `noise_K`, `noise_F`).

        Args:
            *args: Positional arguments forwarded to `Ontologizer` constructor.
            **kwargs: Keyword arguments forwarded to `Ontologizer` constructor.

        Returns:
            Ontologizer: Configured Flax Linen `Ontologizer` model instance.
        """
        _, (ghost_loss, sparse_K, sparse_F, entropy_loss, bcossim_loss, cossim_loss, hmean_loss, _aux) = self.s_loss()
        return Ontologizer(
                *args, **kwargs, sparse_K=sparse_K, sparse_F=sparse_F,
                entropy_loss=entropy_loss, cossim_loss=cossim_loss, bcossim_loss=bcossim_loss,
                hmean_loss=hmean_loss,
                noise_in=self.noise_in, noise_K=self.noise_K, noise_F=self.noise_F,)
                #sd_in=self.sd_in, sd_K=self.sd_K, sd_F=self.sd_F)

    def init(self, model: Ontologizer, save_each: int=1000) -> OntoState:
        """Initializes `OntoState` for `model` using `self.b`, `self.opt(), and `self.rng()."""
        return state_init(model, self.b, self.opt(), self.rng(), save_each,
                          self.n_stats, ghost=self.ghost)

    def update(self, state: OntoState, rng: PRNGKeyArray,
               X: Float[Array, "... b d_in"],
               Y: Float[Array, "... b d_out"],
               *args, **kwargs) -> OntoState:
        """Unused. `self.train` calls `ontologize.ontostate.update` instead."""
        return update(
                state, self.loss, rng, X, Y,
                *args, **kwargs, temperature=self.temperature,
                sd_in=self.sd_in, sd_K=self.sd_K, sd_F=self.sd_F,
                p_drop=self.p_drop,
                grad_clip=self.grad_clip)

    def train(self, state: OntoState, dat: SampleLoader, *args, **kwargs) -> OntoState:
        """Training loop for an `OntoState` and `SampleLoader."""
        return train(
                state, dat, self.loss, self.rng(), self.epochs,
                *args, **kwargs, temperature=self.temperature,
                temperature_end=self.temperature_end,
                anneal_steps=self.anneal_steps,
                sd_in=self.sd_in, sd_K=self.sd_K, sd_F=self.sd_F,
                p_drop=self.p_drop,
                p_drop_start=self.p_drop_start, sd_K_end=self.sd_K_end,
                grad_clip=self.grad_clip)

    def load(self, manager: ocp.CheckpointManager, step):
        """Initializes an `OntoState`, then loads a checkpoint specified by `step`."""
        spec = manager.restore(step, items={'spec': None})['spec']
        model = Ontologizer(**spec)
        state = self.init(model)
        return load_params(state, manager, step)



@dataclass 
class Metadata:
    """Settings controlling serialization & data loading"""
    srctype: str="embedding"
    model_id: str=None
    sae_id: str=None
    layer: int=None

    out_path: Any = "data/out"  # Can be str or Path
    data_path: Any = "data"  # Can be str or Path
    threads: int = 0
    
    checkpoint_each: int = 100000
    save_each: int = 1000
    max_to_keep: int = 10
    resume_from: int = None

    overwrite: bool = False

    test_ratio: Optional[float] = None
    test_file: str = None
    shuffle: bool = False

    def manager(self, *args, **kwargs) -> ocp.CheckpointManager:
        """Creates optax CheckpointManager to save a checkpoint to `self.out_path` 
        every `self.save_each` steps. Every `self.checkpoint_each` steps, 
        the checkpoint is preserved. Otherwise the most recent `self.max_to_keep`
        checkpoints are retained."""
        opts = ocp.CheckpointManagerOptions(
                save_interval_steps=self.save_each,
                max_to_keep=self.max_to_keep,
                keep_period=self.checkpoint_each
                )
        
        # Ensure path is explicitly absolute
        abs_path = os.path.abspath(str(self.out_path))
        
        return ocp.CheckpointManager(
                abs_path,
                checkpointers={
                    'state': ocp.PyTreeCheckpointer(),
                    'spec': ocp.PyTreeCheckpointer()
                    }, 
                options=opts
                )

    def src(self, *args, **kwargs) -> gp.ArrayRecordDataSource:
        """Loads data from `self.data_path`."""
        return gp.ArrayRecordDataSource(self.data_path, *args, **kwargs)

    def loader(self, *args, **kwargs) -> SampleLoader:
        """Creates the appropriate data loader type from `ontologize.training.data`
        as specified by `self.srctype`."""
        method = get_srctype(self.srctype)
        return method(*args, **kwargs, threads=self.threads)

    def model(self, *args, **kwargs):
        """Loads `self.model_id`."""
        return pretrained_transformer(self.model_id, *args, **kwargs)

def log_env(log_path, hyper: Hyperparams, meta: Metadata):
   """Logs the hyperparams and metadata to a JSONL file."""
   
   # 1. Convert dataclasses to nested dictionaries
   config_dict = {
       "type": "env_config",
       "hyper": asdict(hyper),
       "meta": asdict(meta)
   }

   # 2. Custom JSON serializer to handle Paths and other JAX types
   def serializer(obj):
       if isinstance(obj, Path):
           return str(obj)
       if hasattr(obj, 'dtype'): # Handle JAX/Numpy types if they slipped in
           return str(obj)
       raise TypeError(f"Type {type(obj)} not serializable")

   # 3. Append to the log file
   with open(log_path, "a") as f:
       f.write(json.dumps(config_dict, default=serializer) + "\n")
   
   print(f"✓ Environment configuration logged to {log_path}")


@dataclass
class TrainingEnv:
    """Container for immutable training variables."""
    spec: Ontologizer
    hyper: Hyperparams
    meta: Metadata

    args_loss: List[Any] = field(default_factory=list)
    kwargs_loss: dict = field(default_factory=dict)

    args_loader: List[Any] = field(default_factory=list)
    kwargs_loader: dict = field(default_factory=dict)

    args_manager: List[Any] = field(default_factory=list)
    kwargs_manager: dict = field(default_factory=dict)

    def init(self, src, keep_spec: bool=False
             ) -> Tuple[OntoState, SampleLoader, ocp.CheckpointManager]:
        """Initialize `OntoState` with `self.hyper.init(self.spec)`,
        `SampleLoader` with self.meta.loader`, and `CheckpointManager` with 
        `self.meta.manager`"""
        loader = self.meta.loader(
                self.hyper.b, self.hyper.epochs, src,
                *self.args_loader, **self.kwargs_loader)
        manager = self.meta.manager(*self.args_manager, **self.kwargs_manager)

        if self.meta.resume_from is None and not self.meta.overwrite:
            self.meta.resume_from = manager.latest_step()

        state = self.hyper.init(self.spec, save_each=self.meta.save_each)
        if self.meta.resume_from is not None:
            if keep_spec:
                # load architecture from checkpoint
                state = self.hyper.load(manager, self.meta.resume_from)
            else:
                # load params into specified architechture
                state = load_params(state, manager, self.meta.resume_from)
            
            # Truncate loss.csv to the resumed step
            loss_path = manager.directory / "loss.csv"
            if loss_path.exists():
                try:
                    with open(loss_path, "r") as f:
                        lines = f.readlines()
                    # keep up to resume_from lines
                    lines = lines[:self.meta.resume_from]
                    with open(loss_path, "w") as f:
                        f.writelines(lines)
                except Exception as e:
                    print(f"Warning: could not truncate loss.csv: {e}")

        return state, loader, manager

    def train(self, src, encoder=None, decoder=None) -> OntoState:
        """Initialize and train an `OntoState`. Optionally takes an `encoder` which processes
        data before passing it to the `Ontologizer` and a `decoder` which processes 
        `Ontologizer` outputs before loss calculation."""
        state, loader, manager = self.init(src)

        log_path = self.meta.out_path / "log.jsonl"
        if self.meta.resume_from is not None:
            if log_path.exists():
                with open(log_path, "r+") as f:
                    lines = f.readlines()
                    f.seek(0)
                    for line in lines:
                        try:
                            log_entry = json.loads(line)
                            if 'step' in log_entry and log_entry['step'] >= self.meta.resume_from:
                                break
                            f.write(line)
                        except json.JSONDecodeError:
                            # Keep malformed lines
                            f.write(line)
                    f.truncate()

        log_env(log_path, self.hyper, self.meta)
        state = self.hyper.train(
                state, loader, manager, save_each=self.meta.save_each,
                encoder=encoder, decoder=decoder,
                *self.args_loss, **self.kwargs_loss)
        return state


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
from .serialize import migrate_spec
from .ontostate import OntoState, state_init, load_params, update, train
from ontologize.fns.keys import get_loss, get_srctype
from ontologize.data.loaders import HFDataSource, SampleLoader
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
    # dead-entry revival probability per head per sample, the tag-axis
    # counterpart of `p_drop` (see `DictBlock.revive_dead`). Only bites
    # under a hard `select` rule, where an entry outside every sample's
    # support gets no gradient and so can never return; the dense rules
    # keep every entry recoverable and it is a no-op for them.
    p_revive: float = 0.0
    # an entry counts as starved, and so eligible for revival, when it is
    # selected at less than this fraction of the rate uniform usage would
    # give it (`k_sel / k`). Below 1 so a merely below-average entry is
    # left alone and revival is a no-op on a healthy head.
    revive_frac: float = 0.5

    n_stats: int = 18
    lossfn: str = "mse"
    # optional path to a (d_out,) npy of per-dim MSE weights (e.g.
    # inverse-variance for target whitening, normalized to mean 1). Applied
    # as sqrt-weights to output/target/ghost before the loss, so MSE
    # equal-weights dims instead of letting high-variance dims dominate.
    mse_weights: Optional[str] = None

    # scaling values for loss components
    s_g: float = 0.0
    s_L1K: float = 0.0 #disrecommended; tends to paradoxically cause exploding L1_K
    # router L1, which needs `scaled`. Distinct from `s_L1F`: that one
    # reduces `hfwd(P, S)` and so prices the product `S * |W| * c`, which
    # either factor can pay down. This prices the router alone, leaving the
    # dictionary's magnitude to reconstruction -- the separation a gated
    # architecture rests on. Off by default; `scaled` is off too.
    s_L1S: float = 0.0
    s_L1F: float = 0.0
    # setpoint control for `s_L1F`. A fixed coefficient has no equilibrium:
    # L1's gradient magnitude is constant in `F` while MSE's vanishes as
    # reconstruction approaches its floor, so whatever the coefficient, the
    # penalty eventually walks the code down to whatever bound exists (0
    # with free dictionary rows, `h*l` under `norm_rows`) and holds it
    # there. Which coefficient looks best therefore depends only on the
    # step budget, not on any tradeoff. Setting `L1F_target` above 0 makes
    # `s_L1F` the controlled variable instead: it is adjusted per step to
    # hold `L1_F` at the target, so the operating point is chosen directly
    # and the coefficient is whatever sustains it. `s_L1F` is then the
    # starting value rather than the applied one.
    L1F_target: float = 0.0
    # controller gain, a step in log(s) per unit log-error. The loop is
    # slow on purpose: `L1_F` responds to `s_L1F` over thousands of steps,
    # so a gain fast enough to track batch noise oscillates.
    L1F_eta: float = 1e-3
    # EMA horizon for the measured `L1_F` the controller reacts to.
    L1F_ema: float = 0.99
    # steps over which the setpoint ramps from the `L1_F` measured at the
    # first step down to `L1F_target` (`anneal_steps` when 0). Without the
    # ramp the loop is asked for a setpoint an order of magnitude away, and
    # a pure integrator against a plant that responds over thousands of
    # steps answers by running to its bound and overshooting -- measured at
    # every gain tried, costing ~37% MSE against a fixed coefficient at the
    # same `L1_F`. See `ontostate.l1f_target`.
    L1F_ramp: int = 0
    # bounds on the controlled `s_L1F`, so a stuck loop cannot run away.
    L1F_min: float = 1e-12
    L1F_max: float = 1e-3
    s_H: float = 0.0
    s_bcossim: float = 0.0
    s_hcossim: float = 0.0
    # within-head dictionary row collinearity (`DictBlock.rowcos`): the
    # mean off-diagonal cosine between a head's entries, summed over
    # layers. Targets head death directly, where `s_Hm` prices only one
    # route to it -- a head whose entries all decode to the same direction
    # scores a perfect `KL_m` while carrying no information. Scale-free, so
    # unlike `s_L1F` it constrains directions without also shrinking the
    # code, and it reads the weights alone rather than any batch statistic.
    s_kcossim: float = 0.0
    # mean off-diagonal cosine over the dictionary FLATTENED: every row
    # pair, not just pairs inside a head (`DictBlock.flatcos`). The
    # complement of `s_kcossim`, which is within-head by construction and
    # so blind to heads whose rows duplicate another head's. Neither
    # substitutes for the other: only `k-1` of a row's `h*k-1` partners
    # share its head, so this one is dominated by the between-head pairs.
    s_flatcos: float = 0.0
    # setpoint control for `s_kcossim`, the counterpart of `L1F_target`. A
    # fixed weight has the same non-stationarity: |grad MSE| decays ~25x
    # over a run while |grad cossim_k| does not, so any constant is
    # over-applied late -- a 10x weight reduction moved the endpoint MSE
    # cost by 3 points. The controlled variable is the per-layer MAX, since
    # the summed stat falls monotonically through a collapse.
    KCOS_target: float = 0.0
    KCOS_eta: float = 1e-3
    KCOS_ema: float = 0.99
    KCOS_ramp: int = 0
    KCOS_min: float = 1e-12
    KCOS_max: float = 1e-2
    # batch mean-entropy bonus: penalizes KL(E_batch[p] ‖ uniform) per head
    # (bits, mean over heads, summed over layers). Maximizes the entropy of
    # the *mean* classification, not the mean of the entropies: per-sample
    # hardness is untouched, only usage imbalance is taxed. Anchors the
    # corpus-generic code point at the uniform mixture and opposes
    # frozen-head collapse (a frozen head pays the full log2(k) bits, with
    # pressure applied before its softmax saturates and gradients vanish).
    s_Hm: float = 0.0
    # PWAK consensus KL: pulls classifications toward the batch's
    # partition-gated neighbourhood consensus (design intent, measured
    # behaviour, and caveats in `ontologize.fns.pwak`; the diffusion depth
    # schedule is `pwak_s`/`pwak_s_start`/`pwak_tau` below).
    s_pwak: float = 0.0
    # PWAK noise2self partition score: the error of predicting each
    # layer's own (stop-gradiented) input from its partition-gated
    # neighbourhood, as a fraction of the batch's spread. Shares the
    # `pwak_s`/`pwak_tau` schedule with `s_pwak` and is inert at
    # `pwak_s=0`. The only gradient path is the partition gate, so this
    # scores the clustering rather than the dictionary: it can be reduced
    # only by co-assigning samples that predict each other, and that
    # pressure grows with the diffusion depth. Unlike `s_pwak` it has no
    # log and composes with `select="top<k>"`.
    s_L2pwak: float = 0.0

    seed: int = 42
    grad_clip: Optional[float] = 1.0

    # ghost-gradient dead-feature resurrection. When False, training runs
    # `Ontologizer.withStats` instead of `withGhost`, skipping the parallel
    # ghost forward (~a full extra model pass per step) and reporting the
    # MSE_ghost stat as 0. Note ghostgrad only fires for features that are
    # exactly 0 across the whole batch: reachable with relu-gated
    # classifiers or "top<k>" selection (whose masked tags are exact
    # zeros), structurally impossible with bilinear (gate="none")
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
    # PWAK diffusion depth, ramped LINEARLY pwak_s_start -> pwak_s over the
    # same horizon. Each step is one application of the partition-gated
    # transition matrix, so cost is linear in s and the target's pull away
    # from P is bounded by the graph rather than by a temperature constant.
    # Keep it small (2-4): the gating keeps the walk block-diagonal, but
    # soft blocks leak, and as s grows a connected graph washes everything
    # toward its stationary distribution; also only s=1 is strictly
    # self-excluding (noise2self) -- for s >= 2, even-length walks readmit
    # the self-edge, damped. `pwak_tau` is the heat-kernel width of the
    # affinity (0.2 is the value calibrated on the discourse operator).
    pwak_s: int = 0
    pwak_s_start: int = 0
    pwak_tau: float = 0.2

    def s_loss(self, *args, **kwargs) -> Tuple[Float[Array, "11"],
                                               Tuple[bool, bool, bool, bool,
                                                     bool, bool, bool, bool,
                                                     bool, bool]]:
        """Returns a vector of scaling values for loss components and whether each is nonzero.
        This prevents backpropagation through functions used for evaluation but not training."""
        # the 0 is `cossim_k`'s per-head MAX: logged as the quantity a
        # setpoint loop constrains, never itself penalized (the mean beside
        # it is the penalty, so every head receives gradient)
        s = jnp.array([self.s_g, self.s_L1K, self.s_L1F, self.s_H,
                       self.s_bcossim, self.s_hcossim, self.s_kcossim, 0.0,
                       self.s_Hm, self.s_pwak, self.s_L2pwak,
                       self.s_flatcos, self.s_L1S],
                      *args, **kwargs)
        isloss = (self.s_g != 0.0, self.s_L1K != 0.0, self.s_L1F != 0.0,
                  self.s_H != 0.0, self.s_bcossim != 0.0, self.s_hcossim != 0.0,
                  self.s_kcossim != 0.0, self.s_Hm != 0.0, self.s_pwak != 0.0,
                  self.s_L2pwak != 0.0, self.s_flatcos != 0.0,
                  self.s_L1S != 0.0)
        return s, isloss

    def rng(self) -> PRNGKeyArray:
        """Creates PRNGKey from `self.seed`."""
        return jax.random.PRNGKey(self.seed)

    def loss(self, X: Float[Array, "... b d_out"],
             Y: Float[Array, "... b d_out"], X_g: Float[Array, "... b d_out"],
             stats: Float[Array, "4"],
             s_L1F: Optional[Float[Array, ""]] = None,
             s_kcossim: Optional[Float[Array, ""]] = None
             ) -> Tuple[Float[Array, ""], Float[Array, "n_stats"]]:
        """Accepts model output `X`, target `Y`, ghost gradient output `X_g`,
        and validation statistics `stats`. `L2` is defined as `self.lossfn(X, Y)`.
        `L2_g` is defined as `self.lossfn(X_g, Y - X)`. Total loss `L` is defined as
        `L2 + s_g * L2_g + sum([s_L1K, s_L1F, s_H, s_bcossim, s_hcossim, s_Hm, s_pwak, s_L2pwak] * stats)`.

        The row is append-only: a new stat goes on the end so a shorter
        row still aligns column for column (`visualize.loss` reads the
        missing tail as NaN). Column 16 is the flattened row cosine and
        17 the router's L1.

        `s_L1F` and `s_kcossim` override the fields of the same name with
        traced scalars, so a setpoint controller can vary them per step
        without retracing `update` (which holds `lossfn` static). Both
        applied values trail the returned stats row either way, along with
        the per-layer MAX of `cossim_k`.

        That max is logged because the summed `cossim_k` cannot see a head
        collapse: the layer that collapses is one of `l`, and the others
        improve faster than it degrades, so the sum falls throughout while
        the max rises. Controlling the sum would report success while the
        pathology proceeds -- the same way `hmean_kl` scores a collapsed
        head perfectly."""
        f = get_loss(self.lossfn)
        s, _ = self.s_loss() # scaling vector; ignore isloss
        if s_L1F is not None:
            s = s.at[2].set(jnp.asarray(s_L1F, s.dtype))
        if s_kcossim is not None:
            s = s.at[6].set(jnp.asarray(s_kcossim, s.dtype))
        kcos_max = jnp.max(stats[..., 6])   # per-head max, over layers too
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
        # the controlled multipliers and the max trail the row, so a loop's
        # trajectory is recoverable from loss.csv and a resume can pick it
        # back up there
        # column order is append-only: 0..15 are the historical layout,
        # so every reader and every older loss.csv still lines up, and
        # the flattened row cosine lands at 16
        stats = jnp.append(jnp.stack([L, L2]),
                           jnp.concatenate([stats[:11],
                                            jnp.stack([s[2], s[6],
                                                       kcos_max]),
                                            stats[11:]]))
        return L, stats

    def opt(self, *args, **kwargs):
        """Adam optimizer with learning rate `self.lr`. If `self.wd` is nonzero, returns AdamW.
        """
        if self.wd:
            return optax.adamw(self.lr, weight_decay=self.wd, *args, **kwargs)
        return optax.adam(self.lr, *args, **kwargs)

    def ontologizer(self, *args, **kwargs):
        _, (ghost_loss, sparse_K, sparse_F, entropy_loss, bcossim_loss,
            cossim_loss, kcossim_loss, hmean_loss, pwak_loss,
            l2pwak_loss, flatcos_loss, sparse_S) = self.s_loss()
        return Ontologizer(
                *args, **kwargs, sparse_K=sparse_K, sparse_F=sparse_F,
                sparse_S=sparse_S,
                entropy_loss=entropy_loss, cossim_loss=cossim_loss, bcossim_loss=bcossim_loss,
                kcossim_loss=kcossim_loss, flatcos_loss=flatcos_loss,
                hmean_loss=hmean_loss, pwak_loss = pwak_loss,
                l2pwak_loss=l2pwak_loss,
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
                p_drop=self.p_drop, p_revive=self.p_revive,
                revive_frac=self.revive_frac,
                p_drop_start=self.p_drop_start, sd_K_end=self.sd_K_end,
                pwak_s=self.pwak_s, pwak_s_start=self.pwak_s_start,
                pwak_tau=self.pwak_tau,
                s_L1F=self.s_L1F, L1F_target=self.L1F_target,
                L1F_eta=self.L1F_eta, L1F_ema=self.L1F_ema,
                L1F_min=self.L1F_min, L1F_max=self.L1F_max,
                L1F_ramp=self.L1F_ramp,
                s_kcossim=self.s_kcossim, KCOS_target=self.KCOS_target,
                KCOS_eta=self.KCOS_eta, KCOS_ema=self.KCOS_ema,
                KCOS_min=self.KCOS_min, KCOS_max=self.KCOS_max,
                KCOS_ramp=self.KCOS_ramp,
                grad_clip=self.grad_clip)

    def load(self, manager: ocp.CheckpointManager, step):
        """Initializes an `OntoState`, then loads a checkpoint specified by `step`."""
        spec = manager.restore(step, items={'spec': None})['spec']
        model = Ontologizer(**migrate_spec(spec))
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

    # how `self.mc4`'s per-language interleave ends, passed through to
    # `interleave_datasets`. "first_exhausted" caps every language at the
    # size of the smallest split and discards the rest;
    # "all_exhausted" repeats the small languages instead. Inert on the
    # cached-embedding path, which reads a `.npy` written earlier.
    stopping_strategy: str = "first_exhausted"

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

    def mc4(self, src: str = "allenai/c4", *args, text_key: str = "text",
            **kwargs):
        """The per-language interleaved corpus, wrapped for Grain.

        Applies `self.stopping_strategy`; the rest passes through to
        `load_dataset` (`split`, `streaming`, ...). `datasets` is
        imported here rather than at module scope so the cached path,
        which never builds a corpus, does not pay for it."""
        from ontologize.data.multilingual import mc4_data
        return HFDataSource(
            mc4_data(src, *args,
                     stopping_strategy=self.stopping_strategy, **kwargs),
            text_key=text_key)

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
                    # the controller's multiplier is not in the checkpoint;
                    # pick it back up from the last row it logged, so a
                    # resumed run continues the loop instead of restarting
                    # it from the configured starting value. The logged
                    # value is the PROJECTED one, which is legitimately 0
                    # whenever the constraint was slack at that step -- a
                    # multiplicative update can never leave 0, so fall back
                    # to the configured value in that case.
                    if lines:
                        cols = lines[-1].strip().split(",")
                        if len(cols) >= self.hyper.n_stats:
                            for attr, tgt, i in (
                                    ("s_L1F", self.hyper.L1F_target, 13),
                                    ("s_kcossim", self.hyper.KCOS_target, 14)):
                                if tgt <= 0.0:
                                    continue
                                v = float(cols[i])
                                if v > 0.0:
                                    setattr(self.hyper, attr, v)
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
                checkpoint_each=self.meta.checkpoint_each,
                encoder=encoder, decoder=decoder,
                *self.args_loss, **self.kwargs_loss)
        return state


"""Inference-time causal steering and evaluation wrapper for Ontologizer models.

This module provides the `Steerable` class for restoring a trained `Ontologizer`
model from Orbax checkpoints and executing forward passes under causal interventions
(clamping, scaling, adding, or ablating learned ontofeatures) configured via
per-layer `DictIntervention` specifications. It also provides the `ChatEnv` dataclass
bundling a `Steerable` instance with its training hyperparameters and metadata.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
from jaxtyping import Array, Bool, Float, UInt, PRNGKeyArray

from ontologize.ontologizer import Ontologizer, DictIntervention
from ontologize.training.config import Hyperparams, Metadata
from ontologize.training.ontostate import load_params

class Steerable:
    """Load pretrained `Ontologizer` with interventions."""
    def __init__(self, hyper: Hyperparams, meta: Metadata, *args, **kwargs):
        """Restore checkpoint and initialize steerable model environment.

        Args:
            hyper: Hyperparameter configuration specifying noise, temperature, and optimizer state.
            meta: Metadata specifying checkpoint paths and step numbers.
            *args: Additional positional arguments forwarded to `meta.manager`.
            **kwargs: Additional keyword arguments forwarded to `meta.manager`.
        """
        self.hyper = hyper
        self.meta = meta

        manager = self.meta.manager(*args, **kwargs)
        if self.meta.resume_from is None:
            self.meta.resume_from = manager.latest_step()

        spec = manager.restore(self.meta.resume_from, items={'spec': None})['spec']
        self.model = Ontologizer(**spec)
        self.layer_args = [DictIntervention() for _ in range(self.model.l)]

        state = self.hyper.init(self.model, save_each=self.meta.save_each)
        self.state = load_params(state, manager, self.meta.resume_from)

    def __call__(self, X: Float[Array, "... d_in"], *args, **kwargs
                 ) -> Tuple[Float[Array, "... d_out"], Float[Array, "... (h k)"],
                            Float[Array, "l 5"], Optional[PRNGKeyArray]]:
        """Run `self.model` with interventions."""
        return self.model.apply(
                self.state.params, X, *args, **kwargs, method=Ontologizer.withArgs,
                arglist=self.layer_args, temperature=self.hyper.temperature, 
                sd_in=self.hyper.sd_in, sd_K=self.hyper.sd_K, sd_F=self.hyper.sd_F)

@dataclass
class ChatEnv:
    """Container bundling a `Steerable` model with its configuration metadata.

    Note:
        This dataclass shares the name `ChatEnv` with the interactive terminal REPL
        in `ontologize.chat.ChatEnv`.

    Attributes:
        model: Initialized `Steerable` inference wrapper.
        hyper: Associated `Hyperparams` configuration.
        meta: Associated `Metadata` configuration.
    """
    model: Steerable
    hyper: Hyperparams
    meta: Metadata

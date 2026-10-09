# Functions to query `dict`s of classes by key. This is a workaround for serializing 
# `dataclasses` with fields referencing `Callable`s, which can't be written to JSON.

import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable

from .classify import softmax_cl, ste
from .loss import l2, addnoise, addnoise_batchnorm, addnoise_featvar, identity
from ontologize.data.loaders import SampleLoader, EmbeddingLoader, ImageLoader, TextLoader

# Helper function to map string to JAX activation function
def get_activation(name: str) -> Callable:
    """Activation functions used by `ontologize.layers.*`, also the
    selection rules: "argmax" and "ste" are both the straight-through
    estimator. Unknown names fall back to the identity."""
    activations = {
        "none": identity,
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "silu": jax.nn.silu,
        "tanh": jax.numpy.tanh,
        "sigmoid": jax.nn.sigmoid,
        "softmax": softmax_cl,
        "argmax": ste,
        "ste": ste,
    }
    return activations.get(name.lower(), identity)

def get_dtype(name: str) -> Any:
    """`jax.numpy` float types."""
    dtypes = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
        "float64": jnp.float64,
    }
    return dtypes.get(name.lower(), jnp.float32)

def get_loss(fn: str) -> Callable:
    """Loss functions used by `Hyperparams`."""
    losses = {
        "mse": l2,
        "crossentropy": optax.softmax_cross_entropy,
        "crossentropy_int": optax.softmax_cross_entropy_with_integer_labels,
        "binary_crossentropy": optax.sigmoid_binary_cross_entropy
        }
    return losses.get(fn.lower(), l2)

def get_srctype(name: str) -> Callable:
    """Training data types used to specify `SampleLoader` subtype."""
    srctypes = {
        "embedding": EmbeddingLoader,
        "image": ImageLoader,
        "text": TextLoader
        }
    return srctypes.get(name.lower(), SampleLoader)

def get_noise(name:str) -> Callable:
    """Noise functions by name, for the input, the logits and the features.
    Unknown names add none."""
    noisefns = {
        "none": identity,
        "standard": addnoise,
        "normal": addnoise,
        "gaussian": addnoise,
        "batchnorm": addnoise_batchnorm,
        "featvar": addnoise_featvar,
        }
    return noisefns.get(name.lower(), identity)

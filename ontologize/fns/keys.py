# Functions to query `dict`s of classes by key. This is a workaround for serializing 
# `dataclasses` with fields referencing `Callable`s, which can't be written to JSON.

import functools
import re

import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable

from .classify import softmax_cl, ste, topk_cl
from .loss import l2, addnoise, addnoise_batchnorm, addnoise_featvar, identity
from ontologize.data.loaders import SampleLoader, EmbeddingLoader, ImageLoader, TextLoader

def _lookup(table: dict, name: str, what: str):
    """Strict keyed lookup.

    These resolvers used to be `table.get(name.lower(), <default>)`, which
    made every typo and every not-yet-implemented name silently resolve to
    the neutral option: `--gate swish` trained with no gate, `gaussain`
    trained with no noise, `bf16` trained in float32. The defaults are all
    reachable by their own key ("none", "mse", "float32", "embedding"), so
    nothing legitimate needs the fallback, and a wrong name is always a
    mistake worth stopping for.
    """
    try:
        return table[name.lower()]
    except KeyError:
        raise KeyError(
            f"unknown {what} {name!r}; implemented: "
            f"{', '.join(sorted(table))}") from None


# Helper function to map string to JAX activation function
def get_activation(name: str) -> Callable:
    """Activations functions used by `ontologize.layers.*`. Keys stay
    JSON-serializable strings, so parameterized functions are spelled
    into the key: "top<k>" ("top1", "top4", ...) returns `topk_cl`
    bound to that k."""
    match = re.fullmatch(r"top(\d+)", name.lower())
    if match:
        return functools.partial(topk_cl, k=int(match.group(1)))
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
    return _lookup(activations, name, "activation")

def get_dtype(name: str) -> Any:
    """`jax.numpy` float types."""
    dtypes = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
        "float64": jnp.float64,
    }
    return _lookup(dtypes, name, "dtype")

def get_loss(fn: str) -> Callable:
    """Loss functions used by `Hyperparams`."""
    losses = {
        "mse": l2,
        "crossentropy": optax.softmax_cross_entropy,
        "crossentropy_int": optax.softmax_cross_entropy_with_integer_labels,
        "binary_crossentropy": optax.sigmoid_binary_cross_entropy
        }
    return _lookup(losses, fn, "loss fn")

def get_srctype(name: str) -> Callable:
    """Training data types used to specify `SampleLoader` subtype."""
    srctypes = {
        "embedding": EmbeddingLoader,
        "image": ImageLoader,
        "text": TextLoader
        }
    return _lookup(srctypes, name, "srctype")

def get_noise(name:str) -> Callable:
    """Type of noise to add to the training data."""
    noisefns = {
        "none": identity,
        "standard": addnoise,
        "normal": addnoise,
        "gaussian": addnoise,
        "batchnorm": addnoise_batchnorm,
        "featvar": addnoise_featvar,
        }
    return _lookup(noisefns, name, "noise fn")

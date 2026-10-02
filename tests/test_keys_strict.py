"""`fns/keys.py` resolvers reject unknown names instead of defaulting.

Every resolver used to be `table.get(name.lower(), <neutral>)`, so a name
that was mistyped or simply not implemented resolved to the do-nothing
option and training continued: `--gate swish` ran with no gate, noise
`"gaussain"` ran with no noise, dtype `"bf16"` ran in float32. Nothing
legitimate needs that fallback -- every neutral option has its own key --
so these pin the strict behaviour and the reachability of the defaults.
"""
import jax
import jax.numpy as jnp
import pytest

from ontologize.data.loaders import EmbeddingLoader, ImageLoader, TextLoader
from ontologize.fns.keys import (get_activation, get_dtype, get_loss,
                                 get_noise, get_srctype)
from ontologize.fns.loss import identity, l2

RESOLVERS = [
    (get_activation, "activation", "swish"),
    (get_dtype, "dtype", "bf16"),
    (get_loss, "loss fn", "huber"),
    (get_srctype, "srctype", "tokens"),
    (get_noise, "noise fn", "gaussain"),
]


@pytest.mark.parametrize("fn,what,bad", RESOLVERS)
def test_unknown_name_raises(fn, what, bad):
    with pytest.raises(KeyError) as e:
        fn(bad)
    msg = str(e.value)
    assert bad in msg, "the error must name the offending value"
    assert what in msg, "and say which table rejected it"
    assert "implemented:" in msg, "and list the valid options"


@pytest.mark.parametrize("fn,what,bad", RESOLVERS)
def test_empty_string_raises(fn, what, bad):
    """`""` used to resolve to the neutral option too."""
    with pytest.raises(KeyError):
        fn("")


def test_neutral_options_remain_reachable_by_key():
    """Strictness must not cost the behaviour the fallback used to give."""
    assert get_activation("none") is identity
    assert get_noise("none") is identity
    assert get_loss("mse") is l2
    assert get_dtype("float32") is jnp.float32
    assert get_srctype("embedding") is EmbeddingLoader


def test_known_names_still_resolve():
    assert get_activation("sigmoid") is jax.nn.sigmoid
    assert get_activation("relu") is jax.nn.relu
    assert get_dtype("bfloat16") is jnp.bfloat16
    assert get_srctype("image") is ImageLoader
    assert get_srctype("text") is TextLoader
    assert get_noise("batchnorm").__name__ == "addnoise_batchnorm"


def test_case_is_still_ignored():
    assert get_activation("SIGMOID") is jax.nn.sigmoid
    assert get_dtype("BFloat16") is jnp.bfloat16


def test_topk_pattern_still_bypasses_the_table():
    """`top<k>` is parameterized into the key, so it is not a table entry."""
    f = get_activation("top4")
    assert f.keywords == {"k": 4}
    with pytest.raises(KeyError):
        get_activation("topk")          # not a number: not the pattern

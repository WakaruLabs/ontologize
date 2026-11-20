# The layers keep float32 parameters by default. In bfloat16 the grid near a
# weight w is about |w|/128 wide, so any update below |w|/256 rounds away:
# a throttled classifier's Adam steps (about 1e-6 against weights near
# 0.036) would all be lost, and the layer would stop rather than slow.
# Ontologizer and DictEnc pass their own float32 down; these defaults are
# what a script that builds a layer on its own gets.
import jax
import jax.numpy as jnp
import pytest

from ontologize.layers.linear import Linear
from ontologize.layers.nlinear import Bilinear, BilinearBlock, NLinear, NLinearBlock

D_IN, D_OUT, H = 6, 4, 3
LAYERS = {
    "Linear": lambda: Linear(d_in=D_IN, d_out=D_OUT),
    "NLinear": lambda: NLinear(d_in=D_IN, d_out=D_OUT),
    "Bilinear": lambda: Bilinear(d_in=D_IN, d_out=D_OUT),
    "NLinearBlock": lambda: NLinearBlock(d_in=D_IN, d_out=D_OUT, h=H),
    "BilinearBlock": lambda: BilinearBlock(d_in=D_IN, d_out=D_OUT, h=H),
}


@pytest.mark.parametrize("make", LAYERS.values(), ids=LAYERS.keys())
def test_parameters_default_to_float32(make):
    p = make().init(jax.random.PRNGKey(0), jnp.zeros((2, D_IN)))
    assert all(x.dtype == jnp.float32 for x in jax.tree_util.tree_leaves(p))


def test_a_small_step_survives_in_float32_and_not_in_bfloat16():
    w, step = 0.036, 1e-6
    assert jnp.float32(w) + jnp.float32(step) != jnp.float32(w)
    assert jnp.bfloat16(w) + jnp.bfloat16(step) == jnp.bfloat16(w)

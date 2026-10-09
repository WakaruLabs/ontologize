import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Array, Float

from .sparse import Sparse
from ontologize.fns.keys import get_activation, get_dtype
from ontologize.fns.loss import ghostgrad

class Linear(Sparse):
    """Linear layer with optional bias and activation function. Has `fwd` and `rev` 
    methods for forward and reverse passes without bias or activation functions.
    These are used for computing ghost gradients over multiple layers.

    output = activation(Wx) + b
    """
    d_in: int = 0
    d_out: int = 0
    biased: bool = True

    activation: str = "none"
    sparse: bool = False
    noise: str = "none"
    sd: float = 0.0
    dtype_str: str = "bfloat16"
    dtype_p_str: str = "bfloat16"

    def setup(self):
        """Initialize model parameters. Fetch `self.fn`, `self.dtype`,
        and `self.dtype_p` by key. This is a workaround for `Callable`s
        and `jax.numpy` types not being serializable to JSON."""
        super().setup()

        self.weights = self.param(
            'weights',
            nn.initializers.lecun_normal(in_axis=-1, out_axis=-2),
            (self.d_out, self.d_in),
            dtype=self.dtype_p
        )
        if self.biased:
            self.bias = self.param(
                'bias',
                nn.initializers.zeros,
                (self.d_out,),
                dtype=self.dtype_p
            )
        else:
            self.bias = None

    def fwd(self, X: Float[Array, "... d_in"]) -> Float[Array, "... d_out"]:
        """Forward pass without applying bias or activation function."""
        X = X.astype(self.dtype)
        W = self.weights.astype(self.dtype)
        return jnp.einsum("oi, ...i -> ...o", W, X)

    def rev(self, Y: Float[Array, "... d_out"]) -> Float[Array, "... d_in"]:
        """Reverse pass without applying bias or activation function."""
        Y = Y.astype(self.dtype)
        W = self.weights.astype(self.dtype)
        return jnp.einsum("oi, ...o -> ...i", W, Y)

    def __call__(self, X: Float[Array, "... d_in"]) -> Float[Array, "... d_out"]:
        Y = self.fn(self.fwd(X))
        return self.addbias(Y, self.bias)

    def ghost(self, X: Float[Array, "... d_in"], Y: Float[Array, "... d_out"],
              lbound: int=-10.0, ubound: int=10.0) -> Float[Array, "... d_out"]:
        X = X.astype(self.dtype)
        Y = Y.astype(self.dtype)
        W = self.weights.astype(self.dtype)

        return ghostgrad(W, X, Y)

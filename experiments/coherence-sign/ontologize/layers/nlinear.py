#Bilinear layer classes

import jax
import jax.numpy as jnp
import flax.linen as nn
import einops
from typing import Callable, Tuple, Any, Optional
from jaxtyping import Array, Float, UInt
from .sparse import Sparse
from .linear import get_activation, get_dtype
from ontologize.fns.loss import ghostgrad

class NLinear(Sparse):
    """Generalization of `Bilinear` to arbitrary `n`. Like `Linear`, has `fwd`
    methods for computing ghost gradients. Unlike `Linear` and `Bilinear`,
    it does not have a `rev` method, as eigendecomposition is unimplemented."""
    d_in: int = 0
    d_out: int = 0
    n: int = 2
    biased: bool = True
    activation: str = "none"
    gate: str = "none"
    dtype_str: str = "bfloat16"
    dtype_p_str: str = "bfloat16"

    def setup(self):
        super().setup()
        self.fn_gate = get_activation(self.gate)
        # weight shape in PyTorch was (d_out, d_in, n)
        # We'll stick to similar but use Flax's parameter initialization
        self.weight = self.param(
            'weight',
            nn.initializers.lecun_normal(in_axis=-1, out_axis=-2, batch_axis=(0,)),
            (self.n, self.d_out, self.d_in),
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

    def nfwd(self, X: Float[Array, "... d_in"]) -> Float[Array, "n ... d_out"]:
        X = X.astype(self.dtype)
        W = self.weight.astype(self.dtype)
        Ys = jnp.einsum("...m,ndm->n...d", X, W)
        return Ys

    def fwd(self, X: Float[Array, "... d_in"]) -> Float[Array, "... d_out"]:
        Ys = self.nfwd(X)
        return self.fn_gate(Ys[0]) * jnp.prod(Ys[1:], axis=0)

    def __call__(self, X: Float[Array, "... d_in"]) -> Float[Array, "... d_out"]:
        Y = self.fn(self.fwd(X))
        return self.addbias(Y, self.bias)

    def weight_ubind(self) -> Tuple[Float[Array, "d_out d_in"], ...]:
        """Unbind weights as a `Tuple` of `n` matrices"""
        # returns W, V for n=2
        W = self.weight.astype(self.dtype)
        return tuple(W)

    def ghost(self, X: Float[Array, "... d_in"], Y: Float[Array, "... d_out"],
              *args, **kwargs) -> Float[Array, "... d_out"]:
        """Map `fns.loss.ghostgrad` over `n`."""
        W = self.weight.astype(self.dtype)
        X = X.astype(self.dtype)
        Y = Y.astype(self.dtype)
        Ys = jax.vmap(lambda W_i: ghostgrad(W_i, X, Y, *args, **kwargs))(W)
        return jnp.prod(Ys, axis=0)

class Bilinear(NLinear):
    """Bilinear layer with optional bias and activation function 
    as described by Pearce et al. (2025)."""
    def setup(self):
        object.__setattr__(self, 'n', 2)
        super().setup()

    def bilinearTensor(self) -> Float[Array, "d_out d_in d_in"]:
        """Returns the bilinear tensor as defined in Pearce et al. (2025)."""
        W, V = self.weight_ubind()
        B = jnp.einsum("ke, kd -> ked", W, V)
        return 0.5 * (B + jnp.swapaxes(B, 1, 2))

    def interactionMat(self, Y_0: Float[Array, "... d_out"]) -> Float[Array, "... d_in d_in"]:
        """Interaction matrix used to compute eigenfeatures."""
        Y = Y_0.astype(self.dtype)
        B = self.bilinearTensor()
        return jnp.einsum("...k, ked -> ...ed", Y, B)

    def jacobian(self, X: Float[Array, "... d_in"]) -> Float[Array, "... d_out d_in"]:
        X = X.astype(self.dtype)
        W, V = self.weight_ubind() # Each is (d_out, d_in)

        # Compute the two linear projections
        xW = jnp.einsum("...i, oi -> ...o", X, W)
        xV = jnp.einsum("...i, oi -> ...o", X, V)

        # J_k = (xV_k) * W_k + (xW_k) * V_k
        # Result shape: (..., d_out, d_in)
        J = jnp.einsum("...o, oi -> ...oi", xV, W) + jnp.einsum("...o, oi -> ...oi", xW, V)
        return J

    def decompose(self) -> Tuple[Float[Array, "d_out d_in"], Float[Array, "d_out d_in d_in"]]:
        """Eigendecomposition of `self.bilinearTensor()`."""
        B = self.bilinearTensor()
        # jnp.linalg.eigh can be mapped over the first dimension
        eigenvals, eigenvecs = jax.vmap(jnp.linalg.eigh)(B)
        return eigenvals, eigenvecs

    def rev(self, Y: Float[Array, "... d_out"]) -> Float[Array, "... d_in"]:
        """Approximate reverse pass: maps an output back to input space
        through each output unit's top eigenvector (largest `abs`
        eigenvalue), dropping the eigenvalue and its sign."""
        vals, vecs = self.decompose()
        top1 = jnp.argmax(jnp.abs(vals), axis=1)
        # eigh returns eigenvectors as columns: v[:, i] pairs with vals[i]
        vecs_top1 = jax.vmap(lambda v, i: v[:, i])(vecs, top1)
        return jnp.dot(Y.astype(self.dtype), vecs_top1)

    def project(self, Y_0: Float[Array, "... d_out"]
                ) -> Tuple[Float[Array, "... d_out d_in"], 
                           Float[Array, "... d_out d_in d_in"]]:
        """Eigendecomposition of an output of the layer."""
        Y = Y_0.astype(self.dtype)
        vals, vecs = self.decompose()
        vecs = jnp.einsum("ked, ...k -> ...ked", vecs, Y)
        return vals, vecs

class NLinearBlock(Sparse):
    """Multihead version of `NLinear`."""
    d_in: int = 0
    d_out: int = 0
    h: int = 0
    n: int = 2
    biased: bool = True
    activation: str = "none"
    gate: str = "none"
    dtype_str: str = "bfloat16"
    dtype_p_str: str = "bfloat16"

    def setup(self):
        super().setup()
        self.fn_gate = get_activation(self.gate)
        self.weight = self.param(
            'weight',
            nn.initializers.lecun_normal(in_axis=-1, out_axis=-2, batch_axis=(0, 1)),
            (self.n, self.h, self.d_out, self.d_in),
            dtype=self.dtype_p
        )
        if self.biased:
            self.bias = self.param(
                'bias',
                nn.initializers.zeros,
                (self.h, self.d_out),
                dtype=self.dtype_p
            )
        else:
            self.bias = None

    def nfwd(self, X: Float[Array, "... d_in"]) -> Float[Array, "n ... h d_out"]:
        """As `NLinear.nfwd`, but broadcast over the `h` axis."""
        X = X.astype(self.dtype)
        W = self.weight.astype(self.dtype)
        Ys = jnp.einsum("...m,nhdm->n...hd", X, W)
        return Ys

    def fwd(self, X: Float[Array, "... d_in"]) -> Float[Array, "... h d_out"]:
        """As `NLinear.fwd`, but broadcast over the `h` axis."""
        Ys = self.nfwd(X)
        return self.fn_gate(Ys[0]) * jnp.prod(Ys[1:], axis=0)

    def __call__(self, X: Float[Array, "... d_in"]) -> Float[Array, "... h d_out"]:
        # X: (..., d_in) -> (..., h, d_out)
        Y = self.fn(self.fwd(X))
        return self.haddbias(Y, self.bias)

    def weight_ubind(self) -> Tuple[Float[Array, "h d_out d_in"], ...]:
        W = self.weight.astype(self.dtype)
        return tuple(W)

    def ghost(self, X: Float[Array, "... d_in"], Y: Float[Array, "... h d_out"],
              *args, **kwargs) -> Float[Array, "... d_out"]:
        """Map `fns.loss.ghostgrad` over `n` and `h`."""
        W = self.weight.astype(self.dtype)
        X = X.astype(self.dtype)
        Y = Y.astype(self.dtype)
        def f(W_ij, Y_j):
            return ghostgrad(W_ij, X, Y_j, *args, **kwargs)

        Ys = jax.vmap(lambda W_i: jax.vmap(f, (0, -2), -2)(W_i, Y))(W)
        return jnp.prod(Ys, axis=0)

    def rev(self, Y: Float[Array, "... h d_out"]) -> Float[Array, "... d_in"]:
        """Adjoint reverse pass, summed over heads. Only defined for `n=1`
        (a plain per-head linear map), where the adjoint is the transpose;
        the gate is ignored."""
        if self.n != 1:
            raise NotImplementedError(
                "NLinearBlock.rev requires n=1; use BilinearBlock for n=2")
        W = self.weight_ubind()[0]
        Y = Y.astype(self.dtype)
        return jnp.einsum("...hd, hdm -> ...m", Y, W)


class BilinearBlock(NLinearBlock):
    """Multihead version of `Bilinear`."""
    def setup(self):
        object.__setattr__(self, 'n', 2)
        super().setup()

    def bilinearTensor(self) -> Float[Array, "h d_out d_in d_in"]:
        """As `Bilinear.bilinearTensor`, but broadcast over the `h` axis."""
        W, V = self.weight_ubind()
        B = jnp.einsum("hke, hkf -> hkef", W, V)
        return 0.5 * (B + jnp.swapaxes(B, -1, -2))

    def jacobian(self, x: Float[Array, "... d_in"]) -> Float[Array, "... h d_out d_in"]:
        """As `Bilinear.jacobian`, but broadcast over the `h` axis."""
        x = x.astype(self.dtype)
        W, V = self.weight_ubind() # Each is (h, d_out, d_in)

        # Compute the two linear projections
        xW = jnp.einsum("...i, hoi -> ...ho", x, W)
        xV = jnp.einsum("...i, hoi -> ...ho", x, V)

        # J_k = (xV_k) * W_k + (xW_k) * V_k
        # Result shape: (..., h, d_out, d_in)
        J = jnp.einsum("...ho, hoi -> ...hoi", xV, W
                       ) + jnp.einsum("...ho, hoi -> ...hoi", xW, V)
        return J

    def interactionMat(self, Y_0: Float[Array, "... h d_out"]
                       ) -> Float[Array, "... h d_in d_in"]:
        """As `Bilinear.interactionMat`, but broadcast over the `h` axis."""
        Y = Y_0.astype(self.dtype)
        B = self.bilinearTensor()
        return jnp.einsum("...hk, hkef -> ...hef", Y, B)

    def decompose(self) -> Tuple[Float[Array, "h d_out d_in"],
                                 Float[Array, "h d_out d_in d_in"]]:
        """As `Bilinear.decompose`, but broadcast over the `h` axis."""
        B = self.bilinearTensor()
        # B has shape (h, k, d, d) -> we want to map over k and h
        eigenvals, eigenvecs = jax.vmap(jax.vmap(jnp.linalg.eigh))(B)
        return eigenvals, eigenvecs

    def project(self, Y_0: Float[Array, "... h d_out"]
                ) -> Tuple[Float[Array, "... h d_out d_in"], 
                           Float[Array, "... h d_out d_in d_in"]]:
        """As `Bilinear.project`, but broadcast over the `h` axis."""
        Y = Y_0.astype(self.dtype)
        vals, vecs = self.decompose()
        vecs = jnp.einsum("hkef, ...hk -> ...hkef", vecs, Y)
        return vals, vecs

    def rev(self, Y: Float[Array, " ... h d_out"]) -> Float[Array, "... d_in"]:
        """As `Bilinear.rev` (top eigenvector, eigenvalue dropped), but
        broadcast over the `h` axis and summed over heads."""
        vals, vecs = self.decompose()
        top1 = jnp.argmax(jnp.abs(vals), axis=-1)         # (h, d_out)
        # eigh returns eigenvectors as columns: gather v[..., :, top1]
        vecs_top1 = jnp.take_along_axis(
            vecs, top1[..., None, None], axis=-1).squeeze(-1)  # (h, d_out, d_in)
        return jnp.einsum("...hk, hkd -> ...d", Y.astype(self.dtype), vecs_top1)

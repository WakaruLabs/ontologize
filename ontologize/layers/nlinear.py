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

def orient(V: Float[Array, "... d"], lam: Float[Array, "... 1"]
           ) -> Float[Array, "... d"]:
    """Give an eigenvector a well-defined sign.

    `eigh` fixes eigenvectors only up to sign, so a `rev` built on them
    returns a direction whose sign is a property of the LAPACK routine
    rather than of the layer. Two factors settle it: the eigenvalue's
    sign, which says whether the form grows or shrinks along the vector,
    and a tie-break putting the largest-magnitude coordinate positive,
    which is what makes the result reproducible at all.

    This cannot supply the sign a STEERING caller wants. Moving from `x`
    along `v` changes the logit `x'Bx` by `2e*lam*(x.v) + e^2*lam`, so
    the ascent direction depends on `x.v` and is a property of the
    sample, not of the weights. A weight-only linear reverse has no
    access to it; `x'Bx`'s gradient does (`Bilinear.jacobian`)."""
    i = jnp.argmax(jnp.abs(V), axis=-1, keepdims=True)
    lead = jnp.take_along_axis(V, i, axis=-1)
    s = jnp.where(lead < 0, -1.0, 1.0) * jnp.where(lam < 0, -1.0, 1.0)
    return V * s.astype(V.dtype)


def excite(W: Float[Array, "... d"], V: Float[Array, "... d"]
           ) -> Tuple[Float[Array, "... d"], Float[Array, "... 1"]]:
    """Each form `sym(w v^T)`'s positive-eigenvalue eigenvector (unit) and
    that eigenvalue, in closed form.

    The form has rank at most 2: its eigenvalues are |w||v|(cos +- 1)/2,
    on w/|w| +- v/|v|. Along w/|w| + v/|v| the form grows; along the other
    eigenvector it shrinks, and that one has the larger magnitude whenever
    w.v < 0, so a top-|eigenvalue| rule returns a direction that switches
    the output off. Where w = -cv the form is negative semidefinite, no
    direction excites it, and the vector is zero."""
    nw = jnp.linalg.norm(W, axis=-1, keepdims=True)
    nv = jnp.linalg.norm(V, axis=-1, keepdims=True)
    U = W / (nw + 1e-12) + V / (nv + 1e-12)
    lam = 0.5 * (nw * nv + (W * V).sum(-1, keepdims=True))
    return U / jnp.maximum(jnp.linalg.norm(U, axis=-1, keepdims=True), 1e-12), lam


class NLinear(Sparse):
    """Generalization of `Bilinear` to arbitrary `n`. Like `Linear`, has `fwd`
    methods for computing ghost gradients. Unlike `Linear` and `Bilinear, 
    it does not have a `rev` method, as eigendecomposition is unimplemented.
    `gate` is a function only applied to the first slice along the `n` dimension."""
    d_in: int = 0
    d_out: int = 0
    n: int = 2
    biased: bool = True
    activation: str = "none"
    gate: str = "none"
    dtype_str: str = "bfloat16"
    # float32 master weights: in bfloat16 an Adam step below |w|/256 rounds away
    dtype_p_str: str = "float32"

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
        """`self.fwd` without reducing the `n` dimension."""
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

    def withL1(self, X: Float[Array, "... d_in"]
               ) -> Tuple[Float[Array, "... d_out"], Float[Array, ""]]:
        """As `NLinear.__call__` except for returning L1 loss calculated from the gate mask."""
        Ys = self.nfwd(X)
        mask = self.fn_gate(Ys[0])
        L1 = self.l1(mask)
        Y = self.fn(mask * jnp.prod(Ys[1:], axis=0))
        return self.addbias(Y, self.bias), L1

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
        """Approximate reverse pass: each output's exciting direction, the
        positive-eigenvalue eigenvector of its form (`excite`), weighted by
        `Y`. The sign is `orient`'s convention."""
        W, V = self.weight.astype(jnp.float32)
        vecs, lam = excite(W, V)
        return jnp.dot(Y.astype(self.dtype), orient(vecs, lam))

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
    # standardize each head's outputs across `d_out` (zero mean, unit SD),
    # after the activation and the bias. As a classifier this is
    # `logit_norm`: the logits' scale is fixed, so a temperature alone sets
    # the selection's spread. `__call__` and `withL1` apply it, so every
    # caller gets the same logits; the adjoint and eigen-analyses describe
    # the forms before it, which order the entries the same way.
    standardize: bool = False
    dtype_str: str = "bfloat16"
    # float32 master weights: in bfloat16 an Adam step below |w|/256 rounds away
    dtype_p_str: str = "float32"

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

    def standardized(self, Y: Float[Array, "... h d_out"]
                     ) -> Float[Array, "... h d_out"]:
        """Y standardized across each head's outputs when `standardize`.
        The variance floor is a fixed 1e-12, not finfo.eps: bilinear
        logits at initialization have a variance near 1/d_in^2, which eps
        would bias."""
        if not self.standardize:
            return Y
        Y = Y.astype(self.dtype)
        mu = Y.mean(-1, keepdims=True)
        return (Y - mu) / jnp.sqrt(((Y - mu) ** 2).mean(-1, keepdims=True)
                                   + 1e-12)

    def __call__(self, X: Float[Array, "... d_in"]) -> Float[Array, "... h d_out"]:
        # X: (batchsize, m) -> (batchsize, d, h)
        Y = self.fn(self.fwd(X))
        return self.standardized(self.haddbias(Y, self.bias))

    def withL1(self, X: Float[Array, "... d_in"]
               ) -> Tuple[Float[Array, "... h d_out"], Float[Array, ""]]:
        """As `NLinearBlock.__call__` except for returning L1 loss calculated from the gate mask."""
        Ys = self.nfwd(X)
        mask = self.fn_gate(Ys[0])
        L1 = self.l1(mask)
        Y = self.fn(mask * jnp.prod(Ys[1:], axis=0))
        return self.standardized(self.haddbias(Y, self.bias)), L1

    def weight_ubind(self) -> Tuple[Float[Array, "h d_out d_in"], ...]:
        W = self.weight.astype(self.dtype)
        return tuple(W)

    def ghost(self, X: Float[Array, "... d_in"], Y: Float[Array, "... h d_out"],
              *args, **kwargs) -> Float[Array, "... d_out"]:
        """Map `fns.loss.ghostgrad` over `n` and `h`."""
        if self.standardize:
            raise NotImplementedError(
                "the ghost gradient does not pass through `standardize`")
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
        W, V = self.weight_ubind() # Each is (d_out, d_in)

        # Compute the two linear projections
        xW = jnp.einsum("...i, hoi -> ...ho", x, W)
        xV = jnp.einsum("...i, hoi -> ...ho", x, V)

        # J_k = (xV_k) * W_k + (xW_k) * V_k
        # Result shape: (..., d_out, d_in)
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
        """As `Bilinear.rev`, but broadcast over the `h` axis and summed over heads."""
        W, V = self.weight.astype(jnp.float32)            # each (h, d_out, d_in)
        vecs, lam = excite(W, V)
        return jnp.einsum("...hk, hkd -> ...d", Y.astype(self.dtype),
                          orient(vecs, lam))

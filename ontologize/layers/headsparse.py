"""Per-head sparse encoders and bilinear classifiers.

Implements:
- `jumprelu`: JumpReLU activation with straight-through estimator gradients (Rajamanoharan et al., 2024).
- `step`: Gated step function with custom VJP for L0 sparsity penalty.
- `HeadSparseEncoder`: Multihead sparse encoder supporting JumpReLU and Top-K sparsity modes.
- `HeadBilinear`: Multihead bilinear classifier mapping per-head inputs to logits.
"""

from typing import Union
import jax
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Array, Float

from ontologize.fns.keys import get_dtype
from .sparse import Sparse


def _reduce_to_shape(grad: jnp.ndarray, target_shape: tuple[int, ...]) -> jnp.ndarray:
    """Reduces grad to match target_shape by summing over broadcast axes."""
    if len(target_shape) == 0:
        return jnp.sum(grad)
    lead_dims = grad.ndim - len(target_shape)
    if lead_dims > 0:
        grad = jnp.sum(grad, axis=tuple(range(lead_dims)))
    for i, (dim_g, dim_t) in enumerate(zip(grad.shape, target_shape)):
        if dim_t == 1 and dim_g > 1:
            grad = jnp.sum(grad, axis=i, keepdims=True)
    return grad.reshape(target_shape)


@jax.custom_vjp
def jumprelu(
    x: Float[Array, "..."],
    threshold: Union[Float[Array, "..."], float],
    bandwidth: Union[Float[Array, "..."], float] = 1e-3,
) -> Float[Array, "..."]:
    """JumpReLU activation function.

    Computes `x * (x > threshold)`.

    Args:
        x: Input tensor.
        threshold: Threshold tensor or scalar.
        bandwidth: Bandwidth parameter for the straight-through estimator kernel.

    Returns:
        Gated tensor where elements <= threshold are set to 0.
    """
    return jnp.where(x > threshold, x, 0.0).astype(x.dtype)


def _jumprelu_fwd(x, threshold, bandwidth=1e-3):
    return jumprelu(x, threshold, bandwidth), (x, threshold, bandwidth)


def _jumprelu_bwd(res, g):
    x, threshold, bandwidth = res
    dtype = x.dtype if jnp.issubdtype(x.dtype, jnp.floating) else jnp.float32
    u = (x - threshold) / bandwidth
    K = (jnp.abs(u) <= 0.5).astype(dtype)
    gx = (g * (x > threshold)).astype(dtype)
    gt_unreduced = g * (-(threshold / bandwidth) * K)
    t_shape = jnp.shape(threshold)
    gt = _reduce_to_shape(gt_unreduced, t_shape).astype(
        threshold.dtype if hasattr(threshold, "dtype") else gt_unreduced.dtype
    )
    gb = jnp.zeros_like(bandwidth)
    return gx, gt, gb


jumprelu.defvjp(_jumprelu_fwd, _jumprelu_bwd)


@jax.custom_vjp
def step(
    x: Float[Array, "..."],
    threshold: Union[Float[Array, "..."], float],
    bandwidth: Union[Float[Array, "..."], float] = 1e-3,
) -> Float[Array, "..."]:
    """Unit step function with straight-through estimator gradients for L0 penalties.

    Computes `(x > threshold)` as float.

    Args:
        x: Input tensor.
        threshold: Threshold tensor or scalar.
        bandwidth: Bandwidth parameter for the straight-through estimator kernel.

    Returns:
        Step indicator tensor as float.
    """
    dtype = x.dtype if jnp.issubdtype(x.dtype, jnp.floating) else jnp.float32
    return (x > threshold).astype(dtype)


def _step_fwd(x, threshold, bandwidth=1e-3):
    return step(x, threshold, bandwidth), (x, threshold, bandwidth)


def _step_bwd(res, g):
    x, threshold, bandwidth = res
    dtype = x.dtype if jnp.issubdtype(x.dtype, jnp.floating) else jnp.float32
    u = (x - threshold) / bandwidth
    K = (jnp.abs(u) <= 0.5).astype(dtype)
    gx = jnp.zeros_like(x)
    gt_unreduced = g * (-(1.0 / bandwidth) * K)
    t_shape = jnp.shape(threshold)
    gt = _reduce_to_shape(gt_unreduced, t_shape).astype(
        threshold.dtype if hasattr(threshold, "dtype") else gt_unreduced.dtype
    )
    gb = jnp.zeros_like(bandwidth)
    return gx, gt, gb


step.defvjp(_step_fwd, _step_bwd)


class HeadSparseEncoder(nn.Module):
    """Per-head sparse encoder module.

    Projects input activations into per-head feature spaces and applies either
    JumpReLU or Top-K sparsity.
    """
    h: int
    d_in: int
    m: int
    mode: str = "jumprelu"
    k_z: int = 4
    bandwidth: float = 1e-3
    init_threshold: float = 1e-3
    dtype_str: str = "float32"
    # auxiliary SAE objective: a decoder W_dec (h, m, d_in) with tied init
    # (W_dec = W_enc^T, rows unit-normed) so every active latent gets a dense
    # reconstruction gradient instead of only what reaches it through the
    # classifiers' straight-through selection. `auxk` > 0 adds Gao et al.
    # (2024)'s AuxK term: the top-auxk *inactive* latents (by preactivation)
    # reconstruct the residual of the main reconstruction, reviving dead ones.
    decoder: bool = False
    auxk: int = 0

    def setup(self):
        """Initializes encoder weights, biases, and log-threshold parameters."""
        self.dtype = get_dtype(self.dtype_str)
        self.W_enc = self.param(
            "W_enc",
            nn.initializers.lecun_normal(in_axis=-1, out_axis=-2, batch_axis=(0,)),
            (self.h, self.m, self.d_in),
            dtype=self.dtype,
        )
        if self.decoder:
            W = self.W_enc
            self.W_dec = self.param(
                "W_dec",
                lambda rng, shape: (W / (jnp.linalg.norm(W, axis=-1, keepdims=True) + 1e-8)).astype(self.dtype),
                (self.h, self.m, self.d_in),
            )
            self.b_dec = self.param("b_dec", nn.initializers.zeros, (self.h, self.d_in), dtype=self.dtype)
        self.b_enc = self.param(
            "b_enc",
            nn.initializers.zeros,
            (self.h, self.m),
            dtype=self.dtype,
        )
        self.log_threshold = self.param(
            "log_threshold",
            nn.initializers.constant(jnp.log(self.init_threshold)),
            (self.h, self.m),
            dtype=self.dtype,
        )

    def preacts(self, x: Float[Array, "... d_in"]) -> Float[Array, "... h m"]:
        """Computes affine pre-activations for each head.

        Args:
            x: Input tensor of shape `(..., d_in)`.

        Returns:
            Pre-activations of shape `(..., h, m)`.
        """
        x = x.astype(self.dtype)
        W = self.W_enc.astype(self.dtype)
        b = self.b_enc.astype(self.dtype)
        return jnp.einsum("hmi,...i->...hm", W, x) + b

    def __call__(self, x: Float[Array, "... d_in"]) -> Float[Array, "... h m"]:
        """Computes non-negative sparse activations.

        Args:
            x: Input tensor of shape `(..., d_in)`.

        Returns:
            Sparse activations `z >= 0` of shape `(..., h, m)`.
        """
        pre = self.preacts(x)
        if self.mode == "jumprelu":
            threshold = jnp.exp(self.log_threshold)
            return jumprelu(pre, threshold, self.bandwidth)
        elif self.mode == "topk":
            a = jax.nn.relu(pre)
            k = min(self.k_z, self.m)
            if k <= 0:
                return jnp.zeros_like(a)
            if k >= self.m:
                return a
            thr = jax.lax.top_k(a, k)[0][..., -1:]
            return jnp.where(a >= thr, a, 0.0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'jumprelu' or 'topk'.")

    def l0(self, x: Float[Array, "... d_in"]) -> Float[Array, ""]:
        """Computes mean count of active features per sample across heads.

        Args:
            x: Input tensor of shape `(..., d_in)`.

        Returns:
            Scalar float representing the mean active feature count per sample.
        """
        if self.mode == "jumprelu":
            pre = self.preacts(x)
            threshold = jnp.exp(self.log_threshold)
            s = step(pre, threshold, self.bandwidth)
        elif self.mode == "topk":
            z = self(x)
            s = jax.lax.stop_gradient((z > 0).astype(self.dtype))
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'jumprelu' or 'topk'.")

        per_sample_count = s.sum(axis=(-2, -1))
        return per_sample_count.mean()

    def decode(self, z: Float[Array, "... h m"]) -> Float[Array, "... h d_in"]:
        """Per-encoder reconstruction of the input from the sparse code."""
        return jnp.einsum("...hm,hmi->...hi", z.astype(self.dtype), self.W_dec.astype(self.dtype)) \
            + self.b_dec.astype(self.dtype)

    def recon_loss(self, x: Float[Array, "... d_in"]) -> Float[Array, ""]:
        """Auxiliary SAE loss: per-element MSE of each encoder's reconstruction
        of its input (same units as the model's output MSE for unit-scale
        inputs), averaged over encoders, plus AuxK / 32 when `auxk` > 0."""
        x = x.astype(self.dtype)
        z = self(x)
        xh = self.decode(z)
        err = x[..., None, :] - xh                        # (..., h, d_in)
        loss = jnp.mean(err ** 2)
        if self.auxk > 0:
            pre = self.preacts(x)
            inactive = jnp.where(z > 0, -jnp.inf, pre)    # only currently inactive latents
            k = min(self.auxk, self.m)
            thr = jax.lax.top_k(inactive, k)[0][..., -1:]
            z_aux = jnp.where(inactive >= thr, jax.nn.relu(pre), 0.0)
            e = jax.lax.stop_gradient(err)
            eh = jnp.einsum("...hm,hmi->...hi", z_aux, self.W_dec.astype(self.dtype))
            loss = loss + jnp.mean((e - eh) ** 2) / 32.0
        return loss


class HeadBilinear(Sparse):
    """Per-head bilinear classifier on per-head inputs.

    Maps per-head sparse features `z` of shape `(..., h, m)` to per-head logits
    of shape `(..., h, k)` using bilinear projections `(W z) * (V z) + bias`.
    A `Sparse` subclass so `DictEnc` can use it in place of `NLinearBlock`
    (logit noise, the L1 stat); it has no ghost or reverse pass.
    """
    h: int = 0
    m: int = 0
    k: int = 0
    biased: bool = False
    dtype_str: str = "float32"

    def setup(self):
        """Initializes bilinear projection weights `W`, `V` and optional `bias`."""
        super().setup()
        self.W = self.param(
            "W",
            nn.initializers.lecun_normal(in_axis=-1, out_axis=-2, batch_axis=(0,)),
            (self.h, self.k, self.m),
            dtype=self.dtype,
        )
        self.V = self.param(
            "V",
            nn.initializers.lecun_normal(in_axis=-1, out_axis=-2, batch_axis=(0,)),
            (self.h, self.k, self.m),
            dtype=self.dtype,
        )
        if self.biased:
            self.bias = self.param(
                "bias",
                nn.initializers.zeros,
                (self.h, self.k),
                dtype=self.dtype,
            )
        else:
            self.bias = None

    def __call__(self, z: Float[Array, "... h m"]) -> Float[Array, "... h k"]:
        """Evaluates per-head bilinear interaction.

        Args:
            z: Input tensor of shape `(..., h, m)`.

        Returns:
            Logits tensor of shape `(..., h, k)`.
        """
        z = z.astype(self.dtype)
        W = self.W.astype(self.dtype)
        V = self.V.astype(self.dtype)
        if z.shape[-2] == 1 and self.h > 1:
            # one code shared by all heads: contract without materializing
            # the (..., h, m) broadcast (671 MB per layer at m=20480, b=256)
            zW = jnp.einsum("hkm,...m->...hk", W, z[..., 0, :])
            zV = jnp.einsum("hkm,...m->...hk", V, z[..., 0, :])
        else:
            zW = jnp.einsum("hkm,...hm->...hk", W, z)
            zV = jnp.einsum("hkm,...hm->...hk", V, z)
        logits = zW * zV
        if self.biased and self.bias is not None:
            logits = logits + self.bias.astype(self.dtype)
        return logits

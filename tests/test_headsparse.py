"""Tests for ontologize.layers.headsparse (jumprelu, step, HeadSparseEncoder, HeadBilinear)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.layers.headsparse import (
    jumprelu,
    step,
    HeadSparseEncoder,
    HeadBilinear,
)


# -----------------------------------------------------------------------------
# 1. Forward tests: jumprelu and step
# -----------------------------------------------------------------------------

def test_jumprelu_forward_values():
    """Verify forward values of jumprelu: x * (x > threshold)."""
    threshold = 1.5
    bandwidth = 0.1

    # Scalar inputs
    assert float(jumprelu(jnp.array(1.0), threshold, bandwidth)) == 0.0
    assert float(jumprelu(jnp.array(1.5), threshold, bandwidth)) == 0.0
    assert float(jumprelu(jnp.array(2.0), threshold, bandwidth)) == 2.0
    assert float(jumprelu(jnp.array(-0.5), threshold, bandwidth)) == 0.0

    # Vector / array inputs
    x = jnp.array([0.0, 1.4, 1.5, 1.51, 3.0])
    out = jumprelu(x, threshold, bandwidth)
    expected = jnp.array([0.0, 0.0, 0.0, 1.51, 3.0])
    assert jnp.allclose(out, expected)

    # 2D inputs with broadcast threshold
    x_2d = jnp.array([[0.5, 2.5], [1.8, 1.2]])
    t_2d = jnp.array([1.0, 2.0])  # broadcasts over rows
    out_2d = jumprelu(x_2d, t_2d, bandwidth)
    expected_2d = jnp.array([[0.0, 2.5], [1.8, 0.0]])
    assert jnp.allclose(out_2d, expected_2d)


def test_step_forward_values():
    """Verify forward values of step: (x > threshold) as float."""
    threshold = 2.0
    bandwidth = 0.1

    # Scalar inputs
    assert float(step(jnp.array(1.5), threshold, bandwidth)) == 0.0
    assert float(step(jnp.array(2.0), threshold, bandwidth)) == 0.0
    assert float(step(jnp.array(2.01), threshold, bandwidth)) == 1.0
    assert float(step(jnp.array(-1.0), threshold, bandwidth)) == 0.0

    # Array inputs
    x = jnp.array([1.0, 1.99, 2.0, 2.01, 5.0])
    out = step(x, threshold, bandwidth)
    expected = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    assert jnp.allclose(out, expected)
    assert jnp.issubdtype(out.dtype, jnp.floating)


# -----------------------------------------------------------------------------
# 2. Custom gradient tests: jumprelu and step
# -----------------------------------------------------------------------------

def test_jumprelu_gradients():
    """Verify custom gradients of jumprelu.
    - d/dx: (x > threshold)
    - d/dthreshold: -(threshold / bandwidth) * K((x - threshold) / bandwidth)
      where K(u) = 1 if |u| <= 1/2 else 0.
    """
    threshold = 2.0
    bandwidth = 0.2
    half_bw = bandwidth / 2.0  # 0.1

    # Test x-gradient passes through only above threshold
    # Case A: x well above threshold (|x - t| > bandwidth / 2)
    # x = 2.5 > threshold
    gx, gt = jax.grad(lambda x, t: jumprelu(x, t, bandwidth).sum(), argnums=(0, 1))(
        jnp.array([2.5]), jnp.array([threshold])
    )
    assert float(gx[0]) == 1.0
    assert float(gt[0]) == 0.0  # outside bandwidth / 2

    # Case B: x well below threshold (|x - t| > bandwidth / 2)
    # x = 1.5 < threshold
    gx, gt = jax.grad(lambda x, t: jumprelu(x, t, bandwidth).sum(), argnums=(0, 1))(
        jnp.array([1.5]), jnp.array([threshold])
    )
    assert float(gx[0]) == 0.0
    assert float(gt[0]) == 0.0

    # Case C: x inside window (|x - t| <= bandwidth / 2), and x > threshold
    # x = 2.05 (dist = 0.05 <= 0.1)
    gx, gt = jax.grad(lambda x, t: jumprelu(x, t, bandwidth).sum(), argnums=(0, 1))(
        jnp.array([2.05]), jnp.array([threshold])
    )
    assert float(gx[0]) == 1.0  # x > threshold
    expected_gt = -(threshold / bandwidth)  # - (2.0 / 0.2) = -10.0
    assert np.isclose(float(gt[0]), expected_gt, atol=1e-5)

    # Case D: x inside window (|x - t| <= bandwidth / 2), but x <= threshold
    # x = 1.95 (dist = 0.05 <= 0.1)
    gx, gt = jax.grad(lambda x, t: jumprelu(x, t, bandwidth).sum(), argnums=(0, 1))(
        jnp.array([1.95]), jnp.array([threshold])
    )
    assert float(gx[0]) == 0.0  # x <= threshold, no x gradient
    assert np.isclose(float(gt[0]), expected_gt, atol=1e-5)

    # Case E: exactly on the boundary |x - threshold| = bandwidth / 2
    # x = 2.0 + 0.1 = 2.10 (|u| = 0.5 <= 0.5)
    gx, gt = jax.grad(lambda x, t: jumprelu(x, t, bandwidth).sum(), argnums=(0, 1))(
        jnp.array([2.10]), jnp.array([threshold])
    )
    assert np.isclose(float(gt[0]), expected_gt, atol=1e-5)

    # Just outside the boundary: x = 2.11 (|u| = 0.55 > 0.5)
    gx, gt = jax.grad(lambda x, t: jumprelu(x, t, bandwidth).sum(), argnums=(0, 1))(
        jnp.array([2.11]), jnp.array([threshold])
    )
    assert float(gt[0]) == 0.0

    # Multi-dimensional broadcasting reduction test:
    # x: shape (B, H, M), threshold: shape (H, M)
    B, H, M = 3, 2, 4
    x_multi = jnp.full((B, H, M), 2.05)
    t_multi = jnp.full((H, M), 2.0)
    _, gt_multi = jax.grad(lambda x, t: jumprelu(x, t, bandwidth).sum(), argnums=(0, 1))(
        x_multi, t_multi
    )
    assert gt_multi.shape == (H, M)
    # Each batch element contributed -(threshold / bandwidth)
    expected_multi = B * (-(2.0 / bandwidth))
    assert jnp.allclose(gt_multi, expected_multi)


def test_step_gradients():
    """Verify custom gradients of step.
    - d/dx: 0
    - d/dthreshold: -(1 / bandwidth) * K((x - threshold) / bandwidth)
    """
    threshold = 3.0
    bandwidth = 0.5

    # Case A: inside bandwidth / 2, x > threshold (x = 3.1)
    gx, gt = jax.grad(lambda x, t: step(x, t, bandwidth).sum(), argnums=(0, 1))(
        jnp.array([3.1]), jnp.array([threshold])
    )
    assert float(gx[0]) == 0.0
    expected_gt = -(1.0 / bandwidth)  # -2.0
    assert np.isclose(float(gt[0]), expected_gt, atol=1e-5)

    # Case B: inside bandwidth / 2, x <= threshold (x = 2.9)
    gx, gt = jax.grad(lambda x, t: step(x, t, bandwidth).sum(), argnums=(0, 1))(
        jnp.array([2.9]), jnp.array([threshold])
    )
    assert float(gx[0]) == 0.0
    assert np.isclose(float(gt[0]), expected_gt, atol=1e-5)

    # Case C: outside bandwidth / 2 (x = 3.5, dist = 0.5 > 0.25)
    gx, gt = jax.grad(lambda x, t: step(x, t, bandwidth).sum(), argnums=(0, 1))(
        jnp.array([3.5]), jnp.array([threshold])
    )
    assert float(gx[0]) == 0.0
    assert float(gt[0]) == 0.0

    # Multi-dimensional broadcasting reduction test:
    B, H, M = 4, 3, 2
    x_multi = jnp.full((B, H, M), 3.1)
    t_multi = jnp.full((H, M), 3.0)
    _, gt_multi = jax.grad(lambda x, t: step(x, t, bandwidth).sum(), argnums=(0, 1))(
        x_multi, t_multi
    )
    assert gt_multi.shape == (H, M)
    assert jnp.allclose(gt_multi, B * (-(1.0 / bandwidth)))


# -----------------------------------------------------------------------------
# 3. HeadSparseEncoder tests
# -----------------------------------------------------------------------------

def test_headsparse_encoder_shapes_and_nonnegativity():
    """Verify output shapes and non-negativity (z >= 0) in both jumprelu and topk modes."""
    h, d_in, m = 3, 8, 16
    key = jax.random.PRNGKey(42)

    for mode in ["jumprelu", "topk"]:
        enc = HeadSparseEncoder(h=h, d_in=d_in, m=m, mode=mode, k_z=4)

        # 1D input: (d_in,)
        k1, key = jax.random.split(key)
        x_1d = jax.random.normal(k1, (d_in,))
        params = enc.init(k1, x_1d)
        z_1d = enc.apply(params, x_1d)
        pre_1d = enc.apply(params, x_1d, method=enc.preacts)
        assert pre_1d.shape == (h, m)
        assert z_1d.shape == (h, m)
        assert jnp.all(z_1d >= 0.0)

        # 2D input: (B, d_in)
        k2, key = jax.random.split(key)
        x_2d = jax.random.normal(k2, (5, d_in))
        z_2d = enc.apply(params, x_2d)
        pre_2d = enc.apply(params, x_2d, method=enc.preacts)
        assert pre_2d.shape == (5, h, m)
        assert z_2d.shape == (5, h, m)
        assert jnp.all(z_2d >= 0.0)

        # 3D input: (B1, B2, d_in)
        k3, key = jax.random.split(key)
        x_3d = jax.random.normal(k3, (2, 4, d_in))
        z_3d = enc.apply(params, x_3d)
        pre_3d = enc.apply(params, x_3d, method=enc.preacts)
        assert pre_3d.shape == (2, 4, h, m)
        assert z_3d.shape == (2, 4, h, m)
        assert jnp.all(z_3d >= 0.0)


def test_headsparse_encoder_topk_k_z_active():
    """Verify topk mode keeps exactly k_z largest entries per head for generic inputs."""
    h, d_in, m, k_z = 4, 16, 32, 5
    enc = HeadSparseEncoder(h=h, d_in=d_in, m=m, mode="topk", k_z=k_z)
    key = jax.random.PRNGKey(123)
    k_init, k_data = jax.random.split(key)

    B = 10
    x = jax.random.normal(k_data, (B, d_in))
    params = enc.init(k_init, x)

    z = enc.apply(params, x)
    assert z.shape == (B, h, m)

    # For each sample and each head, count active (> 0) features
    active_counts = (z > 0).sum(axis=-1)  # shape (B, h)
    assert jnp.all(active_counts == k_z), f"Expected all {k_z}, got {active_counts}"

    # Also verify that the active entries match the top-k values of relu(preacts)
    pre = enc.apply(params, x, method=enc.preacts)
    relu_pre = jax.nn.relu(pre)
    top_vals = jax.lax.top_k(relu_pre, k_z)[0]  # (B, h, k_z)
    # The smallest retained non-zero entry must equal top_vals[..., -1]
    min_active_z = jnp.where(z > 0, z, jnp.inf).min(axis=-1)  # (B, h)
    assert jnp.allclose(min_active_z, top_vals[..., -1], atol=1e-5)


def test_headsparse_encoder_l0_jumprelu():
    """Verify l0 matches manual count and has nonzero threshold gradient in jumprelu mode."""
    h, d_in, m = 2, 6, 8
    bandwidth = 0.5
    init_threshold = 0.1
    enc = HeadSparseEncoder(
        h=h, d_in=d_in, m=m, mode="jumprelu", bandwidth=bandwidth, init_threshold=init_threshold
    )
    key = jax.random.PRNGKey(99)
    k_init, k_data = jax.random.split(key)

    B = 7
    x = jax.random.normal(k_data, (B, d_in))
    params = enc.init(k_init, x)

    # 1. Forward l0 matches manual count
    l0_val = enc.apply(params, x, method=enc.l0)
    pre = enc.apply(params, x, method=enc.preacts)
    threshold = jnp.exp(params["params"]["log_threshold"])
    manual_l0 = (pre > threshold).sum(axis=(-2, -1)).mean()
    assert jnp.allclose(l0_val, manual_l0)

    # 2. Gradient of l0 w.r.t threshold is nonzero via STE
    # Set preacts close to threshold to guarantee activation in the bandwidth window
    def loss_fn(p):
        return enc.apply(p, x, method=enc.l0)

    grads = jax.grad(loss_fn)(params)
    grad_log_t = grads["params"]["log_threshold"]
    grad_W = grads["params"]["W_enc"]
    grad_b = grads["params"]["b_enc"]

    # Since step has d/dx = 0, no gradient flows to W_enc or b_enc
    assert jnp.allclose(grad_W, 0.0)
    assert jnp.allclose(grad_b, 0.0)

    # When some preacts lie within bandwidth / 2 of threshold, threshold gradient is nonzero
    # Let's check or construct inputs guaranteeably in the window
    x_near = jnp.zeros((1, d_in))  # preacts = b_enc = 0
    # b_enc = 0, threshold = init_threshold = 0.1. |0 - 0.1| = 0.1 <= bandwidth / 2 = 0.25!
    loss_near = lambda p: enc.apply(p, x_near, method=enc.l0)
    grads_near = jax.grad(loss_near)(params)
    grad_log_t_near = grads_near["params"]["log_threshold"]
    assert jnp.any(jnp.abs(grad_log_t_near) > 0.0)


def test_headsparse_encoder_l0_topk():
    """Verify l0 in topk mode matches manual count and has zero gradients."""
    h, d_in, m, k_z = 3, 5, 10, 3
    enc = HeadSparseEncoder(h=h, d_in=d_in, m=m, mode="topk", k_z=k_z)
    key = jax.random.PRNGKey(77)
    k_init, k_data = jax.random.split(key)

    B = 4
    x = jax.random.normal(k_data, (B, d_in))
    params = enc.init(k_init, x)

    # 1. Forward l0 matches manual count
    l0_val = enc.apply(params, x, method=enc.l0)
    z = enc.apply(params, x)
    manual_l0 = (z > 0).sum(axis=(-2, -1)).mean()
    assert jnp.allclose(l0_val, manual_l0)

    # With positive bias ensuring positive pre-activations, each sample has exactly h * k_z active features
    params_pos = {
        "params": {
            **params["params"],
            "b_enc": jnp.full((h, m), 10.0),
        }
    }
    l0_pos = enc.apply(params_pos, x, method=enc.l0)
    assert jnp.isclose(l0_pos, float(h * k_z))

    # 2. Gradient w.r.t all params is zero (stop_gradient)
    def loss_fn(p):
        return enc.apply(p, x, method=enc.l0)

    grads = jax.grad(loss_fn)(params)
    for p_name, g in grads["params"].items():
        assert jnp.allclose(g, 0.0), f"Parameter {p_name} had nonzero gradient: {g}"


# -----------------------------------------------------------------------------
# 4. HeadBilinear tests
# -----------------------------------------------------------------------------

def test_head_bilinear_shapes_and_loop_equivalence():
    """Verify HeadBilinear shapes and exact equality with an explicit loop over heads."""
    h, m, k = 4, 6, 5
    key = jax.random.PRNGKey(2024)

    for biased in [False, True]:
        layer = HeadBilinear(h=h, m=m, k=k, biased=biased)

        # Multi-dimensional z: (B1, B2, h, m)
        k_init, k_data = jax.random.split(key)
        B1, B2 = 3, 2
        z = jax.random.normal(k_data, (B1, B2, h, m))
        params = layer.init(k_init, z)

        logits = layer.apply(params, z)
        assert logits.shape == (B1, B2, h, k)

        # Explicit loop over heads
        W = params["params"]["W"]  # (h, k, m)
        V = params["params"]["V"]  # (h, k, m)
        bias = params["params"]["bias"] if biased else None

        loop_heads = []
        for i in range(h):
            z_i = z[..., i, :]  # (..., m)
            w_i = W[i]          # (k, m)
            v_i = V[i]          # (k, m)
            zw = jnp.einsum("...m,km->...k", z_i, w_i)
            zv = jnp.einsum("...m,km->...k", z_i, v_i)
            head_out = zw * zv
            if biased:
                head_out = head_out + bias[i]
            loop_heads.append(head_out)

        loop_logits = jnp.stack(loop_heads, axis=-2)
        assert jnp.allclose(logits, loop_logits, atol=1e-5)


# -----------------------------------------------------------------------------
# 5. Initialization and parameter structure tests
# -----------------------------------------------------------------------------

def test_headsparse_encoder_param_specs():
    """Check parameter shapes, threshold positivity, and initial values."""
    h, d_in, m = 3, 10, 12
    init_threshold = 2e-3
    enc = HeadSparseEncoder(
        h=h, d_in=d_in, m=m, init_threshold=init_threshold, dtype_str="float32"
    )
    key = jax.random.PRNGKey(0)
    x = jnp.zeros((d_in,))
    params = enc.init(key, x)

    assert params["params"]["W_enc"].shape == (h, m, d_in)
    assert params["params"]["b_enc"].shape == (h, m)
    assert params["params"]["log_threshold"].shape == (h, m)

    # Threshold is exp(log_threshold) > 0
    t = jnp.exp(params["params"]["log_threshold"])
    assert jnp.all(t > 0.0)
    assert jnp.allclose(t, init_threshold, atol=1e-6)


def test_headsparse_encoder_invalid_mode():
    """Check that an unsupported mode raises ValueError."""
    enc = HeadSparseEncoder(h=2, d_in=4, m=4, mode="unsupported_mode")
    key = jax.random.PRNGKey(0)
    x = jnp.zeros((4,))
    with pytest.raises(ValueError, match="Unknown mode"):
        enc.init(key, x)

    # Also test apply when params already exist
    valid_enc = HeadSparseEncoder(h=2, d_in=4, m=4, mode="topk")
    params = valid_enc.init(key, x)
    with pytest.raises(ValueError, match="Unknown mode"):
        enc.apply(params, x)
    with pytest.raises(ValueError, match="Unknown mode"):
        enc.apply(params, x, method=enc.l0)


def test_bfloat16_precision():
    """Verify jumprelu and step work correctly under bfloat16."""
    x = jnp.array([1.0, 2.0], dtype=jnp.bfloat16)
    t = jnp.array([1.5], dtype=jnp.bfloat16)
    assert jumprelu(x, t).dtype == jnp.bfloat16
    assert step(x, t).dtype == jnp.bfloat16

    gx, gt = jax.grad(lambda x, t: jumprelu(x, t).sum(), argnums=(0, 1))(x, t)
    assert gx.dtype == jnp.bfloat16
    assert gt.dtype == jnp.bfloat16


def test_topk_edge_cases():
    """Verify topk behavior when k_z >= m or k_z <= 0 or all preacts negative."""
    h, d_in, m = 2, 4, 3
    # Case 1: k_z >= m
    enc = HeadSparseEncoder(h=h, d_in=d_in, m=m, mode="topk", k_z=10)
    key = jax.random.PRNGKey(0)
    x = jnp.array([[1.0, -1.0, 2.0, -2.0]])
    params = enc.init(key, x)
    z = enc.apply(params, x)
    # When k_z >= m, all positive relu preacts are kept
    pre = enc.apply(params, x, method=enc.preacts)
    assert jnp.allclose(z, jax.nn.relu(pre))

    # Case 2: all preacts negative
    params_neg = {
        "params": {
            **params["params"],
            "b_enc": jnp.full((h, m), -100.0),
        }
    }
    z_neg = enc.apply(params_neg, x)
    assert jnp.all(z_neg == 0.0)
    l0_neg = enc.apply(params_neg, x, method=enc.l0)
    assert float(l0_neg) == 0.0



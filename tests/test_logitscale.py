# logitscale.py: the closed-form logit scale at initialization must be the
# one a lecun_normal bilinear classifier draws, layer 0's logits must be the
# model's own, the noise must be the classifier's own training noise, and
# each checkpoint must be read under the configuration it trained with.
import sys
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"
                       / "ste-arm"))
from logitscale import (closed_form_sd, hyper_at, logits,  # noqa: E402
                        noise_flips, temperature_noise, top1_margin,
                        within_head_sd)


def shaped(n, d, seed=0):
    """Unit rows with a constant 1 appended, as layer 0's gain-shape input."""
    X = np.random.default_rng(seed).normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return np.concatenate([X, np.ones((n, 1))], 1)


def init_weight(d_in, h=64, k=32, seed=0):
    init = nn.initializers.lecun_normal(in_axis=-1, out_axis=-2,
                                        batch_axis=(0, 1))
    return np.asarray(init(jax.random.PRNGKey(seed), (2, h, k, d_in),
                           jnp.float32), np.float64)


def test_within_head_sd_and_margin():
    K = np.array([[[0.0, 1.0, 3.0], [2.0, 2.0, 2.0]]])
    assert within_head_sd(K) == pytest.approx(np.std([0, 1, 3]) / 2)
    np.testing.assert_allclose(top1_margin(K), [[2.0, 0.0]])


def test_closed_form_is_the_initial_bilinear_logit_scale():
    d = 255
    U = shaped(64, d)
    W = init_weight(d + 1)
    K = np.einsum("bm,hkm->bhk", U, W[0]) * np.einsum("bm,hkm->bhk", U, W[1])
    closed = closed_form_sd((U ** 2).sum(1), d + 1, 2, "none", "none")
    assert closed == pytest.approx(2 / (d + 1))
    assert K.std() == pytest.approx(closed, rel=0.05)


@pytest.mark.parametrize("scale", [1.0, np.sqrt(255)])
def test_closed_form_with_a_sigmoid_gate(scale):
    """Near 1/2 the gate halves the other factor; on the sqrt(d) sphere it
    no longer sits near 1/2, and the closed form must follow it there."""
    d = 255
    U = shaped(64, d)
    U[:, :d] *= scale
    W = init_weight(d + 1, seed=1)
    A = np.einsum("bm,hkm->bhk", U, W[0])
    K = (1 / (1 + np.exp(-A))) * np.einsum("bm,hkm->bhk", U, W[1])
    u_sq = (U ** 2).sum(1)
    closed = closed_form_sd(u_sq, d + 1, 2, "sigmoid", "none")
    if scale == 1.0:
        assert closed == pytest.approx(np.sqrt(2 / (d + 1)) / 2, rel=2e-3)
    assert K.std() == pytest.approx(closed, rel=0.05)


def test_closed_form_scales_to_one_on_the_sqrt_d_sphere():
    d = 255
    U = shaped(8, d)
    U[:, :d] *= np.sqrt(d)
    assert closed_form_sd((U ** 2).sum(1), d + 1, 2, "none", "none") \
        == pytest.approx(1.0)


def test_closed_form_is_undefined_without_a_form():
    u_sq = np.full(4, 2.0)
    assert np.isnan(closed_form_sd(u_sq, 9, 2, "none", "relu"))
    assert np.isnan(closed_form_sd(u_sq, 9, 2, "relu", "none"))
    assert np.isnan(closed_form_sd(u_sq, 9, 1, "sigmoid", "none"))


def test_hyper_at_takes_the_configuration_a_step_trained_under():
    entries = [
        {"type": "env_config", "hyper": {"sd_K": 0.02},
         "meta": {"resume_from": None}},
        {"type": "stats"},
        {"type": "env_config", "hyper": {"sd_K": 0.01},
         "meta": {"resume_from": 371900}},
    ]
    assert hyper_at(entries, 100)["sd_K"] == 0.02
    assert hyper_at(entries, 371900)["sd_K"] == 0.01
    assert hyper_at(entries, 400000)["sd_K"] == 0.01
    with pytest.raises(ValueError):
        hyper_at([{"type": "stats"}], 0)


def test_temperature_noise_follows_the_schedules():
    flat = {"temperature": 1.5e-4, "sd_K": 0.3}
    assert temperature_noise(flat, 10 ** 6) == pytest.approx((1.5e-4, 0.3))
    annealed = {"temperature": 1.0, "temperature_end": 0.01, "sd_K": 0.02,
                "sd_K_end": 0.002, "anneal_steps": 1000}
    T, sd = temperature_noise(annealed, 500)
    assert T == pytest.approx(0.1)
    assert sd == pytest.approx(np.sqrt(0.02 * 0.002))
    assert temperature_noise(annealed, 5000) == pytest.approx((0.01, 0.002))


def test_layer0_logits_are_the_classifiers(build, X):
    model, variables = build(resid_gain=True, resid_const=True)
    params = variables["params"]
    U, K = logits(model, params, np.asarray(X))
    Xn = np.asarray(X, np.float64)
    want_U = np.concatenate(
        [Xn / np.linalg.norm(Xn, axis=1, keepdims=True),
         np.ones((len(Xn), 1))], 1)
    np.testing.assert_allclose(U, want_U, atol=1e-6)
    W = np.asarray(params["dictencs_0"]["classifier"]["weight"], np.float64)
    want_K = (np.einsum("bm,hkm->bhk", want_U, W[0])
              * np.einsum("bm,hkm->bhk", want_U, W[1]))
    np.testing.assert_allclose(K, want_K, rtol=1e-4, atol=1e-7)
    # the shape, not the constant, takes the sqrt(d) scale
    Ur, _ = logits(model, params, np.asarray(X), 4.0)
    np.testing.assert_allclose(Ur[:, :-1], 4 * want_U[:, :-1], atol=1e-5)
    np.testing.assert_allclose(Ur[:, -1], 1.0)


@pytest.mark.parametrize("noise", ["normal", "batchnorm"])
def test_noise_flips_use_the_classifiers_noise(build, X, noise):
    model, variables = build(resid_gain=True, resid_const=True, noise_K=noise)
    params = variables["params"]
    _, K = logits(model, params, np.asarray(X))
    key = jax.random.PRNGKey(0)
    flips0, _ = noise_flips(model, params, K, 0.0, 2, key)
    assert flips0 == 0.0
    big = 1e4 * (1.0 if noise == "batchnorm" else float(np.abs(K).max()))
    flips, ratio = noise_flips(model, params, K, big, 4, key)
    k = K.shape[-1]
    assert flips == pytest.approx(1 - 1 / k, abs=0.1)
    assert ratio > 100

# Noise-function regressions. addnoise_featvar's unguarded gradient was
# (x - mean) / (b * std): a feature with zero batch variance underflowed
# the f32 variance to exactly 0, making it 0/0 = NaN -- one NaN feature
# poisons every parameter through the global loss (killed the SONAR run at
# step 300476; fixed with stop_gradient + eps in e636b22).
import jax
import jax.numpy as jnp

from ontologize.fns.loss import addnoise, addnoise_batchnorm, addnoise_featvar


def batch_with_constant_feature():
    X = jax.random.normal(jax.random.PRNGKey(0), (16, 8), jnp.float32)
    return X.at[:, 3].set(2.0)  # zero variance in the batch


def test_featvar_zero_variance_grad_finite():
    X = batch_with_constant_feature()
    rng = jax.random.PRNGKey(1)

    def loss(w):
        return (addnoise_featvar(X * w, rng, 0.1) ** 2).mean()

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g)


def test_featvar_noise_scale_is_constant():
    # the per-feature std must be a constant under differentiation, or the
    # model can shrink feature variance to evade its own noise
    X = batch_with_constant_feature()
    rng = jax.random.PRNGKey(1)

    def noise_energy(w):
        return ((addnoise_featvar(X * w, rng, 0.1) - X * w) ** 2).sum()

    # noise is linear in the (stop-gradiented) std, so d(energy)/dw would be
    # nonzero if the std carried gradient; with the guard it is exactly 0
    assert float(jax.grad(noise_energy)(1.0)) == 0.0


def test_zero_sd_is_identity():
    X = batch_with_constant_feature()
    rng = jax.random.PRNGKey(1)
    assert jnp.array_equal(addnoise(X, rng, 0.0), X)
    assert jnp.array_equal(addnoise_featvar(X, rng, 0.0), X)
    assert jnp.array_equal(addnoise_batchnorm(X, rng, 0.0), X)


def test_batchnorm_noise_zero_input_finite():
    # per-sample norm of a zero row is 0; must not divide by it
    X = batch_with_constant_feature().at[0].set(0.0)
    rng = jax.random.PRNGKey(1)

    def loss(w):
        return (addnoise_batchnorm(X * w, rng, 0.1) ** 2).mean()

    assert jnp.isfinite(jax.grad(loss)(1.0))

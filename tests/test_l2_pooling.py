# l2_pooling with and without its L2 step. encode_corpus.py caches the masked
# mean (normalize=False), which is SONAR's own embedding; the on-the-fly path
# keeps the unit vector. The two must agree up to a row's norm, the default
# must stay the unit vector, and padding must reach neither.
import jax.numpy as jnp
import numpy as np

from ontologize.data.pretrained import l2_pooling


def batch(seed=0, b=4, n=7, d=5):
    rng = np.random.default_rng(seed)
    E = jnp.asarray(rng.normal(size=(b, n, d)), dtype=jnp.float32)
    lengths = np.array([7, 3, 1, 5])[:b]
    mask = jnp.asarray(np.arange(n)[None, :] < lengths[:, None], dtype=jnp.int32)
    return E, mask, lengths


def test_unnormalized_is_the_masked_mean():
    E, mask, lengths = batch()
    raw = np.asarray(l2_pooling(E, mask, normalize=False))
    E_np = np.asarray(E)
    for i, ell in enumerate(lengths):
        np.testing.assert_allclose(raw[i], E_np[i, :ell].mean(axis=0),
                                   rtol=1e-5, atol=1e-6)


def test_default_is_the_unit_vector_of_the_mean():
    E, mask, _ = batch()
    unit = np.asarray(l2_pooling(E, mask))
    raw = np.asarray(l2_pooling(E, mask, normalize=False))
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(raw / np.linalg.norm(raw, axis=1, keepdims=True),
                               unit, rtol=1e-5, atol=1e-6)


def test_padding_reaches_neither():
    E, mask, _ = batch()
    # change only padded positions: no output may move
    E2 = jnp.where(mask[..., None].astype(bool), E, 1e3)
    for normalize in (True, False):
        np.testing.assert_array_equal(
            np.asarray(l2_pooling(E, mask, normalize=normalize)),
            np.asarray(l2_pooling(E2, mask, normalize=normalize)))

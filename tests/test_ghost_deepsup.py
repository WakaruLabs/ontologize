# Ghost-gradient path under deep supervision. `withGhost` returns the decode
# `Y` and its ghost counterpart `Y_g`; with deepsup, `Y` gains a leading
# prefix axis, and `Y_g` has to gain it too. It previously did not: `Y_g` was
# decoded once from the final ghost residual, so `Hyperparams.loss`'s
# `L2_g = lossfn(Y_g, X - Y)` broadcast one (b, d) array against every prefix
# residual instead of pairing prefix with prefix. It broadcasts silently, so
# only a shape assertion catches it.
#
# ghostgrad fires only for outputs that are exactly 0 across the whole batch,
# which random init never produces. These tests force the condition: the
# residual accumulator is elementwise non-negative (softmax weights over
# abs()'d dictionaries), so an all-negative decoder row plus a relu output
# activation makes that output dim exactly dead for every sample, in every
# prefix -- and its ghost value exp(W_0 . R_i) then varies with the prefix,
# which is precisely what the old code could not represent.
import copy

import jax
import jax.numpy as jnp
import pytest

from ontologize.ontologizer import Ontologizer
from ontologize.training.config import Hyperparams

from conftest import KW, B, LAYER_WIDTH

# relu-gated classifier: the regime where the ghost path is live at all
# (with gate="none" bilinear classifiers it is inert, see Hyperparams.ghost).
GKW = dict(KW, gate="relu", activation_dec="relu")
DEAD = 0  # decoder output row forced dead


def make(deepsup, X):
    model = Ontologizer(**dict(GKW, deepsup=deepsup))
    params = model.init(jax.random.PRNGKey(0), X)
    params = copy.deepcopy(params)
    W = params['params']['decoder']['weights']
    # mildly negative so relu kills the row but W_0 . R stays inside the
    # (-10, 10) clip window, where the ghost value still varies with R
    params['params']['decoder']['weights'] = W.at[DEAD, :].set(-0.01)
    return model, params


def ghost_fwd(model, params, X):
    return model.apply(params, X, method=Ontologizer.withGhost)


@pytest.fixture(scope="module")
def deep(X):
    model, params = make(True, X)
    Y, Y_g, _, _ = ghost_fwd(model, params, X)
    return Y, Y_g


@pytest.fixture(scope="module")
def flat(X):
    model, params = make(False, X)
    Y, Y_g, _, _ = ghost_fwd(model, params, X)
    return Y, Y_g


def test_shapes_match_without_deepsup(flat, X):
    Y, Y_g = flat
    assert Y.shape == (B, KW["d_out"])
    assert Y_g.shape == Y.shape


def test_shapes_match_with_deepsup(deep, X):
    # the regression: Y_g used to stay (B, d_out) while Y became (l, B, d_out)
    Y, Y_g = deep
    assert Y.shape == (KW["l"], B, KW["d_out"])
    assert Y_g.shape == Y.shape


def test_ghost_path_is_live(deep):
    # guards the fixture: if this fails the ghost path is inert and the
    # remaining tests are vacuous rather than wrong
    Y, Y_g = deep
    assert jnp.all(Y[..., DEAD] == 0.0)     # the forced-dead output
    assert jnp.all(jnp.isfinite(Y_g))
    assert jnp.any(Y_g != 0.0)              # ghostgrad fired somewhere
    # Y_g sums two terms: the decoder's own ghostgrad, masked to dead output
    # rows, and decoder.fwd(R_g), the ghost residual accumulated through the
    # stack, which is unmasked and signed. Signal on live rows can only come
    # from the second, so this checks the classifier-level ghost is live too
    # -- which needs the relu gate; with gate="none" it is structurally zero.
    live = jnp.delete(Y_g, DEAD, axis=-1)
    assert jnp.any(live != 0.0)


def test_last_prefix_matches_undeepsup(deep, flat):
    # prefix l-1 spans the whole stack, so it must reproduce the plain decode
    # and its ghost exactly (deepsup_sg=False, so `prefix` is the identity)
    Yd, Yd_g = deep
    Yf, Yf_g = flat
    assert jnp.allclose(Yd[-1], Yf, atol=1e-6)
    assert jnp.allclose(Yd_g[-1], Yf_g, atol=1e-6)


@pytest.mark.parametrize("i", range(KW["l"] - 1))
def test_ghost_is_computed_per_prefix(deep, i):
    # each prefix decodes its own ghost residual, so no earlier prefix may
    # equal the last one. This is the exact statement the old code violated:
    # it decoded the final ghost residual once and let it broadcast.
    _, Y_g = deep
    assert Y_g.shape[0] == KW["l"]  # else Y_g[i] indexes the batch, vacuously
    assert not jnp.allclose(Y_g[i], Y_g[-1], atol=1e-4)


def test_loss_pairs_prefix_with_prefix(deep, X):
    # end-to-end: L2_g must be the elementwise per-prefix ghost error, not a
    # broadcast of the last prefix against all of them
    Y, Y_g = deep
    assert Y_g.shape == Y.shape  # else the comparisons below only broadcast
    hyper = Hyperparams(d_in=KW["d_in"], d_out=KW["d_out"], b=B, s_g=1e-4,
                        ghost=True)
    stats = jnp.zeros((KW["l"], LAYER_WIDTH))

    _, row = hyper.loss(Y, X, Y_g, stats)
    L2_g = row[2]

    paired = jnp.mean((Y_g - (X - Y)) ** 2)
    assert jnp.isclose(L2_g, paired, rtol=1e-6)

    # what the pre-fix code computed, for contrast
    broadcast = jnp.mean((Y_g[-1] - (X - Y)) ** 2)
    assert not jnp.isclose(L2_g, broadcast, rtol=1e-3)

# Gated SAE (Rajamanoharan et al. 2024): one tied encoder direction read
# twice, as a gate deciding WHICH latents fire and a magnitude deciding how
# much. The architecture's whole point is that the L1 sits on the gate, so
# it controls sparsity without shrinking the magnitudes it selects.
#
# The gate reaches the code through a step function and therefore takes no
# gradient from reconstruction. That makes the auxiliary term load-bearing
# rather than a regularizer: without it nothing opposes the L1, every gate
# shuts, and the model still trains and still reports a loss. Most of these
# tests exist to make that failure mode impossible to introduce silently.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sae

D, M, B = 16, 32, 8


@pytest.fixture(scope="module")
def gated():
    p = sae.init_params(jax.random.PRNGKey(0), D, M, np.zeros(D), enc="gated")
    X = jax.random.normal(jax.random.PRNGKey(1), (B, D))
    return p, X


def test_param_set_has_no_unused_encoder(gated):
    """`b_enc` belongs to the single-path forms; leaving it in a gated
    checkpoint would let a reader think the gate uses it."""
    p, _ = gated
    assert set(p) == {"W_dec", "W_gate", "b_dec", "b_gate", "b_mag", "r_mag"}


def test_paths_are_tied_and_differ_only_by_scale_and_bias(gated):
    """One direction, two read-outs: `exp(r_mag)` is the only thing that
    may separate the magnitude path from the gate."""
    p, X = gated
    gate, mag = sae.gated_pre(p, X)
    assert np.allclose(gate - p["b_gate"], mag - p["b_mag"], atol=1e-6)

    p2 = {**p, "r_mag": p["r_mag"] + np.log(3.0)}
    gate2, mag2 = sae.gated_pre(p2, X)
    assert np.allclose(gate2, gate, atol=1e-6)          # gate untouched
    assert np.allclose(mag2 - p["b_mag"], 3.0 * (mag - p["b_mag"]), atol=1e-5)


def test_code_is_the_gated_relu(gated):
    p, X = gated
    gate, mag = sae.gated_pre(p, X)
    z = np.asarray(sae.encode(p, X, 0))
    want = np.where(np.asarray(gate) > 0, np.maximum(np.asarray(mag), 0.0), 0.0)
    assert np.array_equal(z, want)
    assert (z >= 0).all()


def test_encode_dispatches_on_params_not_a_flag(gated):
    """Seven downstream scripts call `sae.encode(params, X, topk, ...)`
    with a topk read from meta.json. A gated run has no topk, so the
    dispatch has to come off the params, or every consumer silently
    applies a top-k mask to a gated code."""
    p, X = gated
    for topk in (0, 4, 1000):
        assert np.array_equal(np.asarray(sae.encode(p, X, topk)),
                              np.asarray(sae.encode(p, X, 0)))


def test_reconstruction_alone_leaves_the_gate_without_gradient(gated):
    """The property that makes the auxiliary term necessary. If this ever
    becomes nonzero the gate has stopped being a step function and the
    architecture is no longer Gated."""
    p, X = gated

    def recon_only(pr):
        return jnp.sum((sae.decode(pr, sae.encode(pr, X, 0)) - X) ** 2)

    g = jax.grad(recon_only)(p)
    assert float(jnp.abs(g["b_gate"]).max()) == 0.0
    assert float(jnp.abs(g["W_gate"]).max()) > 0.0   # still reached via w_mag


def test_full_objective_does_reach_the_gate(gated):
    p, X = gated
    g = jax.grad(lambda pr: sae.gated_loss(pr, X, jnp.ones(D), 1e-3)[0])(p)
    assert float(jnp.abs(g["b_gate"]).max()) > 0.0


def test_auxiliary_term_does_not_train_the_dictionary(gated):
    """It trains the gate against a FROZEN decoder. A live decoder would
    let the auxiliary reconstruction pull the dictionary toward the dense
    ReLU(gate) code it uses, which is not the code the model emits.

    `b_dec` is the exception, and legitimately so: it is shared with the
    encoder's centering, so the term reaches it through `X - b_dec` even
    with the decoder copy frozen. Freezing the centering too is what
    drives its gradient to zero, which is how this pins the route rather
    than just the magnitude."""
    p, X = gated

    def aux_only(pr, freeze_centering=False):
        q = {**pr, "b_dec": jax.lax.stop_gradient(pr["b_dec"])} \
            if freeze_centering else pr
        rec = (jax.nn.relu(sae.gated_pre(q, X)[0])
               @ jax.lax.stop_gradient(pr["W_dec"])
               + jax.lax.stop_gradient(pr["b_dec"]))
        return ((rec - X) ** 2).mean()

    g = jax.grad(aux_only)(p)
    assert float(jnp.abs(g["W_dec"]).max()) == 0.0
    assert float(jnp.abs(g["b_dec"]).max()) > 0.0          # via the centering
    g2 = jax.grad(aux_only)(p, True)
    assert float(jnp.abs(g2["b_dec"]).max()) == 0.0        # and only that


def test_l1_presses_the_gate_shut_and_the_auxiliary_term_opposes_it(gated):
    """Sign check on the two forces the architecture balances. Raising the
    L1 must push `b_gate` down; the auxiliary term alone must push it up,
    since a shut gate reconstructs nothing."""
    p, X = gated
    w = jnp.ones(D)
    g_small = jax.grad(lambda pr: sae.gated_loss(pr, X, w, 1e-6)[0])(p)
    g_large = jax.grad(lambda pr: sae.gated_loss(pr, X, w, 1e-1)[0])(p)
    # a larger L1 adds positive gradient on b_gate (descent lowers it)
    assert (np.asarray(g_large["b_gate"]) >= np.asarray(g_small["b_gate"])
            - 1e-9).all()
    assert float(np.asarray(g_large["b_gate"] - g_small["b_gate"]).max()) > 0.0


def test_the_l1_never_reaches_the_magnitude_scale(gated):
    """The anti-shrinkage guarantee, stated exactly.

    `W_gate` is shared between the two paths, so the L1 does press on it.
    What keeps that from shrinking coefficients is `r_mag`/`b_mag`: a free
    per-latent scale and offset on the magnitude path that NO sparsity
    term touches, trained by reconstruction alone. Direction is shared and
    pressured; scale is free. Drop `r_mag` and the L1 shrinks the shared
    direction with nothing able to compensate, which is the ReLU+L1
    pathology wearing the Gated name."""
    p, X = gated
    w = jnp.ones(D)

    def l1_term(pr):
        return jax.nn.relu(sae.gated_pre(pr, X)[0]).sum(-1).mean()

    g = jax.grad(l1_term)(p)
    assert float(jnp.abs(g["r_mag"]).max()) == 0.0
    assert float(jnp.abs(g["b_mag"]).max()) == 0.0
    assert float(jnp.abs(g["W_dec"]).max()) == 0.0
    assert float(jnp.abs(g["b_gate"]).max()) > 0.0     # it does reach the gate

    def recon_term(pr):
        gt, mg = sae.gated_pre(pr, X)
        z = jnp.where(gt > 0, jax.nn.relu(mg), 0.0)
        return (((sae.decode(pr, z) - X) * w) ** 2).mean()

    gr = jax.grad(recon_term)(p)
    assert float(jnp.abs(gr["r_mag"]).max()) > 0.0     # and only recon does


def test_preacts_is_the_magnitude_path(gated):
    """`preacts` is what dead-latent bookkeeping reads, and for a gated
    model the coefficient lives on the magnitude path."""
    p, X = gated
    assert np.allclose(np.asarray(sae.preacts(p, X)),
                       np.asarray(sae.gated_pre(p, X)[1]), atol=1e-6)


def test_a_shut_gate_is_reported_as_dead_not_as_firing(gated):
    """`fired` drives dead-latent accounting; a latent whose gate never
    opens contributes nothing and must count as dead however large its
    magnitude logit is."""
    p, X = gated
    p_shut = {**p, "b_gate": p["b_gate"] - 1e6}
    _, (_, fired) = sae.gated_loss(p_shut, X, jnp.ones(D), 1e-3)
    assert not bool(np.asarray(fired).any())
    assert float(np.abs(np.asarray(sae.encode(p_shut, X, 0))).max()) == 0.0


def test_resampling_reopens_a_shut_gate(gated):
    """The only mechanism that can revive a gated latent. A shut gate gets
    nothing from reconstruction (see above), and `aux_k` acts on the
    magnitude path, so it cannot reopen one. Resampling clears `b_gate`,
    which can -- and that is why `--resample-every` had to stop requiring
    `--enc linear`."""
    import numpy as _np
    import optax
    p, X = gated
    dead = _np.zeros(M, bool)
    dead[:5] = True
    p_shut = {**p, "b_gate": p["b_gate"].at[:5].add(-1e6)}
    assert not _np.asarray(sae.encode(p_shut, X, 0))[:, :5].any()

    tx = optax.adam(1e-3)
    st = tx.init(p_shut)
    out, st2 = sae.resample_dead(p_shut, st, _np.asarray(X), jnp.ones(D),
                                 dead, _np.random.default_rng(0))
    assert float(jnp.abs(out["b_gate"][:5]).max()) == 0.0
    assert float(jnp.abs(out["r_mag"][:5]).max()) == 0.0
    assert float(jnp.abs(out["b_mag"][:5]).max()) == 0.0
    # untouched latents keep their parameters exactly
    assert _np.array_equal(_np.asarray(out["b_gate"][5:]),
                           _np.asarray(p_shut["b_gate"][5:]))
    # and the revived latents now fire on something
    assert _np.asarray(sae.encode(out, X, 0))[:, :5].any()


def test_resampling_zeroes_adam_moments_for_every_gated_key(gated):
    """`keep` has to name every params key; a missed one raises inside the
    moment reset, and a silently unmasked one leaves stale momentum that
    drags the fresh direction straight back."""
    import numpy as _np
    import optax
    p, X = gated
    dead = _np.zeros(M, bool)
    dead[:3] = True
    tx = optax.adam(1e-3)
    st = tx.init(p)
    # give every moment a nonzero history so a missed mask would show
    st = (st[0]._replace(mu=jax.tree.map(lambda v: jnp.ones_like(v), st[0].mu),
                         nu=jax.tree.map(lambda v: jnp.ones_like(v), st[0].nu)),
          ) + tuple(st[1:])
    _, st2 = sae.resample_dead(p, st, _np.asarray(X), jnp.ones(D), dead,
                               _np.random.default_rng(0))
    assert set(st2[0].mu) == set(p)
    for k in ("b_gate", "b_mag", "r_mag"):
        assert float(jnp.abs(st2[0].mu[k][:3]).max()) == 0.0
        assert float(jnp.abs(st2[0].mu[k][3:]).min()) == 1.0
    assert float(jnp.abs(st2[0].mu["W_gate"][:, :3]).max()) == 0.0
    assert float(jnp.abs(st2[0].mu["W_dec"][:3]).max()) == 0.0

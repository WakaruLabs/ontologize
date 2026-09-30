# Tag-axis dead-entry revival (p_revive / DictBlock.revive_dead), the
# counterpart of winner dropout. Under a hard `select` rule death is
# ABSORBING: an entry outside every sample's support has probability
# exactly 0, so it gets exactly zero gradient, so its logits cannot move,
# so it can never return. The dense rules keep every entry recoverable,
# which is why revival is a no-op for them.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.layers.dictblock import DictBlock
from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import update

from conftest import KW, B

H, K_TAGS, D, NB = 2, 8, 4, 16
LIVE = 2   # select="top2": only tags 0 and 1 ever make the support below


def logits():
    """Logits whose top-2 is tags {0,1} for every sample, leaving 2..7 dead."""
    return (jnp.zeros((NB, H, K_TAGS))
            .at[:, :, 0].set(2.0)
            .at[:, :, 1].set(1.0))


def block(select="top2"):
    db = DictBlock(k=K_TAGS, d=D, h=H, select=select,
                   dtype_str="float32", dtype_p_str="float32")
    K = logits()
    return db, db.init(jax.random.PRNGKey(0), K), K


def test_noop_without_rng_or_probability():
    db, params, K = block()
    for kw in ({"p_revive": 0.5, "rng": None}, {"p_revive": 0.0,
                                                "rng": jax.random.PRNGKey(1)}):
        out, _ = db.apply(params, K, method=DictBlock.revive_dead, **kw)
        assert jnp.array_equal(out, K)


def test_noop_for_dense_select():
    # nothing is ever masked out under softmax, so there is no absorbing
    # state to rescue and k_sel == k disables the mechanism
    db, params, K = block(select="softmax")
    out, _ = db.apply(params, K, p_revive=1.0, rng=jax.random.PRNGKey(1),
                      method=DictBlock.revive_dead)
    assert jnp.array_equal(out, K)


def test_promotes_exactly_one_dead_tag_to_the_leader():
    db, params, K = block()
    out, _ = db.apply(params, K, p_revive=1.0, rng=jax.random.PRNGKey(1),
                      method=DictBlock.revive_dead)
    changed = np.asarray(out != K)
    assert changed[..., :LIVE].sum() == 0        # incumbents untouched
    assert np.all(changed.sum(-1) == 1)          # exactly one per (sample, head)
    # promoted to a tie with the leader, not past it: a low logit would give
    # a vanishing softmax share, a dominating one would saturate it
    assert float(np.asarray(out)[changed].min()) == pytest.approx(2.0)
    assert float(out.max()) == pytest.approx(2.0)


def test_revived_tag_enters_the_support_and_takes_mass():
    db, params, K = block()
    out, _ = db.apply(params, K, p_revive=1.0, rng=jax.random.PRNGKey(1),
                      method=DictBlock.revive_dead)
    P = db.apply(params, out, 0.03, method=DictBlock.cluster)
    live = np.asarray(P > 0)
    assert np.all(live.sum(-1) == LIVE)          # still exactly top-2 wide
    # the revived tag ties the leader, so it takes half the head's mass
    revived = np.asarray(out != K)
    assert float(np.asarray(P)[revived].min()) == pytest.approx(0.5, abs=1e-3)


def test_fires_on_starvation_at_a_realistic_batch_size():
    # THE regression. A test for absence from the batch never fires at
    # b=256: a head makes b*k_sel selections over k entries, so a badly
    # starved entry is still picked occasionally. Triggering on rate is
    # what makes the mechanism reach the state it is meant to rescue.
    b, h, k, ksel = 256, 4, 32, 4
    db = DictBlock(k=k, d=4, h=h, select=f"top{ksel}",
                   dtype_str="float32", dtype_p_str="float32")
    key = jax.random.PRNGKey(11)
    # first 8 entries strongly preferred; the tail still sneaks in sometimes
    bias = jnp.concatenate([jnp.full((8,), 1.0), jnp.zeros((k - 8,))])
    K = jax.random.normal(key, (b, h, k)) * 0.5 + bias
    params = db.init(key, K)

    sel = K >= jax.lax.top_k(K, ksel)[0][..., -1:]
    assert int((~sel.any(0)).sum()) == 0, \
        "setup: nothing is absent from the batch, so absence cannot be the test"
    assert float(sel.mean(0)[:, 8:].mean()) < 0.5 * ksel / k, \
        "setup: the tail must actually be starved"

    out, _ = db.apply(params, K, p_revive=1.0, rng=jax.random.PRNGKey(12),
                      method=DictBlock.revive_dead)
    changed = np.asarray(out != K)
    assert changed.sum() > 0, "revival must fire on starvation, not only absence"
    # and it targets the starved tail rather than the preferred entries
    assert changed[:, :, 8:].sum() > 10 * changed[:, :, :8].sum()


def test_revive_frac_zero_makes_nothing_eligible():
    db, params, K = block()
    out, _ = db.apply(params, K, p_revive=1.0, revive_frac=0.0,
                      rng=jax.random.PRNGKey(1), method=DictBlock.revive_dead)
    assert jnp.array_equal(out, K)


def tag_grad_mass(p_revive):
    """Per-tag gradient reaching the dictionary, shape (h, k)."""
    db, params, K = block()

    def f(p):
        F, _, _, _ = db.apply(p, K, rng=jax.random.PRNGKey(1),
                              p_revive=p_revive, method=DictBlock.withStats)
        return F.sum()

    g = jax.grad(f)(params)["params"]["weights"]
    return np.asarray(jnp.abs(g).sum(-1))


def test_revival_is_what_gives_dead_tags_gradient():
    # the whole point: without revival the dead tags are frozen forever
    off, on = tag_grad_mass(0.0), tag_grad_mass(1.0)
    assert np.all(off[:, LIVE:] == 0.0), "dead tags should be frozen without revival"
    assert np.all(off[:, :LIVE] > 0.0)
    assert (on[:, LIVE:] > 0.0).any(), "revival must reach some dead tag"
    # the leader keeps its mass (the revived tag only ties it); at
    # p_revive=1 every head revives, so the promoted tag displaces the
    # weakest incumbent from the support -- the same trade drop_winners
    # makes, and at a sane p_revive it touches only that fraction of pairs
    assert np.all(on[:, 0] > 0.0), "the leader must keep its gradient"


def test_displacement_is_proportional_to_p_revive():
    # at p_revive=1 the weakest incumbent is displaced everywhere; well
    # below 1 it keeps most of its gradient
    weak_off = tag_grad_mass(0.0)[:, LIVE - 1]
    assert np.all(tag_grad_mass(1.0)[:, LIVE - 1] == 0.0)
    assert np.all(tag_grad_mass(0.05)[:, LIVE - 1] > 0.5 * weak_off)


def test_training_steps_finite_with_revival(X):
    d = KW["d_in"]
    hyper = Hyperparams(
        d, d, B, 1, 5e-5, 0.0, 1.0,
        "batchnorm", "normal", "featvar", 0.0, 0.02, 0.1,
        s_Hm=1e-6, p_drop=0.1, p_revive=0.05, ghost=False)
    model = hyper.ontologizer(
        d, d, KW["e_dec"], KW["k"], KW["h"], KW["l"],
        n=2, gate="none", forward="resid", deepsup=True, select="top2",
        dtype_str="float32", dtype_p_str="float32")
    state = hyper.init(model, save_each=10)
    rng = jax.random.PRNGKey(2)
    for step in range(10):
        rng, r = jax.random.split(rng)
        state, L, _ = update(state, hyper.loss, r, X, X, temperature=0.5,
                             p_drop=0.1, p_revive=0.05, sd_K=0.02,
                             sd_in=0.0, sd_F=0.1, grad_clip=1.0)
        assert jnp.isfinite(L), f"step {step}"
    assert jnp.all(jnp.isfinite(state.stats[:10]))


def test_live_under_ste_because_ste_is_not_dense_for_the_dictionary():
    """`ste` keeps one entry per head, so `k_sel` is 1 and revival applies.

    It was `k` before, which tripped the `k_sel >= k` guard and switched the
    mechanism off for the rule the live arm uses. The reasoning that hid it
    -- that hard rules make starvation absorbing, dense ones do not -- is
    right; the classification of `ste` was wrong. A straight-through
    estimator restores gradient to what PRODUCED the code, not to what
    consumes it: `fwd` is `P @ dicts()` and P's forward value is one-hot, so
    an entry that never wins gets exactly zero gradient on its dictionary
    rows, the same as under `top<k>`."""
    import numpy as np
    for select, want in (("ste", 1), ("argmax", 1), ("top4", 4),
                         ("softmax", None)):
        db, params, K = block(select=select)
        want = db.k if want is None else want
        assert db.apply(params, K, method=lambda m, _: m.k_sel) == want, select

    # and the dictionary gradient is what justifies it
    h, k, d, b = 4, 8, 16, 32
    K = jax.random.normal(jax.random.PRNGKey(0), (b, h, k)) * 0.5
    K = K.at[:, :, 0].set(-20.0)          # entry 0 never wins
    grads = {}
    for select in ("ste", "top4", "softmax"):
        db = DictBlock(k=k, d=d, h=h, select=select, sparse=True,
                       dtype_str="float32", dtype_p_str="float32")
        p = db.init(jax.random.PRNGKey(1), K)

        def loss(pr):
            P = db.apply(pr, K, 1.0, method=DictBlock.cluster)
            return jnp.sum(db.apply(pr, P, method=DictBlock.fwd) ** 2)

        g = np.asarray(jax.grad(loss)(p)["params"]["weights"])
        grads[select] = float(np.abs(g[:, 0, :]).max())
    assert grads["ste"] == 0.0            # absorbing, like top-k
    assert grads["top4"] == 0.0
    assert grads["softmax"] > 0.0         # recoverable, unlike the above

# experiments/boundary-count/nullstep.py: the shortest step that flips one
# head with other heads' margins held. The tests pin the linear algebra
# (the step meets its constraints, is the shortest that does, and does not
# exist once the constraints span the target) and the one model-side piece
# the script rebuilds itself: the hard forward with the codes frozen and
# the residual and gains left to follow the input.
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "boundary-count"))
from nullstep import (check_model, clean_logits, collateral,  # noqa: E402
                      constrained_step, crossings, forced_codes, frozen_logits,
                      margin_jacobian, margins, steps_for, weighted_step)

HARD = dict(select="ste", resid_gain=True, resid_const=True)


def test_margins_are_winner_minus_each_entry():
    K = np.array([[0.1, 0.5, 0.3], [2.0, -1.0, 1.5]])
    win, ru, m = margins(K)
    assert win.tolist() == [1, 0] and ru.tolist() == [2, 2]
    assert np.allclose(m, [[0.4, 0.0, 0.2], [0.0, 3.0, 0.5]])
    dK = np.random.default_rng(0).normal(size=(2, 3, 5))
    J = margin_jacobian(dK, win)
    assert np.allclose(J[0, 2], dK[0, 1] - dK[0, 2])
    assert np.allclose(J[1, 0], 0.0)


def test_constrained_step_meets_and_minimizes():
    rng = np.random.default_rng(1)
    d, c = 12, 5
    g, C = rng.normal(size=d), rng.normal(size=(c, d))
    step, share = constrained_step(g, -0.7, C)
    assert np.isclose(g @ step, -0.7)
    assert np.allclose(C @ step, 0.0, atol=1e-10)
    assert 0.0 < share < 1.0
    # any other feasible step is this one plus a vector the constraints
    # and g both annihilate, which can only lengthen it
    N = np.linalg.svd(np.vstack([C, g]))[2][c + 1:]
    for z in rng.normal(size=(8, len(N))):
        other = step + z @ N
        assert np.isclose(g @ other, -0.7)
        assert np.linalg.norm(other) >= np.linalg.norm(step) - 1e-12


def test_unconstrained_step_is_along_the_gradient():
    g = np.array([3.0, 4.0])
    step, share = constrained_step(g, 5.0, np.zeros((0, 2)))
    assert np.allclose(step, [0.6, 0.8]) and share == 1.0


def test_no_step_when_the_constraints_span_the_target():
    rng = np.random.default_rng(2)
    C = rng.normal(size=(4, 6))
    g = rng.normal(size=4) @ C              # inside the constraints' span
    step, share = constrained_step(g, 1.0, C)
    assert step is None and share < 1e-9
    # and, generically, whenever they are as many as the dimensions
    step, share = constrained_step(rng.normal(size=6),
                                   1.0, rng.normal(size=(6, 6)))
    assert step is None


def test_weighted_step_trades_constraints_by_weight():
    rng = np.random.default_rng(3)
    d = 6
    g, C = rng.normal(size=d), rng.normal(size=(9, d))   # 9 > d: no null
    w = np.ones(9)
    step = weighted_step(g, 1.0, C, w, 1e-9)
    assert np.isclose(g @ step, 1.0)
    cost = lambda s, w: float((w * (C @ s) ** 2).sum())
    for z in rng.normal(size=(8, d)):
        z -= g * (g @ z) / (g @ g)           # stays on g . step = 1
        assert cost(step + 0.1 * z, w) >= cost(step, w) - 1e-12
    # weighing one constraint up moves the step toward holding it
    w2 = w.copy()
    w2[0] = 1e6
    step2 = weighted_step(g, 1.0, C, w2, 1e-9)
    assert abs(C[0] @ step2) < 0.01 * abs(C[0] @ step)
    # with fewer constraints than dimensions it reaches the exact null
    exact, _ = constrained_step(g, 1.0, C[:3])
    soft = weighted_step(g, 1.0, C[:3], np.ones(3), 1e-12)
    assert np.allclose(C[:3] @ soft, 0.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(soft), np.linalg.norm(exact), rtol=1e-4)


def test_crossings_and_collateral():
    m = np.array([[0.0, 0.2], [0.0, 1.0], [0.3, 0.0]])
    J = np.zeros((3, 2, 2))
    J[0, 1] = [-1.0, 0.0]
    J[1, 1] = [-1.0, 0.0]
    J[2, 0] = [0.0, -1.0]
    assert crossings(m, J, np.array([0.5, 0.0])).tolist() == [True, False,
                                                              False]
    assert crossings(m, J, np.array([0.5, 0.4])).tolist() == [True, False,
                                                              True]
    A0 = np.zeros((2, 4), int)
    A = A0.copy()
    A[0, 1] = A[0, 2] = A[1, 3] = 1         # the target, one neighbour, one below
    layer, stack = collateral(A, A0, 0, 1)
    assert np.isclose(layer, 1 / 3) and np.isclose(stack, 2 / 7)


def test_steps_for_holds_what_each_name_says():
    rng = np.random.default_rng(4)
    l, h, k, d = 3, 4, 3, 40
    K = rng.normal(size=(l, h, k))
    dK = rng.normal(size=(l, h, k, d))
    win, ru = np.empty((l, h), int), np.empty((l, h), int)
    m, J = np.empty((l, h, k)), np.empty((l, h, k, d))
    for i in range(l):
        win[i], ru[i], m[i] = margins(K[i])
        J[i] = margin_jacobian(dK[i], win[i])
    li, hi = 1, 2
    steps, share = steps_for(li, hi, m, J, win, ru, 1.5, 1e-9)
    target = -1.5 * m[li, hi, ru[li, hi]]
    for name, st in steps.items():
        assert np.isclose(J[li, hi, ru[li, hi]] @ st, target), name
    moved = lambda st: np.abs(J @ st)                     # (l, h, k)
    other = np.ones(h, bool)
    other[hi] = False
    assert moved(steps["null_layer"])[li, other].max() < 1e-9
    ru_moved = lambda st: np.take_along_axis(
        moved(st), ru[..., None], -1)[..., 0]             # (l, h)
    assert ru_moved(steps["null_ru"])[li, other].max() < 1e-9
    assert moved(steps["null_ru"])[li, other].max() > 1e-6
    up = ru_moved(steps["null_upto"])
    assert up[0].max() < 1e-9 and up[li, other].max() < 1e-9
    assert up[2].max() > 1e-6
    full = ru_moved(steps["null_stack"])
    full[li, hi] = 0.0
    assert full.max() < 1e-9
    # each added constraint lengthens the step
    n = {a: np.linalg.norm(s) for a, s in steps.items()}
    assert n["grad"] <= n["null_ru"] <= n["null_upto"] <= n["null_stack"]
    assert set(share) == {"null_layer", "null_ru", "null_upto", "null_stack"}


def test_steps_for_skips_sets_larger_than_the_space():
    rng = np.random.default_rng(5)
    l, h, k, d = 2, 4, 3, 5                 # one layer: 3 * 2 = 6 margins > 5
    K = rng.normal(size=(l, h, k))
    dK = rng.normal(size=(l, h, k, d))
    win, ru = np.empty((l, h), int), np.empty((l, h), int)
    m, J = np.empty((l, h, k)), np.empty((l, h, k, d))
    for i in range(l):
        win[i], ru[i], m[i] = margins(K[i])
        J[i] = margin_jacobian(dK[i], win[i])
    steps, share = steps_for(0, 0, m, J, win, ru, 1.5, 1e-6)
    assert "null_layer" not in steps and share["null_layer"] == 0.0
    assert "null_ru" in steps and "soft_layer" in steps


def test_frozen_forward_matches_the_module(X, build):
    model, params = build(**HARD)
    check_model(model)
    K = model.apply(params, X, 0.5, method=clean_logits)        # (l, b, h, k)
    P = jax.nn.one_hot(K.argmax(-1), model.k)
    Kf = jnp.stack([model.apply(params, X[i], P[:, i], method=frozen_logits)
                    for i in range(4)], 1)
    assert jnp.allclose(Kf, K[:, :4], atol=1e-5)


def test_frozen_jacobian_follows_the_residual_and_the_gain(X, build):
    """The module stops the gradient at each layer's residual input and at
    its gain. Between flips a step moves both, so the frozen forward's
    Jacobian must match finite differences above layer 0 too."""
    model, params = build(**HARD)
    x = X[0]
    K = model.apply(params, x[None], 0.5, method=clean_logits)[:, 0]
    P = jax.nn.one_hot(K.argmax(-1), model.k)
    f = lambda v: model.apply(params, v, P, method=frozen_logits)
    J = jax.jacfwd(f)(x)                                        # (l, h, k, d)
    v = jax.random.normal(jax.random.PRNGKey(7), x.shape)
    v = v / jnp.linalg.norm(v)
    eps = 1e-3
    fd = (f(x + eps * v) - f(x - eps * v)) / (2 * eps)
    lin = J @ v
    assert float(jnp.abs(lin[1:]).max()) > 1e-4     # the layers above move
    assert jnp.allclose(fd, lin, atol=2e-2 * float(jnp.abs(lin).max()))


def test_forced_codes_change_nothing_at_or_below_the_target(X, build):
    model, params = build(**HARD)
    b = X.shape[0]
    A0 = model.apply(params, X, 0.5, method=clean_logits).argmax(-1)
    li = jnp.full((b,), 1)
    hi = jnp.full((b,), 2)
    same = model.apply(params, X, 0.5, li, hi, A0[1, :, 2],
                       method=forced_codes)
    assert (same == A0).all()               # forcing the winner is a no-op
    ki = (A0[1, :, 2] + 1) % model.k
    A = model.apply(params, X, 0.5, li, hi, ki, method=forced_codes)
    assert (A[0] == A0[0]).all()            # nothing below the target
    assert (A[1, :, 2] == ki).all()
    other = jnp.arange(model.h) != 2
    assert (A[1][:, other] == A0[1][:, other]).all()
    assert (A[2] != A0[2]).any()            # the layer above reclassifies


def test_check_model_refuses_other_forwards(build):
    for over in (dict(select="softmax", resid_gain=True, resid_const=True),
                 dict(select="ste", resid_gain=False, resid_const=True),
                 dict(select="ste", resid_gain=True, resid_const=True,
                      scaled=True)):
        model, _ = build(**over)
        with pytest.raises(ValueError):
            check_model(model)

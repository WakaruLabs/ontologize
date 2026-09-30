"""Setpoint control: the dual step, the ramp, the projection, and the
traced overrides that carry a controlled coefficient into the loss.

The point of a loop is that its measurement has an operating point chosen
by the target rather than by the step budget, so the tests that matter are
the fixed point (no motion when on target), the sign of the correction,
and convergence of the closed loop against a monotone plant.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import (DualLoop, dual_apply,
                                            dual_control, dual_target,
                                            update)

B, T = 8, 0.5
KW = dict(d_in=6, e_dec=10, k=4, h=2, l=2)
X = jax.random.normal(jax.random.PRNGKey(0), (B, KW["d_in"]))

LO, HI = 1e-12, 1e-3


def test_on_target_is_a_fixed_point():
    s = 3e-6
    assert dual_control(s, 480.0, 480.0, 1e-3, LO, HI) == pytest.approx(s)


def test_correction_signs():
    s = 3e-6
    # above target -> the multiplier rises; below -> it falls
    assert dual_control(s, 960.0, 480.0, 1e-3, LO, HI) > s
    assert dual_control(s, 240.0, 480.0, 1e-3, LO, HI) < s


def test_disabled_when_target_is_zero():
    for L1_F in (0.0, 1.0, 1e9):
        assert dual_control(3e-6, L1_F, 0.0, 1e-3, LO, HI) == 3e-6


def test_clipped_to_bounds():
    # a large sustained error cannot push the multiplier outside [lo, hi]
    s = 1e-6
    for _ in range(10000):
        s = dual_control(s, 1e9, 1.0, 1e-1, LO, HI)
    assert s == HI
    for _ in range(10000):
        s = dual_control(s, 1e-9, 1e9, 1e-1, LO, HI)
    assert s == LO


def test_zero_measurement_does_not_blow_up():
    s = dual_control(3e-6, 0.0, 480.0, 1e-3, LO, HI)
    assert LO <= s <= HI and np.isfinite(s)


def test_closed_loop_converges():
    """Against a monotone decreasing plant `L1_F = c / s`, the loop should
    settle at the target rather than run to a bound."""
    c, target, s = 1e-3, 480.0, 1e-6
    for _ in range(20000):
        s = dual_control(s, c / s, target, 1e-2, LO, HI)
    assert c / s == pytest.approx(target, rel=1e-3)
    assert LO < s < HI


def test_traced_override_replaces_the_field():
    """`Hyperparams.loss` must use the passed `s_L1F`, not the field, and
    report the applied value in the last stats column."""
    d = KW["d_in"]
    hyper = Hyperparams(d, d, B, s_L1F=1e-9, ghost=False)
    model = hyper.ontologizer(
        d, d, KW["e_dec"], KW["k"], KW["h"], KW["l"], n=2,
        forward="resid", deepsup=True,
        dtype_str="float32", dtype_p_str="float32")
    # the override is large so the change it makes to the total is well
    # above float32 resolution at the loss magnitude -- differencing two
    # ~1.0 totals cannot resolve the ~5e-6 a realistic s_L1F would move
    rows = {}
    for s in (None, 0.0, 1e-4):
        state = hyper.init(model, save_each=4)
        state, L, _ = update(state, hyper.loss, jax.random.PRNGKey(1), X, X,
                             temperature=T, sd_in=0.0, sd_K=0.0, sd_F=0.0,
                             p_drop=0.0, s_L1F=s)
        rows[s] = np.asarray(state.stats[0])

    assert rows[None][13] == pytest.approx(1e-9)   # falls back to the field
    assert rows[0.0][13] == 0.0
    assert rows[1e-4][13] == pytest.approx(1e-4)
    # the override has to change the total, not just the logged value
    L1_F = rows[1e-4][4]
    assert L1_F > 0.0
    assert rows[1e-4][4] == pytest.approx(rows[0.0][4])  # same forward
    assert rows[1e-4][0] - rows[0.0][0] == pytest.approx(1e-4 * L1_F,
                                                        rel=1e-3)


def test_sparse_F_follows_the_field_not_the_override():
    """The gradient gate is a static architecture flag resolved at build
    time, so a run that intends to control `s_L1F` still has to configure a
    nonzero one -- a zero field with a controlled override would leave the
    penalty gradient-free, the bug `sparse_F` had before."""
    d = KW["d_in"]
    off = Hyperparams(d, d, B, s_L1F=0.0, ghost=False)
    on = Hyperparams(d, d, B, s_L1F=1e-9, ghost=False)
    mk = lambda h: h.ontologizer(d, d, KW["e_dec"], KW["k"], KW["h"],
                                 KW["l"], n=2, forward="resid",
                                 dtype_str="float32", dtype_p_str="float32")
    assert mk(off).sparse_F is False
    assert mk(on).sparse_F is True


def test_ramp_endpoints_and_monotonicity():
    start, target, ramp = 5950.0, 480.0, 8000
    assert dual_target(0, 0, ramp, start, target) == pytest.approx(start)
    assert dual_target(ramp, 0, ramp, start, target) == pytest.approx(target)
    # held after the ramp, and monotone decreasing along it
    assert dual_target(ramp * 3, 0, ramp, start, target) == pytest.approx(target)
    vals = [dual_target(s, 0, ramp, start, target)
            for s in range(0, ramp + 1, 200)]
    assert all(b <= a for a, b in zip(vals, vals[1:]))


def test_ramp_resumes_from_where_it_left_off():
    """`step_0` is where the start was measured, so a resumed run covers the
    remaining steps rather than restarting the schedule."""
    start, target, ramp = 5950.0, 480.0, 8000
    mid = dual_target(4000, 0, ramp, start, target)
    # resuming at 4000 having measured `mid` there lands on the same curve
    assert dual_target(6000, 4000, ramp, mid, target) == pytest.approx(
        dual_target(6000, 0, ramp, start, target), rel=1e-6)
    assert dual_target(ramp, 4000, ramp, mid, target) == pytest.approx(target)


def test_ramp_degenerates_safely():
    # ramp already passed, and a zero start (no measurement yet)
    assert dual_target(10, 100, 50, 5950.0, 480.0) == 480.0
    assert dual_target(5, 0, 10, 0.0, 480.0) == pytest.approx(240.0)


def test_ramp_keeps_the_error_small():
    """The point of the ramp: against a plant that tracks the setpoint with
    a lag, the controller sees a bounded error instead of a 12x one, so the
    multiplier stays off its bound."""
    start, target, ramp = 5950.0, 480.0, 8000
    s, L1_F = 3e-6, start
    peak = 0.0
    for step in range(ramp + 4000):
        tgt = dual_target(step, 0, ramp, start, target)
        s = dual_control(s, L1_F, tgt, 1e-2, LO, HI)
        # plant: L1_F relaxes toward the level this multiplier sustains
        L1_F += 0.002 * ((1e-3 / max(s, 1e-12)) ** 0.5 * 15.0 - L1_F)
        peak = max(peak, s)
    assert peak < HI          # never pinned at the bound
    assert L1_F == pytest.approx(target, rel=0.15)


def test_apply_is_zero_while_the_constraint_is_slack():
    assert dual_apply(6.6e-5, 300.0, 480.0) == 0.0
    assert dual_apply(6.6e-5, 480.0, 480.0) == 0.0   # slack at equality
    assert dual_apply(6.6e-5, 481.0, 480.0) == 6.6e-5
    # disabled controller passes the multiplier straight through
    assert dual_apply(3e-6, 1.0, 0.0) == 3e-6


def test_projection_stops_a_one_way_plant_at_the_target():
    """With a plant that only ever descends -- which is what `L1_F` does --
    a multiplier that merely decays keeps pushing past the target forever,
    while the projected one parks there."""
    target, decayed, projected = 480.0, 5950.0, 5950.0
    s_d = s_p = 3e-6
    for _ in range(40000):
        s_d = dual_control(s_d, decayed, target, 1e-2, LO, HI)
        s_p = dual_control(s_p, projected, target, 1e-2, LO, HI)
        # one-way: the penalty lowers L1_F, nothing raises it back
        decayed -= 2e4 * s_d
        projected -= 2e4 * dual_apply(s_p, projected, target)
    assert decayed < 0.9 * target          # overshoots and never returns
    # the projection stops within one step of the plant's motion and holds
    # there; the residual undershoot is the discretization, not drift
    assert projected == pytest.approx(target, rel=0.05)
    assert projected > decayed


def test_projection_resumes_after_the_target_moves_down():
    """The unprojected variable keeps evolving underneath, so a descending
    ramp re-engages the penalty instead of being stuck at zero."""
    s, L1_F = 3e-6, 500.0
    # slack at first: nothing applied, but the dual keeps moving
    s = dual_control(s, L1_F, 600.0, 1e-2, LO, HI)
    assert dual_apply(s, L1_F, 600.0) == 0.0
    assert s > 0.0
    # ramp brings the target below L1_F: the penalty comes back on
    assert dual_apply(s, L1_F, 400.0) == s


def test_loop_is_inert_when_target_is_zero():
    d = DualLoop(3e-6, 0.0, 1e-2, 0.99, LO, HI, 1000, 0)
    assert d.on is False
    for m in (0.0, 5.0, 1e9):
        assert d.step(500, m) == 3e-6


def test_loop_holds_a_two_way_plant_at_the_target():
    """`cossim_k` differs from `L1_F` in moving both ways -- a collapsing
    layer descends, then climbs. The projection makes the loop a
    thermostat: off while healthy (costing nothing), re-engaging on drift."""
    target, ramp = 0.2, 500
    d = DualLoop(1e-5, target, 5e-2, 0.5, LO, 1e-2, ramp, 0)
    m, off = 0.65, 0
    for step in range(6000):
        s = d.step(step, m)
        off += s == 0.0
        # two-way plant: penalty pulls down, an intrinsic drift pushes up
        m += -4.0 * s + 2e-4
    assert m == pytest.approx(target, rel=0.15)
    assert off > 0          # the penalty does switch off, unlike a fixed weight
    assert 0.0 < d.s < 1e-2  # and the dual variable stays off both bounds


def test_max_sees_a_collapse_the_sum_does_not():
    """Why the loop watches the per-layer max. As one layer collapses the
    others improve faster, so the sum falls throughout while the max
    rises -- a loop on the sum would report success the whole way."""
    early = [0.388, 0.408, 0.424, 0.449, 0.508]   # before the turn
    late = [0.247, 0.264, 0.252, 0.235, 0.799]    # after it
    assert sum(late) < sum(early)     # the sum improves
    assert max(late) > max(early)     # while the max degrades
    tgt = 0.6
    assert dual_apply(1e-4, sum(late) / len(late), tgt) == 0.0   # sum: slack
    assert dual_apply(1e-4, max(late), tgt) == 1e-4              # max: engaged

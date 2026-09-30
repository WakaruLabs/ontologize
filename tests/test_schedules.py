# Annealing schedules: temperature and sd_K geometric, p_drop and the
# pwak_s diffusion depth linear (the depth as an integer), all sharing the
# anneal_steps horizon and held at their end values afterwards.
# Indexed by absolute step, so checkpoint resume lands at the right point.
import math

from ontologize.training.ontostate import schedules

ARGS = dict(temperature=1.0, temperature_end=0.03,
            p_drop=0.1, p_drop_start=0.0,
            sd_K=0.02, sd_K_end=0.006,
            pwak_s=4, pwak_s_start=0)


def test_start_values():
    assert schedules(0, 100, **ARGS) == (1.0, 0.0, 0.02, 0)


def test_end_values_and_hold():
    end = (0.03, 0.1, 0.006, 4)
    assert schedules(100, 100, **ARGS) == end
    assert schedules(100000, 100, **ARGS) == end


def test_midpoint_interpolation():
    T, pd, sk, ps = schedules(50, 100, **ARGS)
    assert math.isclose(T, math.sqrt(1.0 * 0.03))    # geometric
    assert math.isclose(pd, 0.05)                    # linear
    assert math.isclose(sk, math.sqrt(0.02 * 0.006)) # geometric
    assert ps == 2                                   # linear, integer


def test_none_endpoints_disable_each_schedule():
    T, pd, sk, ps = schedules(50, 100, 1.0, None, 0.1, None, 0.02, None)
    assert (T, pd, sk, ps) == (1.0, 0.1, 0.02, 0)


def test_zero_anneal_steps_disables_all():
    assert schedules(50, 0, **ARGS) == (1.0, 0.1, 0.02, 4)


def test_geometric_falls_back_to_linear_at_zero():
    # a zero endpoint would make the geometric form degenerate
    T, _, _, _ = schedules(50, 100, 1.0, 0.0, 0.1, None, 0.02, None)
    assert math.isclose(T, 0.5)

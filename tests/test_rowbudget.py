# experiments/recipe-ladder/rowbudget.py: SAE codes held to a fixed
# per-row budget, read exactly. The tests pin that rate control's FVU
# cannot rise with the budget, which the least-budget search relies on;
# that the search returns the least budget meeting a target; that the
# longest-fit and one-step readings choose what they say, with no side
# bits charged to a step every row shares; and that rate control is never
# worse than each row's longest code that fits.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "recipe-ladder"))
from rowbudget import (cap_step, least_budget, longest_fit,  # noqa: E402
                       nearest_mean_step, picked, read, row_fvu)

import quantrate as q  # noqa: E402


def toy(seed: int = 0, s: int = 9, n: int = 40):
    """Finer steps give longer codes; error falls with them, but not
    monotonically within a row."""
    rng = np.random.default_rng(seed)
    L = np.sort(rng.uniform(50, 400, size=(s, n)), axis=0)
    E = (np.sort(rng.uniform(0.1, 2.0, size=(s, n)), axis=0)[::-1]
         + rng.uniform(0, 0.5, size=(s, n)))
    return L, E


def control_fvu(L, E, budget, side):
    return row_fvu(picked(E, q.rate_control(L, E, budget, side)), 1, 1.0)


def test_rate_control_fvu_does_not_rise_with_the_budget():
    for seed in range(5):
        L, E = toy(seed)
        side = 3.17
        cands = np.unique(L + side)
        budgets = np.concatenate([[cands[0] - 1], cands, [cands[-1] + 1]])
        fvus = np.array([control_fvu(L, E, b, side) for b in budgets])
        assert np.all(np.diff(fvus) <= 1e-12)


def test_least_budget_is_the_least_breakpoint_meeting_the_target():
    L, E = toy(1)
    side = 3.17
    cands = np.unique(L + side)
    fvus = np.array([control_fvu(L, E, b, side) for b in cands])
    for t in np.quantile(fvus, [0.0, 0.1, 0.5, 0.9, 1.0]):
        brute = cands[np.argmax(fvus <= t)]
        got = least_budget(L, E, t, side, 1, 1.0)
        assert got == brute
        # a hair below it the target is missed, unless it is the first
        if got > cands[0]:
            assert control_fvu(L, E, got - 1e-9, side) > t
    assert least_budget(L, E, fvus.min() - 1e-6, side, 1, 1.0) is None


def test_longest_fit_takes_the_longest_code_that_fits_else_the_shortest():
    L = np.array([[10.0, 10.0], [20.0, 30.0], [40.0, 50.0]])
    assert longest_fit(L, 35.0, 0.0).tolist() == [1, 1]
    assert longest_fit(L, 45.0, 0.0).tolist() == [2, 1]
    assert longest_fit(L, 5.0, 0.0).tolist() == [0, 0]     # nothing fits
    # side bits count against the budget
    assert longest_fit(L, 45.0, 6.0).tolist() == [1, 1]


def test_one_step_readings_choose_one_step_for_every_row():
    L = np.array([[10.0, 10.0], [20.0, 30.0], [40.0, 50.0]])
    E = np.array([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]])
    assert nearest_mean_step(L, 30.0) == 1          # means 10, 25, 45
    assert cap_step(L, E, 35.0) == 1                # step 2's longest is 50
    assert cap_step(L, E, 50.0) == 2
    assert cap_step(L, E, 30.0) == 1                # no side bits charged
    assert cap_step(L, E, 5.0) is None


def test_rate_control_is_never_worse_than_the_longest_fit():
    for seed in range(5):
        L, E = toy(seed)
        for b in np.quantile(L, [0.2, 0.5, 0.8]):
            rc = q.rate_control(L, E, b, 3.17)
            lf = longest_fit(L, b, 3.17)
            assert np.all(picked(E, rc) <= picked(E, lf))


def test_read_counts_the_rows_the_search_moves_and_improves():
    L = np.array([[10.0, 10.0], [20.0, 30.0], [40.0, 50.0]])
    E = np.array([[3.0, 3.0], [1.0, 2.0], [2.0, 1.0]])
    r = read(L, E, 45.0, 0.0, 1, 1.0, ["a", "b", "c"])
    rc = r["rate_control"]
    # row 0: longest fit is step 2 (error 2), the search takes step 1
    # (error 1); row 1: step 2 does not fit, and step 1 is both
    assert rc["off_longest_share"] == 0.5
    assert rc["improved_share"] == 0.5
    assert rc["fvu"] == 1.5 and r["longest_fit"]["fvu"] == 2.0
    assert rc["bits_mean"] == 25.0 and rc["over_share"] == 0.0
    assert r["one_step_mean"]["step"] == "c"        # mean 45
    assert r["one_step_mean"]["over_share"] == 0.5
    assert r["one_step_cap"]["step"] == "b"

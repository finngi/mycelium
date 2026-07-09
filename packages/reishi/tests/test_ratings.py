"""Bradley-Terry ratings computed over Comparison records: pure Board-layer
aggregation, no store IO (comparisons and trials are passed in directly, the
same style test_board_measurement_key.py uses for Trial).
"""

import math

from reishi.primitives.comparison import Comparison
from reishi.primitives.ratings import ratings
from reishi.primitives.trial import Trial

# Trial ids are prefixed t- so they never collide with the winner literals
# ("a" == trial_a won, "b" == trial_b won, "tie"), which name a slot, not a trial.
T_A, T_B, T_C, T_D = "t-a", "t-b", "t-c", "t-d"


def _cmp(trial_a: str, trial_b: str, winner: str, i: int) -> Comparison:
    return Comparison(
        id=f"cmp-{i}", trial_a=trial_a, trial_b=trial_b, winner=winner, judge="human"
    )


def _trial(trial_id: str, recipe: str) -> Trial:
    return Trial(id=trial_id, recipe=recipe, seed=0)


def test_empty_input_returns_empty_list():
    assert ratings([]) == []


def test_orders_by_strength_a_beats_b_beats_c():
    comparisons = [
        _cmp(T_A, T_B, "a", 1),  # T_A beats T_B
        _cmp(T_A, T_B, "a", 2),  # T_A beats T_B again
        _cmp(T_B, T_C, "a", 3),  # T_B beats T_C
    ]
    trials = [_trial(T_A, "r-a"), _trial(T_B, "r-b"), _trial(T_C, "r-c")]

    rows = ratings(comparisons, trials=trials)

    assert [r["trial"] for r in rows] == [T_A, T_B, T_C]
    strengths = {r["trial"]: r["strength"] for r in rows}
    assert strengths[T_A] > strengths[T_B] > strengths[T_C]


def test_recipe_joined_onto_rows_when_trial_loadable():
    comparisons = [_cmp(T_A, T_B, "a", 1)]
    trials = [_trial(T_A, "r-a"), _trial(T_B, "r-b")]

    rows = {r["trial"]: r for r in ratings(comparisons, trials=trials)}

    assert rows[T_A]["recipe"] == "r-a"
    assert rows[T_B]["recipe"] == "r-b"


def test_ties_are_counted_on_both_sides():
    comparisons = [_cmp(T_A, T_B, "tie", 1)]
    trials = [_trial(T_A, "r-a"), _trial(T_B, "r-b")]

    rows = {r["trial"]: r for r in ratings(comparisons, trials=trials)}

    assert rows[T_A]["ties"] == 1
    assert rows[T_B]["ties"] == 1
    assert rows[T_A]["wins"] == 0
    assert rows[T_A]["losses"] == 0
    assert rows[T_A]["n"] == 1


def test_undefeated_item_does_not_diverge():
    # T_A beats everyone and is never beaten: raw Bradley-Terry sends its
    # strength to infinity without the phantom-opponent regularization.
    comparisons = [
        _cmp(T_A, T_B, "a", 1),
        _cmp(T_A, T_C, "a", 2),
        _cmp(T_A, T_D, "a", 3),
    ]
    trials = [_trial(t, t) for t in (T_A, T_B, T_C, T_D)]

    rows = {r["trial"]: r for r in ratings(comparisons, trials=trials)}

    assert math.isfinite(rows[T_A]["strength"])
    assert rows[T_A]["strength"] > rows[T_B]["strength"]


def test_unknown_trial_reference_warns_once_and_still_rates(capsys):
    comparisons = [
        _cmp(T_A, T_B, "a", 1),
        _cmp(T_A, T_C, "a", 2),
    ]
    # T_B and T_C reference no loadable trial (e.g. archived) -- only T_A is known.
    trials = [_trial(T_A, "r-a")]

    rows = {r["trial"]: r for r in ratings(comparisons, trials=trials)}

    assert rows[T_B]["recipe"] is None
    assert rows[T_C]["recipe"] is None
    assert math.isfinite(rows[T_B]["strength"])

    err = capsys.readouterr().err
    assert err.count("[WARN]") == 1
    assert "2 trial reference" in err

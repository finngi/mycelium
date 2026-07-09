"""Ratings: Bradley-Terry strengths over Comparison records, computed at the
Board layer and never stored -- a rating is a property of the whole
comparison population, not of one trial, so it goes stale the instant a new
comparison arrives (docs/design/claims-under-review.md section C). Sibling
to board.py rather than folded into it: board.py aggregates one scalar
metric per trial, this aggregates a relation between pairs of trials, a
different enough shape to warrant its own module.
"""

import sys
from collections import defaultdict

from reishi.primitives.comparison import Comparison
from reishi.primitives.trial import Trial
from reishi.primitives.trial import load_all as load_all_trials

_MAX_ITERS = 200
_TOL = 1e-9
# Every item plays one virtual tie against a phantom opponent fixed at
# strength 1.0. Without it, an item that has only ever won has zero recorded
# losses, so the MM update's denominator for that item never accumulates any
# opposing-strength term and its strength diverges to infinity (the mirror
# case -- only ever lost -- collapses to zero the same way). The phantom tie
# bounds both directions and, as a side effect, connects every item to every
# other (directly or via the phantom), so a comparison set with more than one
# disconnected component still converges instead of leaving one component's
# strengths on an arbitrary, incomparable scale. Fixing the phantom's own
# strength at 1.0 also anchors the overall scale, so no separate
# normalization pass is needed.
_PHANTOM_STRENGTH = 1.0
_PHANTOM_GAME_WEIGHT = 1.0  # one virtual game vs the phantom
_PHANTOM_WIN_CREDIT = 0.5  # ...scored as a tie, i.e. half a win


def ratings(
    comparisons: list[Comparison], *, trials: list[Trial] | None = None
) -> list[dict]:
    if not comparisons:
        return []

    if trials is None:
        trials = load_all_trials()
    recipe_by_trial = {t.id: t.recipe for t in trials}

    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    ties: dict[str, int] = defaultdict(int)
    n: dict[str, int] = defaultdict(int)
    # (winner, loser) -> weighted win count; a tie is scored as half a win in
    # each direction (the standard Davidson-lite simplification for handling
    # ties in a Bradley-Terry fit without a full Davidson tie parameter).
    pairwise: dict[tuple[str, str], float] = defaultdict(float)
    items: set[str] = set()

    for c in comparisons:
        a, b = c.trial_a, c.trial_b
        items.add(a)
        items.add(b)
        n[a] += 1
        n[b] += 1
        if c.winner == "a":
            wins[a] += 1
            losses[b] += 1
            pairwise[(a, b)] += 1.0
        elif c.winner == "b":
            wins[b] += 1
            losses[a] += 1
            pairwise[(b, a)] += 1.0
        elif c.winner == "tie":
            ties[a] += 1
            ties[b] += 1
            pairwise[(a, b)] += 0.5
            pairwise[(b, a)] += 0.5
        else:
            raise ValueError(f"comparison {c.id!r} has unknown winner {c.winner!r}")

    unknown = sorted(item for item in items if item not in recipe_by_trial)
    if unknown:
        print(
            f"[WARN] ratings: {len(unknown)} trial reference(s) have no loadable "
            "trial manifest (archived or deleted?); rating them anyway",
            file=sys.stderr,
        )

    strength = _fit_bradley_terry(items, pairwise)

    rows = [
        {
            "trial": item,
            "recipe": recipe_by_trial.get(item),
            "wins": wins[item],
            "losses": losses[item],
            "ties": ties[item],
            "n": n[item],
            "strength": strength[item],
        }
        for item in items
    ]
    return sorted(rows, key=lambda r: r["strength"], reverse=True)


def _fit_bradley_terry(
    items: set[str], pairwise: dict[tuple[str, str], float]
) -> dict[str, float]:
    """Minorization-maximization fit of Bradley-Terry strengths p_i > 0 where
    P(i beats j) = p_i / (p_i + p_j) -- Hunter (2004)'s generalization of
    Zermelo's algorithm. Pure python: numpy/scipy are not workspace
    dependencies. See the module docstring for the phantom-opponent
    regularization that keeps this finite for undefeated/never-defeated
    items and disconnected comparison sets.
    """
    strength = dict.fromkeys(items, 1.0)
    for _ in range(_MAX_ITERS):
        updated = {}
        for i in items:
            win_total = _PHANTOM_WIN_CREDIT
            denom = _PHANTOM_GAME_WEIGHT / (strength[i] + _PHANTOM_STRENGTH)
            for j in items:
                if j == i:
                    continue
                win_total += pairwise.get((i, j), 0.0)
                games = pairwise.get((i, j), 0.0) + pairwise.get((j, i), 0.0)
                if games:
                    denom += games / (strength[i] + strength[j])
            updated[i] = win_total / denom if denom else strength[i]
        delta = max(abs(updated[i] - strength[i]) for i in items)
        strength = updated
        if delta < _TOL:
            break
    return strength

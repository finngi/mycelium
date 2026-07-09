"""Board: computes a per-recipe ranking over done trial manifests and returns
the rows. Reads trials, persists nothing.
"""

import sys
from collections import defaultdict

from reishi.primitives import trial as trial_store


def _is_scalar(v: object) -> bool:
    # Only scalar leaves are coordinates the board can aggregate. bool is
    # excluded: a bare True/False is more likely a stray flag than an intended
    # 0/1 -- rate-style metrics come from an aggregator, not a raw bool leaf.
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def build(metric: str = "f1", task: str | None = None) -> list[dict]:
    trials = [t for t in trial_store.load_all() if t.status == "done"]
    if task:
        trials = [t for t in trials if t.spec.get("task") == task]

    by_recipe: dict[str, list] = defaultdict(list)
    for t in trials:
        by_recipe[t.recipe].append(t)

    rows = []
    for recipe_name, group in by_recipe.items():
        values = []
        for t in group:
            if metric not in t.metrics:
                continue
            v = t.metrics[metric]
            if not _is_scalar(v):
                print(
                    f"[WARN] metric '{metric}' on trial '{t.id}' is non-scalar; skipped",
                    file=sys.stderr,
                )
                continue
            values.append(v)
        row = {
            "recipe": recipe_name,
            "task": group[0].spec.get("task"),
            "base_model": group[0].spec.get("base_model"),
            "trials": len(group),
            "scored": len(values),
        }
        # A recipe whose trials were never scored is a visible state, not an
        # absence -- emit the row with null metrics rather than dropping it.
        if values:
            row[metric] = sum(values) / len(values)
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
        else:
            row[metric] = None
            row[f"{metric}_min"] = None
            row[f"{metric}_max"] = None
        rows.append(row)
    return sorted(
        rows,
        key=lambda r: r[metric] if r[metric] is not None else float("-inf"),
        reverse=True,
    )

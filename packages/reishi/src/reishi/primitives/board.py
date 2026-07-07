"""Board: aggregation over trial manifests. Computed, never stored as truth."""

from collections import defaultdict

from reishi.primitives import trial as trial_store


def build(metric: str = "f1", task: str | None = None) -> list[dict]:
    trials = [t for t in trial_store.load_all() if t.status == "done"]
    if task:
        trials = [t for t in trials if t.spec.get("task") == task]

    by_recipe: dict[str, list] = defaultdict(list)
    for t in trials:
        by_recipe[t.recipe].append(t)

    rows = []
    for recipe_name, group in by_recipe.items():
        values = [t.metrics[metric] for t in group if metric in t.metrics]
        row = {
            "recipe": recipe_name,
            "task": group[0].spec.get("task"),
            "base_model": group[0].spec.get("base_model"),
            "trials": len(group),
            "scored": len(values),
        }
        # An all-unscored recipe is a real, visible state (e.g. enoki trials
        # never scored), not an absence -- emit the row with null metrics
        # rather than dropping it.
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

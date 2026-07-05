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
        if not values:
            continue
        rows.append(
            {
                "recipe": recipe_name,
                "task": group[0].spec.get("task"),
                "base_model": group[0].spec.get("base_model"),
                "trials": len(group),
                metric: sum(values) / len(values),
                f"{metric}_min": min(values),
                f"{metric}_max": max(values),
            }
        )
    return sorted(rows, key=lambda r: r[metric], reverse=True)

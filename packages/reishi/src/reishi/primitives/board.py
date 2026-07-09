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


# Measurement-key (K) fields from EvalInfo, see math-foundations.md section 0.
# Two trials are only comparable if they agree on all of these.
_K_FIELDS = (
    "task",
    "codec",
    "scorer",
    "scorer_version",
    "dataset",
    "dataset_revision",
    "split",
    "eval_n",
)


def _mismatched_key_fields(trials: list) -> list[str]:
    # A trial with no eval info contributes nothing to any field's value set,
    # so it can't create a mismatch by itself; only fields two or more
    # trials actually pin, and pin to different values, count as mixed.
    mismatched = []
    for field in _K_FIELDS:
        values = {t.eval[field] for t in trials if t.eval.get(field) is not None}
        if len(values) > 1:
            mismatched.append(field)
    return mismatched


def build(metric: str = "f1", task: str | None = None) -> list[dict]:
    trials = [t for t in trial_store.load_all() if t.status == "done"]
    if task:
        trials = [t for t in trials if t.spec.get("task") == task]

    by_recipe: dict[str, list] = defaultdict(list)
    for t in trials:
        by_recipe[t.recipe].append(t)

    rows = []
    # Warn once per (recipe, metric), not per trial: a recipe whose task stores
    # rich metrics on every trial would otherwise flood stderr and bury other
    # warnings sharing the stream.
    non_scalar: set[tuple[str, str]] = set()
    mixed_keys: dict[str, list[str]] = {}
    for recipe_name, group in by_recipe.items():
        values = []
        contributing = []
        for t in group:
            if metric not in t.metrics:
                continue
            v = t.metrics[metric]
            if not _is_scalar(v):
                non_scalar.add((recipe_name, metric))
                continue
            values.append(v)
            contributing.append(t)
        mismatched = _mismatched_key_fields(contributing)
        if mismatched:
            mixed_keys[recipe_name] = mismatched
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
    for recipe_name, metric_name in sorted(non_scalar):
        print(
            f"[WARN] metric '{metric_name}' on recipe '{recipe_name}' has non-scalar values; skipped",
            file=sys.stderr,
        )
    for recipe_name in sorted(mixed_keys):
        fields = ", ".join(mixed_keys[recipe_name])
        print(
            f"[WARN] recipe '{recipe_name}' mixes measurement keys ({fields}); "
            "averaging across different measurements",
            file=sys.stderr,
        )
    return sorted(
        rows,
        key=lambda r: r[metric] if r[metric] is not None else float("-inf"),
        reverse=True,
    )

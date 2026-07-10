"""Sweep: a recipe template plus a search space over its hparams.

Each suggested point becomes an ordinary reishi Recipe; a Sweep never trains
anything itself.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import NotRequired, TypedDict

import yaml

from reishi.primitives.recipe import RUNTIMES, Recipe, RecipeManifest


class ParamSpec(TypedDict):
    type: str  # "float" | "loguniform" | "int" | "categorical"
    low: NotRequired[float]
    high: NotRequired[float]
    step: NotRequired[float]
    choices: NotRequired[list[object]]


class ObjectiveSpec(TypedDict):
    # Plain key (looked up in ProducerResult["metrics"], e.g. "f1") or a
    # dotted path into a trial manifest namespace ("metrics.field_f1",
    # "observables.wall_time_s") -- see physarum.objective.resolve_metric.
    metric: str
    direction: str  # "maximize" | "minimize"


# Epsilon-constraint, not a weight: math-foundations.md section 1 proves a
# weighted sum can't reach every Pareto-optimal compromise, while "maximise
# the objective subject to this bound" can. `metric` uses the same path
# language as ObjectiveSpec.metric; exactly one of max/min applies per entry.
class ConstraintSpec(TypedDict):
    metric: str
    max: NotRequired[float]
    min: NotRequired[float]


class SweepManifest(TypedDict):
    name: str
    template: RecipeManifest
    search_space: dict[str, ParamSpec]
    goal: ObjectiveSpec
    constraints: list[ConstraintSpec]
    sampler: str
    n_trials: int


def _recipe_from_dict(d: dict) -> Recipe:
    known = {f for f in Recipe.__dataclass_fields__}
    unknown = set(d) - known
    if unknown:
        raise ValueError(
            f"sweep template: unknown recipe fields: {', '.join(sorted(unknown))}"
        )
    missing = {"name", "task"} - set(d)
    if missing:
        raise ValueError(
            f"sweep template: missing required fields: {', '.join(sorted(missing))}"
        )
    if not d.get("train_dataset") and not d.get("eval_dataset"):
        raise ValueError(
            "sweep template: needs at least one of 'train_dataset' or 'eval_dataset'"
        )
    return Recipe(**d)


@dataclass(frozen=True)
class Sweep:
    name: str
    template: RecipeManifest
    search_space: dict[str, ParamSpec]
    goal: ObjectiveSpec
    constraints: list[ConstraintSpec] = field(default_factory=list)
    sampler: str = "tpe"  # validated by the resolved backend, not here
    n_trials: int = 20

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Sweep":
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: sweep yaml must be a mapping")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"{path}: unknown sweep fields: {', '.join(sorted(unknown))}"
            )
        missing = {"name", "template", "search_space", "goal"} - set(raw)
        if missing:
            raise ValueError(
                f"{path}: missing required fields: {', '.join(sorted(missing))}"
            )
        if not isinstance(raw["template"], dict):
            raise ValueError(f"{path}: template must be a mapping")
        template = _recipe_from_dict(raw["template"]).to_manifest()
        return cls(**{**raw, "template": template})

    def validate(self) -> None:
        from reishi.primitives import task as task_registry

        task_registry.get(self.template["task"])
        if self.template["runtime"] not in RUNTIMES:
            raise ValueError(f"unknown runtime '{self.template['runtime']}'")
        if self.goal["direction"] not in ("maximize", "minimize"):
            raise ValueError("goal.direction must be 'maximize' or 'minimize'")
        for c in self.constraints:
            if "metric" not in c:
                raise ValueError(f"constraint missing 'metric': {c}")
            if ("max" in c) == ("min" in c):
                raise ValueError(
                    f"constraint on '{c['metric']}' must set exactly one of 'max' or 'min'"
                )
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if not self.search_space:
            raise ValueError("search_space must have at least one parameter")
        for key in self.search_space:
            if not key.startswith("hparams."):
                raise ValueError(
                    f"search_space key '{key}' must start with 'hparams.' (only hparams are swept)"
                )

    def to_manifest(self) -> SweepManifest:
        return {
            "name": self.name,
            "template": self.template,
            "search_space": dict(self.search_space),
            "goal": dict(self.goal),
            "constraints": [dict(c) for c in self.constraints],
            "sampler": self.sampler,
            "n_trials": self.n_trials,
        }

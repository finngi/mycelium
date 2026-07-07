"""Sweep: a recipe template plus a search space over its trainer hyperparameters.

physarum's own primitive, layered on reishi's Recipe -- a sweep never trains
anything itself; each suggested point becomes an ordinary Recipe, planned and
run through the exact same Trainer contract oyster and enoki already
implement.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

import yaml

from reishi.primitives.recipe import ACCELERATORS, Recipe, RecipeManifest


class ParamSpec(TypedDict):
    type: str  # "float" | "loguniform" | "int" | "categorical"
    low: NotRequired[float]
    high: NotRequired[float]
    step: NotRequired[float]
    choices: NotRequired[list[object]]


class ObjectiveSpec(TypedDict):
    metric: str  # key into TrainerResult["metrics"], e.g. "f1"
    direction: str  # "maximize" | "minimize"


class SweepManifest(TypedDict):
    name: str
    template: RecipeManifest
    search_space: dict[str, ParamSpec]
    objective: ObjectiveSpec
    sampler: str
    n_trials: int


def _recipe_from_dict(d: dict) -> Recipe:
    known = {f for f in Recipe.__dataclass_fields__}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"sweep template: unknown recipe fields: {', '.join(sorted(unknown))}")
    missing = {"name", "task", "dataset"} - set(d)
    if missing:
        raise ValueError(f"sweep template: missing required fields: {', '.join(sorted(missing))}")
    return Recipe(**d)


@dataclass(frozen=True)
class Sweep:
    name: str
    template: RecipeManifest
    search_space: dict[str, ParamSpec]
    objective: ObjectiveSpec
    sampler: str = "tpe"  # a name the resolved search backend understands; validating it is that backend's job, not this manifest's
    n_trials: int = 20

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Sweep":
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: sweep yaml must be a mapping")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"{path}: unknown sweep fields: {', '.join(sorted(unknown))}")
        missing = {"name", "template", "search_space", "objective"} - set(raw)
        if missing:
            raise ValueError(f"{path}: missing required fields: {', '.join(sorted(missing))}")
        if not isinstance(raw["template"], dict):
            raise ValueError(f"{path}: template must be a mapping")
        template = _recipe_from_dict(raw["template"]).to_manifest()
        return cls(**{**raw, "template": template})

    def validate(self) -> None:
        from reishi.primitives import task as task_registry

        task_registry.get(self.template["task"])
        if self.template["accelerator"] not in ACCELERATORS:
            raise ValueError(f"unknown accelerator '{self.template['accelerator']}'")
        if self.objective["direction"] not in ("maximize", "minimize"):
            raise ValueError("objective.direction must be 'maximize' or 'minimize'")
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if not self.search_space:
            raise ValueError("search_space must have at least one parameter")
        for key in self.search_space:
            if not key.startswith("trainer."):
                raise ValueError(f"search_space key '{key}' must start with 'trainer.' (only trainer hyperparameters are swept)")

    def to_manifest(self) -> SweepManifest:
        return {
            "name": self.name,
            "template": self.template,
            "search_space": dict(self.search_space),
            "objective": dict(self.objective),
            "sampler": self.sampler,
            "n_trials": self.n_trials,
        }

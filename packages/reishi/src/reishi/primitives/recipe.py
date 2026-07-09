"""Recipe: a frozen spec of model x dataset x prompt x trainer, loaded from
YAML (from_yaml), checked (validate), and serialized (to_manifest). It holds
fields only; nothing here trains -- executors read the manifest and act on it.
"""

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import yaml

ACCELERATORS = ("local", "mlx", "l4", "h100", "v5e")


def _describe_unknown(unknown: set[str], known: set[str]) -> str:
    lines = []
    for field_name in sorted(unknown):
        match = difflib.get_close_matches(field_name, known, n=1)
        suffix = f" -- did you mean '{match[0]}'?" if match else ""
        lines.append(f"  '{field_name}'{suffix}")
    return "\n".join(lines)


class RecipeManifest(TypedDict):
    name: str
    task: str
    base_model: str | None
    dataset: str
    accelerator: str
    prompt: str | None
    seeds: int
    priority: int
    trainer: dict  # free-form hyperparameters; shape not validated here


@dataclass(frozen=True)
class Recipe:
    name: str
    task: str
    dataset: str
    base_model: str | None = None
    accelerator: str = "l4"
    prompt: str | None = None
    seeds: int = 1
    priority: int = 0
    trainer: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Recipe":
        """Load and validate a Recipe from a YAML file.

        Unknown fields are a hard error, not a warning: recipes are
        human-authored, and a typo like `trainer_kwarg` for `trainer_kwargs`
        would otherwise misconfigure a training run silently. Forward
        compatibility only matters at authoring time -- once planned, the
        spec rides as an opaque dict on `Trial.spec`, so tightening this
        check doesn't affect round-tripping.
        """
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: recipe yaml must be a mapping")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"{path}: unknown recipe fields:\n" + _describe_unknown(unknown, known)
            )
        missing = {"name", "task", "dataset"} - set(raw)
        if missing:
            raise ValueError(
                f"{path}: missing required fields: {', '.join(sorted(missing))}"
            )
        return cls(**raw)

    def validate(self) -> None:
        from reishi.primitives import task as task_registry

        task_registry.get(self.task)
        if self.accelerator not in ACCELERATORS:
            raise ValueError(
                f"unknown accelerator '{self.accelerator}' (one of {', '.join(ACCELERATORS)})"
            )
        if self.seeds < 1:
            raise ValueError("seeds must be >= 1")

    def to_manifest(self) -> RecipeManifest:
        return {
            "name": self.name,
            "task": self.task,
            "base_model": self.base_model,
            "dataset": self.dataset,
            "accelerator": self.accelerator,
            "prompt": self.prompt,
            "seeds": self.seeds,
            "priority": self.priority,
            "trainer": dict(self.trainer),
        }

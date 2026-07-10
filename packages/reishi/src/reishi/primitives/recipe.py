"""Recipe: a frozen spec of model x dataset x prompt x hparams, loaded from
YAML (from_yaml), checked (validate), and serialized (to_manifest). It holds
fields only; nothing here trains -- executors read the manifest and act on it.
"""

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import yaml

RUNTIMES = ("cpu", "mlx", "l4", "h100", "v5e")


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
    train_dataset: str | None
    eval_dataset: str | None
    runtime: str
    prompt: str | None
    n_seeds: int
    priority: int
    hparams: dict  # free-form hyperparameters; shape not validated here


@dataclass(frozen=True)
class Recipe:
    name: str
    task: str
    train_dataset: str | None = None
    eval_dataset: str | None = None
    base_model: str | None = None
    runtime: str = "l4"
    prompt: str | None = None
    n_seeds: int = 1
    priority: int = 0
    hparams: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Recipe":
        """Load and validate a Recipe from a YAML file.

        Unknown fields are a hard error, not a warning: recipes are
        human-authored, and a typo like `hparams_kwarg` for `hparams_kwargs`
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
        missing = {"name", "task"} - set(raw)
        if missing:
            raise ValueError(
                f"{path}: missing required fields: {', '.join(sorted(missing))}"
            )
        if not raw.get("train_dataset") and not raw.get("eval_dataset"):
            raise ValueError(
                f"{path}: recipe needs at least one of 'train_dataset' or 'eval_dataset'"
            )
        return cls(**raw)

    def validate(self) -> None:
        from reishi.primitives import task as task_registry

        task_registry.get(self.task)
        if self.runtime not in RUNTIMES:
            raise ValueError(
                f"unknown runtime '{self.runtime}' (one of {', '.join(RUNTIMES)})"
            )
        if self.n_seeds < 1:
            raise ValueError("n_seeds must be >= 1")
        if not self.train_dataset and not self.eval_dataset:
            raise ValueError(
                "recipe needs at least one of 'train_dataset' or 'eval_dataset'"
            )

    def to_manifest(self) -> RecipeManifest:
        return {
            "name": self.name,
            "task": self.task,
            "base_model": self.base_model,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.eval_dataset,
            "runtime": self.runtime,
            "prompt": self.prompt,
            "n_seeds": self.n_seeds,
            "priority": self.priority,
            "hparams": dict(self.hparams),
        }

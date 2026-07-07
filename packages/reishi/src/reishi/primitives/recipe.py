"""Recipe: declarative model x dataset x prompt x trainer spec.

A recipe never says HOW to train — the accelerator field selects a trainer
(TRL/PEFT on CUDA, XLA/JAX on TPU) at execution time.

base_model is optional: absent means training from scratch, with the
architecture described in the trainer spec.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import yaml

ACCELERATORS = ("local", "mlx", "l4", "h100", "v5e")


class RecipeManifest(TypedDict):
    name: str
    task: str
    base_model: str | None
    dataset: str
    accelerator: str
    prompt: str | None
    seeds: int
    priority: int
    trainer: dict  # per-accelerator hyperparameters -- shape varies by trainer, not a fixed contract


@dataclass(frozen=True)
class Recipe:
    name: str
    task: str
    dataset: str
    base_model: str | None = None
    accelerator: str = "l4"
    prompt: str | None = None
    seeds: int = 1
    priority: int = 0  # higher claims first; ties break oldest-first
    trainer: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Recipe":
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: recipe yaml must be a mapping")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"{path}: unknown recipe fields: {', '.join(sorted(unknown))}")
        missing = {"name", "task", "dataset"} - set(raw)
        if missing:
            raise ValueError(f"{path}: missing required fields: {', '.join(sorted(missing))}")
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

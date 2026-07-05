"""Task: the definition of a problem — output schema, codec, decoder, scorer.

The scorer is the invariant that keeps trials comparable across trainers and
accelerators: an L4 trial and a v5e trial of the same task land on the same
board only because they were scored by the same function.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    output_fields: tuple[str, ...]
    score: Callable[[dict, dict], dict] | None = None
    codec: str = "json"

    def to_manifest(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "output_fields": list(self.output_fields),
            "codec": self.codec,
            "scorer": "registered" if self.score else "missing",
        }


_REGISTRY: dict[str, Task] = {}


def register(task: Task) -> Task:
    if task.name in _REGISTRY:
        raise ValueError(f"task '{task.name}' already registered")
    _REGISTRY[task.name] = task
    return task


def get(name: str) -> Task:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"unknown task '{name}' (registered: {known})")
    return _REGISTRY[name]


def all_tasks() -> list[Task]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]

"""Task: the definition of a problem — output schema, codec, decoder, scorer.

The scorer is the invariant that keeps trials comparable across trainers and
accelerators: an L4 trial and a v5e trial of the same task land on the same
board only because they were scored by the same function.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NotRequired, TypedDict


class ScoreCounts(TypedDict):
    """One example's contribution to a trial's aggregate metrics.

    Task.score's type hint only promises "returns a dict" -- a task's scorer
    could return any shape. This is the one shape aggregate() understands.
    """

    tp: int
    fp: int
    fn: int
    exact_match: bool
    invalid: bool


class AggregateMetrics(TypedDict):
    n: int
    field_precision: NotRequired[float]
    field_recall: NotRequired[float]
    field_f1: NotRequired[float]
    exact_match: NotRequired[float]
    invalid_output_rate: NotRequired[float]


class TaskManifest(TypedDict):
    name: str
    description: str
    output_fields: list[str]
    codec: str
    scorer: str


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    output_fields: tuple[str, ...]
    score: Callable[[dict, dict], ScoreCounts] | None = None
    codec: str = "json"

    def to_manifest(self) -> TaskManifest:
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


def aggregate(scores: list[ScoreCounts]) -> AggregateMetrics:
    """Combine per-example score() outputs into a trial's final metrics.

    Standard shape for a Task.score(pred, gold) return: {tp, fp, fn,
    exact_match, invalid} counts for one example. Any task using that shape
    gets micro precision/recall/f1 plus exact-match and invalid-output rates
    for free -- the field-level detail is what varies per task, not how
    per-example counts roll up into a trial-level number.
    """
    n = len(scores)
    if n == 0:
        return {"n": 0}
    tp = sum(s.get("tp", 0) for s in scores)
    fp = sum(s.get("fp", 0) for s in scores)
    fn = sum(s.get("fn", 0) for s in scores)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": n,
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": f1,
        "exact_match": sum(1 for s in scores if s.get("exact_match")) / n,
        "invalid_output_rate": sum(1 for s in scores if s.get("invalid")) / n,
    }

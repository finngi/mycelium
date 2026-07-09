"""Task: a named problem's output contract -- decode a model output, score it
against a reference, and roll per-example scores into one dict.

score, decoder, and aggregator are all optional: decode() falls back to the
codec named by `codec`, aggregate() falls back to field_aggregate. pred and
ref are typed Any, so a Task is not tied to any particular output shape. ref
is whatever the row carries -- not necessarily a gold label (a reference-free
scorer may use it to hold the input instead; see math-foundations.md 3(ii)).
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

from reishi.primitives.codec import get_codec

Decoder = Callable[[str], Any]
Scorer = Callable[[Any, Any], Mapping[str, object]]
Aggregator = Callable[[list[Mapping[str, object]]], dict]


class ScoreCounts(TypedDict):
    """Per-example score shape that field_aggregate() consumes. Task.score may
    return any Mapping; this is only the shape the built-in aggregator reads.
    """

    tp: int
    fp: int
    fn: int
    exact_match: bool
    invalid: bool


class AggregateMetrics(TypedDict):
    """Shape field_aggregate() returns. A custom aggregator may return any dict."""

    n: int
    field_precision: NotRequired[float]
    field_recall: NotRequired[float]
    field_f1: NotRequired[float]
    exact_match: NotRequired[float]
    invalid_output_rate: NotRequired[float]


class TaskManifest(TypedDict):
    name: str
    description: str
    output_fields: NotRequired[list[str]]
    codec: str
    scorer: str


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    score: Scorer | None = None
    decoder: Decoder | None = None
    aggregator: Aggregator | None = None
    output_fields: tuple[str, ...] = ()
    codec: str = "json"

    def decode(self, raw: str) -> Any:
        if self.decoder is not None:
            return self.decoder(raw)
        return get_codec(self.codec).decode(raw)

    def aggregate(self, scores: list[Mapping[str, object]]) -> dict:
        agg = self.aggregator or field_aggregate
        return agg(scores)

    def to_manifest(self) -> TaskManifest:
        m: TaskManifest = {
            "name": self.name,
            "description": self.description,
            "codec": self.codec,
            "scorer": "registered" if self.score else "missing",
        }
        if self.output_fields:
            m["output_fields"] = list(self.output_fields)
        return m


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


def field_aggregate(scores: list[Mapping[str, object]]) -> AggregateMetrics:
    """Default aggregator: micro precision/recall/f1 over summed tp/fp/fn, plus
    exact-match and invalid-output rates. Used when a Task sets no aggregator and
    its scorer emits ScoreCounts.
    """
    n = len(scores)
    if n == 0:
        return {"n": 0}
    tp = sum(int(s.get("tp", 0)) for s in scores)
    fp = sum(int(s.get("fp", 0)) for s in scores)
    fn = sum(int(s.get("fn", 0)) for s in scores)
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


aggregate = field_aggregate

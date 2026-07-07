"""Task: the definition of a problem — f(x)=y, plus how to score and roll up.

A Task is a function `X -> Y` with a scorer for `y_hat` vs `y` and an
aggregator over per-example scores. Nothing about it is bound to extraction:
pred/gold are `Any`, decode returns `Any`, aggregate returns a plain dict.
reishi ships a tp/fp/fn field-extraction kit (`field_aggregate`) and a json
codec as opt-in defaults, never a mandate -- a Task supplies its own `decoder`
or `aggregator` to replace them.

The scorer is the invariant that keeps trials comparable across trainers and
accelerators: an L4 trial and a v5e trial of the same task land on the same
board only because they were scored by the same function.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

from reishi.primitives.codec import get_codec

Decoder = Callable[[str], Any]
Scorer = Callable[[Any, Any], Mapping[str, object]]
Aggregator = Callable[[list[Mapping[str, object]]], dict]


class ScoreCounts(TypedDict):
    """The extraction kit's per-example shape -- one option, not the contract.

    A Task.score may return any Mapping. This is the shape field_aggregate()
    understands: tasks whose scorer emits these counts get micro
    precision/recall/f1 plus exact-match and invalid-output rates for free.
    """

    tp: int
    fp: int
    fn: int
    exact_match: bool
    invalid: bool


class AggregateMetrics(TypedDict):
    """The extraction kit's aggregate shape -- one option, not the contract.

    What field_aggregate() returns; a custom aggregator returns whatever dict
    its task needs.
    """

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
    """The extraction kit's roll-up: micro P/R/F1 over tp/fp/fn counts.

    Opt-in default for a Task with no aggregator whose scorer emits ScoreCounts
    ({tp, fp, fn, exact_match, invalid} per example). The field-level detail is
    what varies per task, not how per-example counts roll up into a trial-level
    number.
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

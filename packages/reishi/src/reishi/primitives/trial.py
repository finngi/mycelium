"""Trial: one recipe x seed run as a mutable dataclass that round-trips to a
manifest. status moves through STATUSES (planned -> running -> done/failed),
with metrics and artifact URIs filled in as it goes.
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import NotRequired, TypedDict

from reishi import store
from reishi.primitives.recipe import Recipe, RecipeManifest

STATUSES = ("planned", "running", "done", "failed")


class TrialArtifacts(TypedDict):
    weights: NotRequired[str]  # adapter/checkpoint URI
    # URI of persisted raw predictions, so a trial can be re-scored without re-running the model
    predictions: NotRequired[str]


# Metrics alone don't say what scored the trial or where; EvalInfo records that provenance.
class EvalInfo(TypedDict, total=False):
    scorer: str  # task name / scorer id that produced the metrics
    scored_at: str
    placement: str  # cpu | accelerator | local -- where scoring ran
    source: str  # "live" | "replay"


class ExecutionInfo(TypedDict):
    runner: NotRequired[str]
    claimed_at: NotRequired[str]
    heartbeat: NotRequired[str]
    attempt: NotRequired[int]
    finished_at: NotRequired[str]
    last_error: NotRequired[str]


class TrialManifest(TypedDict):
    id: str
    recipe: str
    seed: int
    status: str
    created: str
    metrics: dict  # task-specific; field-scored tasks store AggregateMetrics here
    artifacts: TrialArtifacts
    spec: RecipeManifest
    execution: ExecutionInfo
    eval: NotRequired[EvalInfo]


@dataclass
class Trial:
    id: str
    recipe: str
    seed: int
    status: str = "planned"
    created: str = ""
    metrics: dict = field(default_factory=dict)
    artifacts: TrialArtifacts = field(default_factory=TrialArtifacts)
    # Empty default only fires for a bare Trial or a load of a manifest missing
    # "spec"; a real recipe always fills every RecipeManifest field.
    spec: RecipeManifest = field(default_factory=dict)  # type: ignore[assignment]
    execution: ExecutionInfo = field(default_factory=ExecutionInfo)
    eval: EvalInfo = field(default_factory=dict)  # type: ignore[assignment]
    # Unknown top-level manifest keys, carried verbatim. Without this a newer
    # manifest loaded in an older checkout would lose its new fields on the next
    # save (oyster's heartbeat re-saves every 30s) -- the additivity guarantee
    # depends on the round trip being lossless.
    extra: dict = field(default_factory=dict)

    def to_manifest(self) -> TrialManifest:
        m = {
            "id": self.id,
            "recipe": self.recipe,
            "seed": self.seed,
            "status": self.status,
            "created": self.created,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "spec": self.spec,
            "execution": self.execution,
            "eval": self.eval,
        }
        # Known keys win, so a stale carried-over value never shadows the live one.
        return {**self.extra, **m}  # type: ignore[typeddict-item]

    @classmethod
    def from_manifest(cls, m: Mapping[str, object]) -> "Trial":
        # Tolerant reader: unknown keys land in `extra` rather than being dropped,
        # so a newer manifest round-trips losslessly through an older checkout.
        # Route both sides off `known` (which excludes the `extra` field itself),
        # so a manifest key literally named "extra" is preserved, not swallowed.
        # The loop widens values to object, so mypy can't check the
        # **kwargs unpacking -- hence the type: ignore.
        known = {k for k in cls.__dataclass_fields__ if k != "extra"}
        fields: dict = {}
        extra: dict = {}
        for k, v in m.items():
            (fields if k in known else extra)[k] = v
        return cls(**fields, extra=extra)  # type: ignore[arg-type]


def plan(recipe: Recipe) -> list[Trial]:
    recipe.validate()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return [
        Trial(
            id=f"{recipe.name}-s{seed}-{uuid.uuid4().hex[:6]}",
            recipe=recipe.name,
            seed=seed,
            created=now,
            spec=recipe.to_manifest(),
        )
        for seed in range(recipe.seeds)
    ]


def save(t: Trial) -> None:
    store.save("trials", t.id, t.to_manifest())


def load(trial_id: str) -> Trial:
    return Trial.from_manifest(store.load("trials", trial_id))


def load_all() -> list[Trial]:
    return [Trial.from_manifest(m) for m in store.load_all("trials")]


def resolve(ref: str) -> Trial:
    matches = [t for t in load_all() if t.id == ref or t.id.startswith(ref)]
    if not matches:
        raise FileNotFoundError(f"no trial matching '{ref}'")
    if len(matches) > 1:
        ids = ", ".join(t.id for t in matches)
        raise ValueError(f"'{ref}' is ambiguous: {ids}")
    return matches[0]

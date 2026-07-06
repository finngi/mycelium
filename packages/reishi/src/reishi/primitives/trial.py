"""Trial: one recipe x seed execution — Ray Tune's term, used deliberately.

A trial is data (a manifest), not a log line: planned -> running -> done/failed,
with metrics and artifact URIs attached along the way.
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
    weights: NotRequired[str]  # adapter/checkpoint URI, local path or hf://<repo>


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
    metrics: dict  # task-specific; extract-style tasks land Task.aggregate's AggregateMetrics here
    artifacts: TrialArtifacts
    spec: RecipeManifest
    execution: ExecutionInfo


@dataclass
class Trial:
    id: str
    recipe: str
    seed: int
    status: str = "planned"
    created: str = ""
    metrics: dict = field(default_factory=dict)
    artifacts: TrialArtifacts = field(default_factory=TrialArtifacts)
    # RecipeManifest's fields are all required (a real recipe always has them); the empty default
    # here only fires for a bare, unplanned Trial or a tolerant from_manifest() load of an old
    # manifest missing "spec" -- neither is a real recipe.
    spec: RecipeManifest = field(default_factory=dict)  # type: ignore[assignment]
    execution: ExecutionInfo = field(default_factory=ExecutionInfo)

    def to_manifest(self) -> TrialManifest:
        return {
            "id": self.id,
            "recipe": self.recipe,
            "seed": self.seed,
            "status": self.status,
            "created": self.created,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "spec": self.spec,
            "execution": self.execution,
        }

    @classmethod
    def from_manifest(cls, m: Mapping[str, object]) -> "Trial":
        # m is raw store.load() output, not yet trusted as TrialManifest-shaped -- unlike
        # to_manifest() (a construction we control), this is a read boundary from disk/Postgres.
        # Tolerant reader: ignore unknown keys so newer manifests load in older checkouts.
        # A dict comprehension over Mapping.items() widens every value to `object` regardless
        # of m's declared shape, so mypy can't verify the **kwargs unpacking below -- rewriting
        # this field-by-field (like Dataset.from_manifest) would fix that, but at the cost of
        # duplicating every dataclass field default here instead of delegating to Trial's own.
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in m.items() if k in known})  # type: ignore[arg-type]


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

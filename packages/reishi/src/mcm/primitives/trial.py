"""Trial: one recipe x seed execution — Ray Tune's term, used deliberately.

A trial is data (a manifest), not a log line: planned -> running -> done/failed,
with metrics and artifact URIs attached along the way.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mcm import store
from mcm.primitives.recipe import Recipe

STATUSES = ("planned", "running", "done", "failed")


@dataclass
class Trial:
    id: str
    recipe: str
    seed: int
    status: str = "planned"
    created: str = ""
    metrics: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)  # e.g. adapter/checkpoint URIs
    spec: dict = field(default_factory=dict)  # frozen copy of the recipe manifest
    execution: dict = field(default_factory=dict)  # executor provenance: runner, claimed_at, heartbeat, attempt

    def to_manifest(self) -> dict:
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
    def from_manifest(cls, m: dict) -> "Trial":
        # Tolerant reader: ignore unknown keys so newer manifests load in older checkouts.
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in m.items() if k in known})


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

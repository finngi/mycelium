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

STATUSES = ("planned", "running", "done", "failed", "pruned")


class TrialArtifacts(TypedDict):
    weights: NotRequired[str]  # adapter/checkpoint URI
    # URI of persisted raw outputs, so a trial can be re-scored without re-running the model
    outputs: NotRequired[str]


# Metrics alone don't say what scored the trial or where; ScoringInfo records that provenance.
class ScoringInfo(TypedDict, total=False):
    scorer: str  # task name / scorer id that produced the metrics
    scored_at: str
    scored_on: str  # cpu | gpu | tpu -- where scoring ran
    source: str  # "live" | "replay"
    # Measurement-key (K) pinning fields, see math-foundations.md section 0:
    # K = (task, codec, scorer closure, aggregator, dataset version, split,
    # n_eval). Two trials are comparable only if they share K; board.build
    # warns when a recipe group mixes these.
    task: str  # Task.name active at scoring time
    codec: str  # Task.codec active at scoring time
    scorer_version: str  # scorer closure id (e.g. oracle model/lib versions)
    aggregator: str  # aggregator identity (module.qualname) -- the roll-up in K
    dataset: str  # Dataset.name -- the eval set's ref
    dataset_revision: str  # Dataset.revision -- the pinned version within K
    split: str  # eval split name (test | val | ood)
    n_eval_rows: int  # number of eval rows scored -- n_eval in K


class ExecutionInfo(TypedDict):
    runner: NotRequired[str]
    claimed_at: NotRequired[str]
    heartbeat: NotRequired[str]
    attempt: NotRequired[int]
    finished_at: NotRequired[str]
    last_error: NotRequired[str]
    log: NotRequired[str]  # captured stdout/stderr path/URI, executor-written


class TrialManifest(TypedDict):
    id: str
    recipe_name: str
    seed: int
    status: str
    created: str
    metrics: dict  # task-specific; field-scored tasks store AggregateMetrics here
    # Executor-written run-resource facts (wall_time_s, cost_usd, artifact_bytes,
    # latency_ms_p50 -- unit-suffixed), disjoint from metrics: a scorer judges
    # answers, only the executor can observe what the run itself cost.
    # See math-foundations.md 3(iii).
    observables: dict
    artifacts: TrialArtifacts
    spec: RecipeManifest
    execution: ExecutionInfo
    scoring: NotRequired[ScoringInfo]
    # One ScoringInfo per eval run (val, OOD, adversarial, ...). Additive
    # alongside `scoring`, which stays the primary/most-recent one -- see
    # record_scoring().
    scorings: NotRequired[list[ScoringInfo]]


@dataclass
class Trial:
    id: str
    recipe_name: str
    seed: int
    status: str = "planned"
    created: str = ""
    metrics: dict = field(default_factory=dict)
    observables: dict = field(default_factory=dict)
    artifacts: TrialArtifacts = field(default_factory=TrialArtifacts)
    # Empty default only fires for a bare Trial or a load of a manifest missing
    # "spec"; a real recipe always fills every RecipeManifest field.
    spec: RecipeManifest = field(default_factory=dict)  # type: ignore[assignment]
    execution: ExecutionInfo = field(default_factory=ExecutionInfo)
    # `scoring` is the primary/most-recent eval; `scorings` is the full multi-eval-set
    # history (val + OOD + adversarial, ...) -- record_scoring() keeps both in sync.
    scoring: ScoringInfo = field(default_factory=dict)  # type: ignore[assignment]
    scorings: list[ScoringInfo] = field(default_factory=list)
    # Unknown top-level manifest keys, carried verbatim. Without this a newer
    # manifest loaded in an older checkout would lose its new fields on the next
    # save (oyster's heartbeat re-saves every 30s) -- the additivity guarantee
    # depends on the round trip being lossless.
    extra: dict = field(default_factory=dict)

    def to_manifest(self) -> TrialManifest:
        m = {
            "id": self.id,
            "recipe_name": self.recipe_name,
            "seed": self.seed,
            "status": self.status,
            "created": self.created,
            "metrics": self.metrics,
            "observables": self.observables,
            "artifacts": self.artifacts,
            "spec": self.spec,
            "execution": self.execution,
            "scoring": self.scoring,
            "scorings": self.scorings,
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


def record_scoring(trial: Trial, metrics: dict, info: ScoringInfo) -> None:
    """Record one eval run (val, OOD, adversarial, ...): dual-write, never a
    relocate. Appends `info` to `trial.scorings` and sets `trial.scoring` to it
    (the primary = most recent), then merges `metrics` into `trial.metrics` bare
    (so existing board/objective/watch consumers stay sighted) and, when
    `info` carries a split, again under a `'<split>/'` prefix. No split ->
    bare write only. An existing metrics key is only ever overwritten with a
    newer value for the same name, never deleted or renamed.
    """
    trial.scorings.append(info)
    trial.scoring = info
    split = info.get("split")
    for key, value in metrics.items():
        trial.metrics[key] = value
        if split:
            trial.metrics[f"{split}/{key}"] = value


def plan(recipe: Recipe) -> list[Trial]:
    recipe.validate()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return [
        Trial(
            id=f"{recipe.name}-s{seed}-{uuid.uuid4().hex[:6]}",
            recipe_name=recipe.name,
            seed=seed,
            created=now,
            spec=recipe.to_manifest(),
        )
        for seed in range(recipe.n_seeds)
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

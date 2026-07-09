"""Turns one Sweep and one already-resolved Trainer into a search backend's objective.

physarum's real job is consuming trials for an optimize -> result -> suggest
-> action loop; Optuna is one pluggable search backend behind that loop, not
its foundation, so nothing here is typed against Optuna directly -- only
against the scalar-suggestion surface a backend must offer (see Suggester).
Building an actual mcm Trial from those suggestions, running it, and
reporting the result back is entirely this module's job, not the backend's.
Keeping Trainer structural (not imported from oyster/enoki) means this
module has no sibling dependency beyond reishi.
"""

from typing import Callable, Protocol, TypedDict

from reishi.primitives import trial as trial_store
from reishi.primitives.recipe import Recipe, RecipeManifest
from reishi.primitives.trial import TrialArtifacts, TrialManifest

from physarum.primitives.sweep import ParamSpec, Sweep


class TrainerResult(TypedDict):
    metrics: dict
    artifacts: TrialArtifacts


Trainer = Callable[[TrialManifest], TrainerResult]


class Suggester(Protocol):
    """The only surface a search backend must offer to drive a Sweep -- Optuna's
    Trial satisfies this structurally, so nothing here imports optuna."""

    number: int  # this suggestion's ordinal within its sweep, used to name the Trial it becomes

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: float | None = None,
    ) -> float: ...
    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1) -> int: ...
    def suggest_categorical(self, name: str, choices: list[object]) -> object: ...
    def set_user_attr(self, key: str, value: object) -> None: ...


def suggest(ot: Suggester, search_space: dict[str, ParamSpec]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, spec in search_space.items():
        kind = spec["type"]
        if kind == "loguniform":
            out[key] = ot.suggest_float(key, spec["low"], spec["high"], log=True)
        elif kind == "float":
            out[key] = ot.suggest_float(
                key, spec["low"], spec["high"], step=spec.get("step")
            )
        elif kind == "int":
            out[key] = ot.suggest_int(
                key, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1))
            )
        elif kind == "categorical":
            out[key] = ot.suggest_categorical(key, spec["choices"])
        else:
            raise ValueError(f"unknown search_space type '{kind}' for '{key}'")
    return out


def build_recipe(
    template: RecipeManifest,
    suggested: dict[str, object],
    sweep_name: str,
    trial_number: int,
) -> Recipe:
    trainer_cfg = dict(template["trainer"])
    for key, value in suggested.items():
        trainer_cfg[key.removeprefix("trainer.")] = value
    return Recipe(
        name=f"{sweep_name}-t{trial_number}",
        task=template["task"],
        dataset=template["dataset"],
        base_model=template["base_model"],
        accelerator=template["accelerator"],
        prompt=template["prompt"],
        seeds=1,  # one suggestion -> exactly one Trial, by design
        trainer=trainer_cfg,
    )


def make_objective(sweep: Sweep, trainer_fn: Trainer) -> Callable[[Suggester], float]:
    metric = sweep.objective["metric"]

    def objective(ot: Suggester) -> float:
        suggested = suggest(ot, sweep.search_space)
        recipe = build_recipe(sweep.template, suggested, sweep.name, ot.number)

        [t] = trial_store.plan(recipe)  # validates recipe internally
        trial_store.save(t)
        ot.set_user_attr(
            "mcm_trial_id", t.id
        )  # the only link between the search backend's state and reishi's

        try:
            result = trainer_fn(t.to_manifest())
            if metric not in result["metrics"]:
                available = ", ".join(sorted(result["metrics"])) or "none"
                raise KeyError(
                    f"sweep objective metric '{metric}' not in trial metrics (available: {available})"
                )
            t.metrics, t.artifacts, t.status = (
                result["metrics"],
                result["artifacts"],
                "done",
            )
            trial_store.save(t)
        except Exception as e:
            # One bad trial (a trainer crash, a missing metric) must not sink a
            # 60-trial sweep -- mark it failed and re-raise so the search
            # backend's own per-trial failure handling (Optuna's `catch=`, set
            # at the study.optimize() call site) can record it and move on,
            # instead of the exception unwinding study.optimize() entirely and
            # losing every trial after it.
            t.status = "failed"
            t.execution = {**t.execution, "last_error": str(e)}
            trial_store.save(t)
            raise

        return t.metrics[metric]

    return objective

"""Turns one Sweep and one resolved Trainer into a search backend's objective.

The Suggester Protocol below is structural: the objective takes only the
scalar-suggestion/report surface a backend must offer, and Trainer is a
structural callable too. Optuna's own Trial satisfies Suggester without a
wrapper, so the one unavoidable Optuna import here is optuna.TrialPruned --
the only way to signal a pruned trial back to study.optimize().
"""

from typing import Callable, Protocol, TypedDict

import optuna

from reishi.primitives import trial as trial_store
from reishi.primitives.recipe import Recipe, RecipeManifest
from reishi.primitives.trial import TrialArtifacts, TrialManifest

from physarum.primitives.sweep import ParamSpec, Sweep


class TrainerResult(TypedDict):
    metrics: dict
    artifacts: TrialArtifacts


Trainer = Callable[[TrialManifest], TrainerResult]


class Suggester(Protocol):
    """The surface a search backend must offer to drive a Sweep. Optuna's
    Trial satisfies it structurally -- report/should_prune are Trial's own
    method names and signatures, so no adapter is needed for those either."""

    number: int  # ordinal within the sweep; names the Trial it becomes

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
    def report(self, value: float, step: int) -> None: ...
    def should_prune(self) -> bool: ...


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
        seeds=1,  # one suggestion -> exactly one Trial
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
        )  # the only link between the backend's state and reishi's

        try:
            result = trainer_fn(t.to_manifest())
            if metric not in result["metrics"]:
                available = ", ".join(sorted(result["metrics"])) or "none"
                raise KeyError(
                    f"sweep objective metric '{metric}' not in trial metrics (available: {available})"
                )
            value = result["metrics"][metric]
            # trainer_fn runs synchronously to completion -- there is no
            # mid-training callback yet (see physarum AGENTS.md gap #1), so
            # there's only ever one step to report. This can still flag a
            # trial as statistically weak against the rest of the study, but
            # unlike real ASHA/Hyperband pruning it can never abort training
            # early to save compute.
            ot.report(value, step=0)
            if ot.should_prune():
                t.metrics, t.artifacts, t.status = (
                    result["metrics"],
                    result["artifacts"],
                    "pruned",
                )
                trial_store.save(t)
                raise optuna.TrialPruned()
            t.metrics, t.artifacts, t.status = (
                result["metrics"],
                result["artifacts"],
                "done",
            )
            trial_store.save(t)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            # Mark failed before re-raising: the backend's per-trial handling
            # (Optuna's catch= at the study.optimize() call site) then records
            # it and moves on, rather than the exception sinking the whole sweep.
            t.status = "failed"
            t.execution = {**t.execution, "last_error": str(e)}
            trial_store.save(t)
            raise

        return t.metrics[metric]

    return objective

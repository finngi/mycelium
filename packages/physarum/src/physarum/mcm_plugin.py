"""mcm plugin registering the `sweep` domain.

mcm sweep optimize sweep.yaml   # run a sweep to completion, in-process
mcm sweep watch my-sweep        # localhost page graphing its trials live
"""

import sys
from collections.abc import Callable
from datetime import datetime, timezone
from typing import cast

import optuna

import reishi.tasks  # noqa: F401  (populate the task registry)
from reishi import store
from reishi.cli.grammar import Command, Verb
from reishi.cli.output import emit

from physarum.objective import Trainer, make_objective
from physarum.primitives.sweep import ParamSpec, Sweep
from physarum.watch import DEFAULT_PORT, serve as watch_serve

DOMAINS = ("sweep",)
VERBS = (
    Verb(
        "optimize", home="sweep", readonly=False
    ),  # not "run": that verb's home domain is "recipe"
    Verb("watch", home="sweep", readonly=True),
)

_SAMPLERS = {
    "tpe": optuna.samplers.TPESampler,
    "cmaes": optuna.samplers.CmaEsSampler,
    "random": optuna.samplers.RandomSampler,
}


_GridValue = str | float | int | bool | None


def _grid_search_space(
    search_space: dict[str, ParamSpec],
) -> dict[str, list[_GridValue]]:
    # GridSampler needs every value it will ever suggest listed up front, so
    # only enumerable types have a full grid: continuous float/loguniform
    # params can't feed it.
    grid: dict[str, list[_GridValue]] = {}
    for key, spec in search_space.items():
        if spec["type"] == "categorical":
            grid[key] = cast("list[_GridValue]", list(spec["choices"]))
        elif spec["type"] == "int":
            step = int(spec.get("step", 1))
            grid[key] = list(range(int(spec["low"]), int(spec["high"]) + 1, step))
        else:
            raise ValueError(
                f"grid sampler needs an enumerable search_space (categorical, or stepped int) -- '{key}' is '{spec['type']}'"
            )
    return grid


def _resolve_sampler(
    name: str, search_space: dict[str, ParamSpec]
) -> optuna.samplers.BaseSampler:
    if name == "grid":
        return optuna.samplers.GridSampler(_grid_search_space(search_space))
    if name not in _SAMPLERS:
        raise ValueError(
            f"unknown sampler '{name}' (one of {', '.join((*_SAMPLERS, 'grid'))})"
        )
    return _SAMPLERS[name]()


def _resolve_trainer(accelerator: str) -> Trainer:
    if accelerator == "mlx":
        try:
            from oyster.trainers import TRAINERS
        except ImportError as e:
            raise ValueError(
                f"accelerator 'mlx' needs oyster installed (uv pip install -e '.[mlx]'): {e}"
            ) from e
        return TRAINERS["mlx"]
    if accelerator == "local":
        try:
            from physarum.trainers.trafilatura_extract import train as local_train
        except ImportError as e:
            raise ValueError(
                f"accelerator 'local' needs trafilatura installed (uv pip install -e '.[local]'): {e}"
            ) from e
        return local_train
    raise ValueError(
        f"no trainer resolvable for accelerator '{accelerator}' yet (one of 'local', 'mlx' is wired up)"
    )


def _flag_value(flags: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for f in flags:
        if f.startswith(prefix):
            return f[len(prefix) :]
    if name not in flags:
        return None
    i = flags.index(name)
    return flags[i + 1] if i + 1 < len(flags) else None


def _make_progress_callback(
    total: int,
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # study.best_value raises until at least one trial has completed.
        # `total` is the configured n_trials, not len(study.trials): the latter
        # equals trial.number + 1 in this single-threaded loop, so lines would
        # otherwise read "K/K". trial.number is 0-indexed, hence the +1.
        best = (
            study.best_value
            if study.trials and any(t.value is not None for t in study.trials)
            else None
        )
        if trial.state != optuna.trial.TrialState.COMPLETE:
            # objective() already recorded the real error on the Trial manifest
            # before study.optimize(catch=...) swallowed it; this line is only
            # the backend's own view.
            print(
                f"[WARN] trial {trial.number + 1}/{total} {trial.state.name.lower()} "
                f"(best so far: {best}) params={trial.params}",
                file=sys.stderr,
            )
            return
        print(
            f"[INFO] trial {trial.number + 1}/{total} done: value={trial.value} "
            f"(best so far: {best}) params={trial.params}",
            file=sys.stderr,
        )

    return callback


def sweep_optimize(cmd: Command) -> int:
    if not cmd.objects:
        print("[FAIL] mcm sweep optimize needs a sweep yaml path", file=sys.stderr)
        return 1
    sweep = Sweep.from_yaml(cmd.objects[0])
    sweep.validate()

    trainer_fn = _resolve_trainer(sweep.template["accelerator"])
    study = optuna.create_study(
        study_name=sweep.name,
        direction=sweep.objective["direction"],
        sampler=_resolve_sampler(sweep.sampler, sweep.search_space),
    )
    # `sweep watch` takes only a name, so it can't learn n_trials from the yaml;
    # this sidecar carries it. started_at lets watch hide trials from an earlier
    # run of a same-named sweep, since reishi's store never deletes old manifests
    # (see watch.trials_for_sweep's started_at filter).
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store.save(
        "sweeps",
        sweep.name,
        {"name": sweep.name, "n_trials": sweep.n_trials, "started_at": started_at},
    )
    print(
        f"[INFO] sweep '{sweep.name}' starting: {sweep.n_trials} trials -> "
        f"run `mcm sweep watch {sweep.name}` in another terminal to graph convergence live",
        file=sys.stderr,
    )
    study.optimize(
        make_objective(sweep, trainer_fn),
        n_trials=sweep.n_trials,
        callbacks=[_make_progress_callback(sweep.n_trials)],
        # objective() marks a failing trial "failed" and re-raises; catch stops
        # that exception unwinding study.optimize() and losing every later trial.
        catch=(Exception,),
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(
            f"[FAIL] sweep '{sweep.name}': all {len(study.trials)} trials failed",
            file=sys.stderr,
        )
        return 1

    best = study.best_trial
    best_trial_id = best.user_attrs.get("mcm_trial_id")
    failed = len(study.trials) - len(completed)
    suffix = f" ({failed} of {len(study.trials)} trials failed)" if failed else ""
    print(
        f"[OK] sweep '{sweep.name}' done: best value {best.value} (mcm trial {best_trial_id}){suffix}",
        file=sys.stderr,
    )
    emit(
        {
            "best_value": best.value,
            "best_trial": best_trial_id,
            "best_params": best.params,
        },
        cmd.flags,
    )
    return 0


def sweep_watch(cmd: Command) -> int:
    if not cmd.objects:
        print("[FAIL] mcm sweep watch needs a sweep name", file=sys.stderr)
        return 1
    port_flag = _flag_value(cmd.flags, "--port")
    try:
        port = int(port_flag) if port_flag is not None else DEFAULT_PORT
    except ValueError:
        print(f"[FAIL] --port must be an integer, got '{port_flag}'", file=sys.stderr)
        return 1
    watch_serve(cmd.objects[0], port=port)
    return 0


HANDLERS = {
    ("sweep", "optimize"): sweep_optimize,
    ("sweep", "watch"): sweep_watch,
}

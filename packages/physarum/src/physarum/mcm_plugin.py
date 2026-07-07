"""mcm plugin: the `sweep` domain. With physarum installed, the one mcm CLI
grows sweep vocabulary -- same grammar, same canonical echo, same -o json.

    mcm sweep optimize sweep.yaml   # > mcm sweep optimize   run a sweep to completion, in-process
    mcm sweep watch my-sweep        # > mcm sweep watch      localhost page graphing its trials live
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
    Verb("optimize", home="sweep", readonly=False),  # not "run" -- already home to "recipe"
    Verb("watch", home="sweep", readonly=True),
)

_SAMPLERS = {
    "tpe": optuna.samplers.TPESampler,
    "cmaes": optuna.samplers.CmaEsSampler,
    "random": optuna.samplers.RandomSampler,
}


_GridValue = str | float | int | bool | None


def _grid_search_space(search_space: dict[str, ParamSpec]) -> dict[str, list[_GridValue]]:
    # GridSampler needs every value it will ever suggest listed up front, so
    # unbounded continuous params ("float"/"loguniform") can't feed it --
    # only the enumerable types (categorical's own choices, or a stepped int
    # range) have a well-defined full grid.
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


def _resolve_sampler(name: str, search_space: dict[str, ParamSpec]) -> optuna.samplers.BaseSampler:
    if name == "grid":
        return optuna.samplers.GridSampler(_grid_search_space(search_space))
    if name not in _SAMPLERS:
        raise ValueError(f"unknown sampler '{name}' (one of {', '.join((*_SAMPLERS, 'grid'))})")
    return _SAMPLERS[name]()


def _resolve_trainer(accelerator: str) -> Trainer:
    # mlx delegates to oyster's cluster-execution stack; local needs nothing
    # beyond trafilatura since it never leaves this process.
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
    raise ValueError(f"no trainer resolvable for accelerator '{accelerator}' yet (one of 'local', 'mlx' is wired up)")


def _flag_value(flags: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for f in flags:
        if f.startswith(prefix):
            return f[len(prefix) :]
    if name not in flags:
        return None
    i = flags.index(name)
    return flags[i + 1] if i + 1 < len(flags) else None


def _make_progress_callback(total: int) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # study.best_value raises until at least one trial has completed -- trial.number
        # is 0-indexed, so +1 here matches the 1-of-N a human expects to read. `total` is
        # the sweep's configured n_trials, not len(study.trials) -- the latter only ever
        # equals trial.number + 1 in this single-threaded loop, so every line would
        # otherwise read "K/K" instead of "K/60".
        best = study.best_value if study.trials and any(t.value is not None for t in study.trials) else None
        if trial.state != optuna.trial.TrialState.COMPLETE:
            # study.optimize(catch=...) swallows the exception and keeps going --
            # objective() already recorded the real error on the Trial manifest,
            # this is just the search backend's own view of what happened.
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
    # A read-only sidecar for `sweep watch`, which only ever reads the store and
    # has no other way to learn n_trials -- it can't parse this sweep's yaml
    # (it only takes a name), and it must stay startable independently of this
    # process. Not a reishi primitive: it's physarum's own bookkeeping over
    # reishi's generic (kind, name) store, same as Sweep itself.
    #
    # started_at lets `sweep watch` hide trials left over from an earlier run
    # of a sweep with this same name (reishi's store has no delete -- old Trial
    # manifests are never removed, only superseded by this newer sidecar) --
    # see watch.trials_for_sweep's started_at filter.
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store.save("sweeps", sweep.name, {"name": sweep.name, "n_trials": sweep.n_trials, "started_at": started_at})
    print(
        f"[INFO] sweep '{sweep.name}' starting: {sweep.n_trials} trials -> "
        f"run `mcm sweep watch {sweep.name}` in another terminal to graph convergence live",
        file=sys.stderr,
    )
    study.optimize(
        make_objective(sweep, trainer_fn),
        n_trials=sweep.n_trials,
        callbacks=[_make_progress_callback(sweep.n_trials)],
        # One bad trial (a trainer crash, a missing metric) must not sink the
        # whole sweep -- objective() already marks it "failed" on the Trial
        # manifest before re-raising; this just stops Optuna from propagating
        # that past study.optimize() and losing every trial after it.
        catch=(Exception,),
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f"[FAIL] sweep '{sweep.name}': all {len(study.trials)} trials failed", file=sys.stderr)
        return 1

    best = study.best_trial
    best_trial_id = best.user_attrs.get("mcm_trial_id")
    failed = len(study.trials) - len(completed)
    suffix = f" ({failed} of {len(study.trials)} trials failed)" if failed else ""
    print(f"[OK] sweep '{sweep.name}' done: best value {best.value} (mcm trial {best_trial_id}){suffix}", file=sys.stderr)
    emit({"best_value": best.value, "best_trial": best_trial_id, "best_params": best.params}, cmd.flags)
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

import sys

from reishi import store
from reishi.cli.grammar import Command
from reishi.cli.output import emit
from reishi.primitives import board, dataset, task, trial
from reishi.primitives.recipe import Recipe


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}", file=sys.stderr)
    return 1


def _need_object(cmd: Command, what: str) -> str | None:
    if not cmd.objects:
        print(f"[FAIL] mcm {cmd.domain} {cmd.action} needs {what}", file=sys.stderr)
        return None
    return cmd.objects[0]


def status(cmd: Command) -> int:
    emit(
        {
            "store": str(store.root()),
            "tasks": [t.name for t in task.all_tasks()],
            "datasets": len(dataset.load_all()),
            "trials": len(trial.load_all()),
        },
        cmd.flags,
    )
    return 0


def task_list(cmd: Command) -> int:
    emit(
        [t.to_manifest() for t in task.all_tasks()],
        cmd.flags,
        columns=["name", "output_fields", "codec", "scorer"],
    )
    return 0


def dataset_list(cmd: Command) -> int:
    emit(
        [d.to_manifest() for d in dataset.load_all()],
        cmd.flags,
        columns=["name", "task", "uri", "eval_only"],
    )
    return 0


def dataset_describe(cmd: Command) -> int:
    name = _need_object(cmd, "a dataset name")
    if name is None:
        return 1
    emit(dataset.load(name).to_manifest(), cmd.flags)
    return 0


def recipe_run(cmd: Command) -> int:
    path = _need_object(cmd, "a recipe yaml path")
    if path is None:
        return 1
    recipe = Recipe.from_yaml(path)
    trials = trial.plan(recipe)

    if "--plan" in cmd.flags:
        for t in trials:
            trial.save(t)
        print(
            f"[OK] planned {len(trials)} trial(s) for recipe '{recipe.name}'",
            file=sys.stderr,
        )
        emit(
            [{"id": t.id, "seed": t.seed, "status": t.status} for t in trials],
            cmd.flags,
        )
        return 0

    return _fail(
        f"no trainer installed for accelerator '{recipe.accelerator}' yet "
        f"-> use --plan to record the {len(trials)} trial(s) without executing"
    )


def trial_list(cmd: Command) -> int:
    rows = [
        {
            "id": t.id,
            "recipe": t.recipe,
            "seed": t.seed,
            "status": t.status,
            "created": t.created,
            "metrics": t.metrics or None,
        }
        for t in trial.load_all()
    ]
    emit(
        rows,
        cmd.flags,
        columns=["id", "recipe", "seed", "status", "created", "metrics"],
    )
    return 0


def trial_describe(cmd: Command) -> int:
    ref = _need_object(cmd, "a trial id (prefix ok)")
    if ref is None:
        return 1
    emit(trial.resolve(ref).to_manifest(), cmd.flags)
    return 0


def trial_logs(cmd: Command) -> int:
    ref = _need_object(cmd, "a trial id (prefix ok)")
    if ref is None:
        return 1
    t = trial.resolve(ref)
    return _fail(
        f"trial '{t.id}' has no logs yet (status: {t.status}; log streaming lands with the trainer)"
    )


def board_show(cmd: Command) -> int:
    metric = "f1"
    if "--metric" in cmd.flags:
        i = cmd.flags.index("--metric")
        if i + 1 < len(cmd.flags):
            metric = cmd.flags[i + 1]
    rows = board.build(metric=metric, task=cmd.objects[0] if cmd.objects else None)
    emit(rows, cmd.flags)
    return 0


# `experiment submit` is canonical vocabulary (see grammar.py) but reishi ships
# no handler for it: submitting a recipe to real hardware is an executor's job,
# contributed via an `mcm.plugins` entry point (enoki -> RayJob). Absent any
# executor, dispatch falls through to a clean "not implemented".
HANDLERS = {
    ("task", "list"): task_list,
    ("dataset", "list"): dataset_list,
    ("dataset", "describe"): dataset_describe,
    ("recipe", "run"): recipe_run,
    ("trial", "list"): trial_list,
    ("trial", "describe"): trial_describe,
    ("trial", "logs"): trial_logs,
    ("board", "show"): board_show,
    ("board", "list"): board_show,
}

import base64
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

import reishi.tasks  # noqa: F401  (populate the task registry)
from reishi import store
from reishi.cli.grammar import Command
from reishi.cli.output import emit
from reishi.primitives import board, dataset, task, trial
from reishi.primitives.recipe import Recipe

# Only accelerators with a verified-correct node selector/toleration entry
# here are supported; anything else fails cleanly rather than guessing at
# values the scheduler would silently mis-place on.
_GPU_NODE_SELECTORS = {
    "l4": {"cloud.google.com/compute-class": "gpu-l4", "kubernetes.io/arch": "amd64"},
}
_GPU_TOLERATIONS = {
    "l4": [
        {"key": "cloud.google.com/compute-class", "operator": "Equal", "value": "gpu-l4", "effect": "NoSchedule"},
        {"key": "nvidia.com/gpu", "operator": "Equal", "value": "present", "effect": "NoSchedule"},
        {"key": "cloud.google.com/gke-spot", "operator": "Equal", "value": "true", "effect": "NoSchedule"},
    ],
}

# placeholder default image -- override with --image or the MCM_TRAIN_IMAGE env var
# named in enoki's Dockerfile. Override with --image or MCM_TRAIN_IMAGE once
# a real build/push pipeline picks its own tags.
_DEFAULT_IMAGE = "localhost/enoki-train:latest"


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
    emit([t.to_manifest() for t in task.all_tasks()], cmd.flags,
         columns=["name", "output_fields", "codec", "scorer"])
    return 0


def dataset_list(cmd: Command) -> int:
    emit([d.to_manifest() for d in dataset.load_all()], cmd.flags,
         columns=["name", "task", "uri", "eval_only"])
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
        print(f"[OK] planned {len(trials)} trial(s) for recipe '{recipe.name}'", file=sys.stderr)
        emit([{"id": t.id, "seed": t.seed, "status": t.status} for t in trials], cmd.flags)
        return 0

    return _fail(
        f"no trainer installed for accelerator '{recipe.accelerator}' yet "
        f"-> use --plan to record the {len(trials)} trial(s) without executing"
    )


def _flag_value(flags: list[str], name: str) -> str | None:
    if name not in flags:
        return None
    i = flags.index(name)
    return flags[i + 1] if i + 1 < len(flags) else None


def _enoki_root() -> Path:
    # mcm-reishi and mcm-enoki are checked out as sibling repos (both
    # scoped under ml/); this file lives at <ml>/mcm-reishi/src/reishi/cli/.
    # ENOKI_HOME overrides for any other layout.
    env = os.environ.get("ENOKI_HOME")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[4] / "mcm-enoki"


def experiment_submit(cmd: Command) -> int:
    name = _need_object(cmd, "an experiment name")
    if name is None:
        return 1

    recipe_path = Path("experiments") / name / "recipe.yaml"
    if not recipe_path.exists():
        return _fail(f"no recipe at {recipe_path} (convention: experiments/<name>/recipe.yaml)")
    recipe = Recipe.from_yaml(recipe_path)
    recipe.validate()

    if recipe.accelerator not in _GPU_NODE_SELECTORS:
        return _fail(
            f"'mcm experiment submit' only supports accelerator 'l4' for now "
            f"(recipe '{name}' wants '{recipe.accelerator}') -- its node selector/tolerations "
            f"aren't verified against a live workload yet, so this fails instead of guessing"
        )

    template_path = _enoki_root() / "jobs" / "rayjob.yaml"
    if not template_path.exists():
        return _fail(
            f"no RayJob template at {template_path} -- set ENOKI_HOME if mcm-enoki "
            "isn't checked out as a sibling of mcm-reishi"
        )

    image = _flag_value(cmd.flags, "--image") or os.environ.get("MCM_TRAIN_IMAGE") or _DEFAULT_IMAGE
    node_selector = yaml.safe_dump(
        _GPU_NODE_SELECTORS[recipe.accelerator], default_flow_style=True, width=10**9
    ).strip()
    tolerations = yaml.safe_dump(
        _GPU_TOLERATIONS[recipe.accelerator], default_flow_style=True, width=10**9
    ).strip()

    rendered = template_path.read_text()
    substitutions = {
        "{{name}}": name,
        "{{image}}": image,
        # /recipes/ is the path the entrypoint writes the recipe to inside
        # the container (nothing mounts it -- see rayjob.yaml); nested under
        # <name>/ to mirror the experiments/<name>/recipe.yaml convention
        # and avoid collisions between experiments sharing a filename.
        "{{recipe}}": f"{name}/recipe.yaml",
        # The image has no experiments/ directory baked in, so the
        # entrypoint recreates the recipe file itself from this inline blob.
        "{{recipe_b64}}": base64.b64encode(recipe_path.read_bytes()).decode("ascii"),
        "{{accelerator}}": recipe.accelerator,
        "{{gpu_node_selector}}": node_selector,
        "{{gpu_tolerations}}": tolerations,
    }
    for placeholder, value in substitutions.items():
        rendered = rendered.replace(placeholder, value)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix=f"mcm-{name}-", delete=False) as f:
        f.write(rendered)
        manifest_path = f.name

    try:
        result = subprocess.run(["kubectl", "apply", "-f", manifest_path], capture_output=True, text=True)
        if result.returncode != 0:
            return _fail(f"kubectl apply failed: {result.stderr.strip()}")
    finally:
        os.unlink(manifest_path)

    print(f"[OK] submitted RayJob 'mcm-{name}' -> {result.stdout.strip()}", file=sys.stderr)
    return 0


def trial_list(cmd: Command) -> int:
    rows = [
        {"id": t.id, "recipe": t.recipe, "seed": t.seed, "status": t.status,
         "created": t.created, "metrics": t.metrics or None}
        for t in trial.load_all()
    ]
    emit(rows, cmd.flags, columns=["id", "recipe", "seed", "status", "created", "metrics"])
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
    return _fail(f"trial '{t.id}' has no logs yet (status: {t.status}; log streaming lands with the trainer)")


def board_show(cmd: Command) -> int:
    metric = "f1"
    if "--metric" in cmd.flags:
        i = cmd.flags.index("--metric")
        if i + 1 < len(cmd.flags):
            metric = cmd.flags[i + 1]
    rows = board.build(metric=metric, task=cmd.objects[0] if cmd.objects else None)
    emit(rows, cmd.flags)
    return 0


HANDLERS = {
    ("task", "list"): task_list,
    ("dataset", "list"): dataset_list,
    ("dataset", "describe"): dataset_describe,
    ("recipe", "run"): recipe_run,
    ("experiment", "submit"): experiment_submit,
    ("trial", "list"): trial_list,
    ("trial", "describe"): trial_describe,
    ("trial", "logs"): trial_logs,
    ("board", "show"): board_show,
    ("board", "list"): board_show,
}

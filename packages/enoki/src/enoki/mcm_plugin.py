"""mcm plugin: enoki implements `experiment submit` -- render a recipe into a
KubeRay RayJob and kubectl apply it. With enoki installed, the one mcm CLI can
dispatch a recipe to the cloud cluster; without it, `mcm experiment submit`
degrades to a clean "not implemented".

The `experiment`/`submit` vocabulary is reishi's (canonical grammar); enoki only
contributes the HANDLER, so it declares no new DOMAINS/VERBS. Execution knowledge
-- the node selectors, the training image, the RayJob template, kubectl -- lives
here in the executor, never in reishi.
"""

import base64
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from reishi.cli.grammar import Command
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


def _flag_value(flags: list[str], name: str) -> str | None:
    if name not in flags:
        return None
    i = flags.index(name)
    return flags[i + 1] if i + 1 < len(flags) else None


def _enoki_root() -> Path:
    # jobs/rayjob.yaml ships at the enoki package root; this file lives at
    # <root>/src/enoki/mcm_plugin.py. ENOKI_HOME overrides for the training
    # image or any layout where enoki isn't installed from source.
    env = os.environ.get("ENOKI_HOME")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


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
            f"no RayJob template at {template_path} -- set ENOKI_HOME if enoki "
            "isn't installed from source"
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


HANDLERS = {
    ("experiment", "submit"): experiment_submit,
}

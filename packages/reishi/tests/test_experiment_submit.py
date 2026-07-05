"""experiment_submit: recipe -> rendered RayJob -> kubectl apply.

kubectl is always monkeypatched here -- these tests never touch a real
cluster. jobs/rayjob.yaml is read from the real sibling mcm-enoki checkout
(the same lookup experiment_submit uses at runtime), not faked, so a change
to the template's placeholders is caught here too.
"""

import base64
import glob
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import reishi.tasks  # noqa: F401  (populate the task registry)
from reishi.cli import commands
from reishi.cli.grammar import Command

RECIPE = """
name: htmlmd-smoke
task: htmlmd
dataset: htmlmd-fixture
base_model: jinaai/ReaderLM-v2
accelerator: {accelerator}
seeds: 1
trainer:
  method: lora
  rank: 16
"""


def _write_recipe(tmp_path, name: str, accelerator: str = "l4"):
    d = tmp_path / "experiments" / name
    d.mkdir(parents=True)
    (d / "recipe.yaml").write_text(RECIPE.format(accelerator=accelerator))


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _fake_run_ok(calls):
    # experiment_submit deletes the rendered manifest file once kubectl apply
    # returns, so snapshot its content here, at call time, rather than
    # re-reading the (by-then-gone) path afterwards.
    def _run(args, capture_output, text):
        manifest_path = args[args.index("-f") + 1]
        manifest = yaml.safe_load(open(manifest_path))
        calls.append({"args": args, "manifest": manifest})
        return SimpleNamespace(returncode=0, stdout="rayjob.ray.io/mcm-htmlmd-smoke created", stderr="")

    return _run


def test_submit_renders_every_placeholder_for_l4_recipe(tmp_path, monkeypatch, capsys):
    _write_recipe(tmp_path, "htmlmd-smoke", accelerator="l4")
    calls = []
    monkeypatch.setattr(commands.subprocess, "run", _fake_run_ok(calls))

    rc = commands.experiment_submit(Command(domain="experiment", action="submit", objects=["htmlmd-smoke"]))

    assert rc == 0
    assert len(calls) == 1
    manifest = calls[0]["manifest"]

    assert manifest["metadata"]["name"] == "mcm-htmlmd-smoke"
    # No volume/ConfigMap mounts a recipe file into the image, so the
    # entrypoint recreates it from an inline base64 blob before running the
    # driver -- assert both the shape and that it decodes to the real recipe.
    entrypoint = manifest["spec"]["entrypoint"]
    assert entrypoint.startswith("mkdir -p /recipes/htmlmd-smoke && echo ")
    assert entrypoint.endswith(
        " | base64 -d > /recipes/htmlmd-smoke/recipe.yaml && "
        "python -m enoki.driver /recipes/htmlmd-smoke/recipe.yaml"
    )
    recipe_b64 = entrypoint.split("echo ", 1)[1].split(" | base64", 1)[0]
    assert base64.b64decode(recipe_b64).decode() == RECIPE.format(accelerator="l4")

    head_image = manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0]["image"]
    assert head_image == commands._DEFAULT_IMAGE

    worker = manifest["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]
    assert worker["groupName"] == "l4"
    worker_spec = worker["template"]["spec"]
    assert worker_spec["nodeSelector"] == commands._GPU_NODE_SELECTORS["l4"]
    assert worker_spec["tolerations"] == commands._GPU_TOLERATIONS["l4"]
    # yaml.safe_dump must round-trip 'true' as a string, not a YAML bool --
    # a real toleration requires a string value here.
    spot_toleration = next(t for t in worker_spec["tolerations"] if t["key"] == "cloud.google.com/gke-spot")
    assert spot_toleration["value"] == "true"
    assert isinstance(spot_toleration["value"], str)

    assert "[OK] submitted RayJob 'mcm-htmlmd-smoke'" in capsys.readouterr().err


def test_submit_honors_image_override_flag(tmp_path, monkeypatch):
    _write_recipe(tmp_path, "htmlmd-smoke", accelerator="l4")
    calls = []
    monkeypatch.setattr(commands.subprocess, "run", _fake_run_ok(calls))

    rc = commands.experiment_submit(
        Command(domain="experiment", action="submit", objects=["htmlmd-smoke"],
                flags=["--image", "example.com/custom:tag"])
    )

    assert rc == 0
    manifest = calls[0]["manifest"]
    head_image = manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0]["image"]
    assert head_image == "example.com/custom:tag"


def test_submit_rejects_unsupported_accelerator_without_calling_kubectl(tmp_path, monkeypatch, capsys):
    _write_recipe(tmp_path, "h100-run", accelerator="h100")
    calls = []
    monkeypatch.setattr(commands.subprocess, "run", _fake_run_ok(calls))

    rc = commands.experiment_submit(Command(domain="experiment", action="submit", objects=["h100-run"]))

    assert rc == 1
    assert calls == []
    assert "only supports accelerator 'l4'" in capsys.readouterr().err


def test_submit_fails_cleanly_when_recipe_missing(monkeypatch):
    calls = []
    monkeypatch.setattr(commands.subprocess, "run", _fake_run_ok(calls))

    rc = commands.experiment_submit(Command(domain="experiment", action="submit", objects=["does-not-exist"]))

    assert rc == 1
    assert calls == []


def test_submit_reports_kubectl_failure(tmp_path, monkeypatch, capsys):
    _write_recipe(tmp_path, "htmlmd-smoke", accelerator="l4")

    def _run_fail(args, capture_output, text):
        return SimpleNamespace(returncode=1, stdout="", stderr="the server doesn't have a resource type")

    monkeypatch.setattr(commands.subprocess, "run", _run_fail)

    rc = commands.experiment_submit(Command(domain="experiment", action="submit", objects=["htmlmd-smoke"]))

    assert rc == 1
    assert "kubectl apply failed" in capsys.readouterr().err


def _rendered_manifest_leftovers() -> list[str]:
    return glob.glob(str(Path(tempfile.gettempdir()) / "mcm-htmlmd-smoke-*.yaml"))


def test_submit_cleans_up_the_rendered_manifest_tempfile_on_success(tmp_path, monkeypatch):
    _write_recipe(tmp_path, "htmlmd-smoke", accelerator="l4")
    monkeypatch.setattr(commands.subprocess, "run", _fake_run_ok([]))

    before = _rendered_manifest_leftovers()
    rc = commands.experiment_submit(Command(domain="experiment", action="submit", objects=["htmlmd-smoke"]))
    after = _rendered_manifest_leftovers()

    assert rc == 0
    assert after == before


def test_submit_cleans_up_the_rendered_manifest_tempfile_on_kubectl_failure(tmp_path, monkeypatch):
    _write_recipe(tmp_path, "htmlmd-smoke", accelerator="l4")

    def _run_fail(args, capture_output, text):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(commands.subprocess, "run", _run_fail)

    before = _rendered_manifest_leftovers()
    rc = commands.experiment_submit(Command(domain="experiment", action="submit", objects=["htmlmd-smoke"]))
    after = _rendered_manifest_leftovers()

    assert rc == 1
    assert after == before


def test_submit_uses_the_real_rayjob_template():
    """The real sibling-repo lookup (no ENOKI_HOME override) must find
    mcm-enoki's actual jobs/rayjob.yaml, not a path that only exists in test
    fixtures -- otherwise this suite could pass against a template that
    doesn't match what ships."""
    assert (commands._enoki_root() / "jobs" / "rayjob.yaml").exists()

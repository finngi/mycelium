"""CLI wiring for the local executor: recipe_run resolves a producer via the
registry before executing, --plan stays untouched, trial_logs reads back
captured output, and status reports installed producers."""

import json

import pytest

from reishi import store
from reishi.cli import commands
from reishi.cli.grammar import Command
from reishi.execution import registry
from reishi.primitives import trial as trial_store


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    store.use_backend(store.LocalFilesystemBackend())
    yield
    store.use_backend(store.LocalFilesystemBackend())


@pytest.fixture(autouse=True)
def isolated_registry():
    registry._producers = None
    yield
    registry._producers = None


def _recipe_yaml(tmp_path, runtime="cpu"):
    path = tmp_path / "recipe.yaml"
    path.write_text(
        f"name: demo\ntask: fixture\ntrain_dataset: some-ds\nruntime: {runtime}\n"
    )
    return path


def test_recipe_run_without_installed_producer_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "entry_points", lambda group: [])
    path = _recipe_yaml(tmp_path)
    cmd = Command(domain="recipe", action="run", objects=[str(path)])
    assert commands.recipe_run(cmd) == 1


def test_recipe_run_resolves_registered_producer_and_executes(tmp_path, monkeypatch):
    def fake_producer(manifest):
        return {"metrics": {"f1": 1.0}, "artifacts": {}}

    monkeypatch.setattr(registry, "get", lambda runtime: fake_producer)
    path = _recipe_yaml(tmp_path)
    cmd = Command(domain="recipe", action="run", objects=[str(path)])

    assert commands.recipe_run(cmd) == 0
    trials = trial_store.load_all()
    assert len(trials) == 1
    assert trials[0].status == "done"
    assert trials[0].metrics == {"f1": 1.0}


def test_recipe_run_plan_flag_is_unchanged(tmp_path):
    path = _recipe_yaml(tmp_path)
    cmd = Command(domain="recipe", action="run", objects=[str(path)], flags=["--plan"])

    assert commands.recipe_run(cmd) == 0
    trials = trial_store.load_all()
    assert len(trials) == 1
    assert trials[0].status == "planned"


def test_trial_logs_prints_captured_content(tmp_path, capsys):
    log_path = tmp_path / "t.log"
    log_path.write_text("hello from the log\n")
    t = trial_store.Trial(
        id="t-s0-aaa111", recipe_name="r", seed=0, execution={"log": str(log_path)}
    )
    trial_store.save(t)

    cmd = Command(domain="trial", action="logs", objects=["t-s0-aaa111"])
    assert commands.trial_logs(cmd) == 0
    assert "hello from the log" in capsys.readouterr().out


def test_trial_logs_without_a_log_keeps_the_old_fail_message():
    t = trial_store.Trial(id="t-s1-bbb222", recipe_name="r", seed=0)
    trial_store.save(t)

    cmd = Command(domain="trial", action="logs", objects=["t-s1-bbb222"])
    assert commands.trial_logs(cmd) == 1


def test_status_includes_sorted_producers_key(monkeypatch, capsys):
    monkeypatch.setattr(registry, "supported", lambda: {"mlx", "cpu"})
    cmd = Command(domain=None, action=None, flags=["-o", "json"])

    assert commands.status(cmd) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["producers"] == ["cpu", "mlx"]

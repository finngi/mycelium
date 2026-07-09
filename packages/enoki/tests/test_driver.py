"""driver.run(): recipe -> trials -> trainer, with status/metrics/failure handling.

Trainers are monkeypatched fakes registered in enoki.trainers.TRAINERS -- these
tests never need a GPU, a cluster, or the real ReaderLM-v2 weights.
"""

import pytest

from reishi import store
from reishi.primitives import trial

from enoki import driver
from enoki.trainers import TRAINERS

RECIPE = """
name: htmlmd-smoke
task: fixture
dataset: htmlmd-fixture
accelerator: l4
seeds: 2
"""


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path / "store"))
    store.use_backend(store.LocalFilesystemBackend())
    yield
    store.use_backend(store.LocalFilesystemBackend())


def _write_recipe(tmp_path, body: str = RECIPE):
    p = tmp_path / "recipe.yaml"
    p.write_text(body)
    return p


def test_run_success_marks_trials_done_with_metrics_and_artifacts(
    tmp_path, monkeypatch
):
    def fake_trainer(manifest):
        return {
            "metrics": {"train_loss": 0.5},
            "artifacts": {"weights": "/tmp/adapter"},
        }

    monkeypatch.setitem(TRAINERS, "l4", fake_trainer)

    rc = driver.run(str(_write_recipe(tmp_path)))

    assert rc == 0
    trials = trial.load_all()
    assert len(trials) == 2
    assert all(t.status == "done" for t in trials)
    assert all(t.metrics == {"train_loss": 0.5} for t in trials)
    assert all(t.artifacts == {"weights": "/tmp/adapter"} for t in trials)


def test_run_failure_marks_trial_failed_and_records_last_error(tmp_path, monkeypatch):
    def fake_trainer(manifest):
        raise RuntimeError("boom")

    monkeypatch.setitem(TRAINERS, "l4", fake_trainer)

    rc = driver.run(str(_write_recipe(tmp_path)))

    assert rc == 1
    trials = trial.load_all()
    assert len(trials) == 2
    assert all(t.status == "failed" for t in trials)
    assert all(t.execution["last_error"] == "boom" for t in trials)


def test_run_unknown_accelerator_fails_before_planning_any_trial(tmp_path):
    recipe_path = _write_recipe(
        tmp_path, RECIPE.replace("accelerator: l4", "accelerator: h100")
    )

    rc = driver.run(str(recipe_path))

    assert rc == 1
    assert trial.load_all() == []

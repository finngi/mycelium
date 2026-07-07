"""Contract tests for the scheduler: priority, fit, claim, retry, drain."""

from datetime import datetime, timedelta, timezone

import pytest

from reishi.primitives import trial as trial_store
from reishi.primitives.recipe import Recipe

from oyster import machine, queue, worker
from oyster.trainers import TRAINERS


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("OYSTER_MEM_BUDGET_GB", "40")
    monkeypatch.setattr(machine, "BUSY_FILE", tmp_path / "busy")
    monkeypatch.setattr(machine, "training_process_running", lambda: False)


def plan_one(name: str, priority: int = 0, base_model: str = "x/tiny-0.5B",
             accelerator: str = "mlx") -> trial_store.Trial:
    r = Recipe(name=name, task="extract", dataset="d-1", base_model=base_model,
               accelerator=accelerator, priority=priority)
    t = trial_store.plan(r)[0]
    trial_store.save(t)
    return t


def test_priority_desc_then_fifo():
    plan_one("low-1")
    plan_one("hot", priority=5)
    plan_one("low-2")
    order = [t.recipe for t in queue.eligible(40, {"mlx"})]
    assert order == ["hot", "low-1", "low-2"]


def test_footprint_gate_leaves_big_models():
    plan_one("small")
    plan_one("huge", base_model="x/chonk-9B")  # est 9000M*2B*3 = 54GB > 40GB
    names = [t.recipe for t in queue.eligible(40, {"mlx"})]
    assert names == ["small"]


def test_accelerator_gate():
    plan_one("cuda-only", accelerator="l4")
    assert queue.eligible(40, {"mlx"}) == []


def test_claim_takes_ownership_once():
    t = plan_one("job")
    claimed = queue.claim(t, "mac-a")
    assert claimed is not None and claimed.status == "running"
    assert claimed.execution["runner"] == "mac-a"
    assert claimed.execution["attempt"] == 1
    assert queue.claim(t, "mac-b") is None  # already running


def test_fail_requeues_until_attempts_exhausted():
    t = plan_one("flaky")
    for attempt in range(1, queue.MAX_ATTEMPTS + 1):
        assert queue.claim(trial_store.load(t.id), "mac-a") is not None
        failed = trial_store.load(t.id)
        queue.fail(failed, "boom")
        expected = "failed" if attempt == queue.MAX_ATTEMPTS else "planned"
        assert trial_store.load(t.id).status == expected


def test_requeue_stale_reaps_dead_runners():
    fresh, stale, exhausted = plan_one("fresh"), plan_one("stale"), plan_one("gone")
    now = datetime.now(timezone.utc)
    old = (now - timedelta(hours=3)).isoformat(timespec="seconds")
    for t, hb, attempt in ((fresh, now.isoformat(timespec="seconds"), 1),
                           (stale, old, 1), (exhausted, old, queue.MAX_ATTEMPTS)):
        t.status = "running"
        t.execution = {"runner": "mac-x", "heartbeat": hb, "attempt": attempt}
        trial_store.save(t)

    queue.requeue_stale(timeout_min=90)
    assert trial_store.load(fresh.id).status == "running"
    assert trial_store.load(stale.id).status == "planned"
    assert trial_store.load(exhausted.id).status == "failed"


def test_worker_drain_claims_nothing():
    plan_one("job")
    machine.BUSY_FILE.touch()
    worker.run()
    assert trial_store.load_all()[0].status == "planned"


def test_worker_runs_queue_with_fake_trainer(monkeypatch):
    ran = []

    def fake_trainer(manifest):
        ran.append(manifest["id"])
        return {"metrics": {"f1": 0.9}, "artifacts": {"weights": "hf://x"}}

    monkeypatch.setitem(TRAINERS, "mlx", fake_trainer)
    plan_one("a", priority=1)
    plan_one("b")
    worker.run()
    assert len(ran) == 2
    statuses = {t.recipe: t.status for t in trial_store.load_all()}
    assert statuses == {"a": "done", "b": "done"}
    done = trial_store.load(ran[0])
    assert done.metrics["f1"] == 0.9 and done.execution["runner"] == machine.name()


def test_worker_without_trainers_exits_clean(monkeypatch):
    for name in list(TRAINERS):
        monkeypatch.delitem(TRAINERS, name, raising=False)
    plan_one("job")
    assert worker.run() == 0
    assert trial_store.load_all()[0].status == "planned"

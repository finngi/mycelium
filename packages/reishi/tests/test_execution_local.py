"""local executor: planned -> running -> done/failed transitions (with the
save-before-run ordering actually visible to a reloading producer), the
executor-clock-wins wall_time_s contract, log capture, and batch semantics."""

from pathlib import Path

import pytest

from reishi import store
from reishi.execution import local
from reishi.primitives.trial import Trial, load


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    store.use_backend(store.LocalFilesystemBackend())
    yield
    store.use_backend(store.LocalFilesystemBackend())


def _trial(trial_id: str) -> Trial:
    return Trial(id=trial_id, recipe_name="r", seed=0, spec={"task": "fixture"})


def test_running_status_is_saved_before_the_producer_is_called():
    seen = {}

    def producer(manifest):
        # Reloading from the store (not just reading the in-memory Trial)
        # proves the running/claimed_at/attempt save happened before this call.
        reloaded = load(manifest["id"])
        seen["status"] = reloaded.status
        seen["attempt"] = reloaded.execution.get("attempt")
        seen["runner"] = reloaded.execution.get("runner")
        return {"metrics": {}, "artifacts": {}}

    t = _trial("t-s0-aaa111")
    local.execute([t], producer)
    assert seen["status"] == "running"
    assert seen["attempt"] == 1
    assert seen["runner"].startswith("local:")


def test_successful_trial_merges_fields_and_lands_done():
    def producer(manifest):
        return {
            "metrics": {"f1": 0.9},
            "artifacts": {"weights": "hf://x"},
            "observables": {"tokens": 100, "wall_time_s": 999},
        }

    t = _trial("t-s0-aaa111")
    code = local.execute([t], producer)

    assert code == 0
    assert t.status == "done"
    assert t.metrics == {"f1": 0.9}
    assert t.artifacts == {"weights": "hf://x"}
    assert t.observables["tokens"] == 100
    # The executor's own wall clock always wins over whatever the producer
    # itself reported for wall_time_s.
    assert t.observables["wall_time_s"] != 999
    assert isinstance(t.observables["wall_time_s"], float)
    assert "finished_at" in t.execution
    assert t.execution["log"]

    reloaded = load(t.id)
    assert reloaded.status == "done"
    assert reloaded.metrics == {"f1": 0.9}


def test_failing_producer_marks_failed_and_batch_continues_with_exit_1():
    def bad(manifest):
        raise RuntimeError("boom")

    t1, t2 = _trial("t-s0-aaa111"), _trial("t-s1-bbb222")
    code = local.execute([t1, t2], bad)

    assert code == 1
    assert t1.status == "failed" and t2.status == "failed"
    assert t1.execution["last_error"] == "RuntimeError: boom"
    assert "finished_at" in t1.execution
    assert load(t1.id).status == "failed"
    assert load(t2.id).status == "failed"


def test_log_file_captures_producer_stdout_and_traceback_on_failure():
    def bad(manifest):
        print("hello from producer")
        raise RuntimeError("boom")

    t = _trial("t-s0-aaa111")
    local.execute([t], bad)

    log_path = Path(t.execution["log"])
    assert log_path.exists()
    content = log_path.read_text()
    assert "hello from producer" in content
    assert "RuntimeError: boom" in content
    assert "Traceback" in content


def test_producer_output_is_still_teed_to_the_real_stdout(capsys):
    def producer(manifest):
        print("visible on terminal too")
        return {"metrics": {}, "artifacts": {}}

    t = _trial("t-s0-aaa111")
    local.execute([t], producer)
    assert "visible on terminal too" in capsys.readouterr().out


def test_progress_messages_use_ascii_markers(capsys):
    t_ok, t_fail = _trial("t-s0-aaa111"), _trial("t-s1-bbb222")

    def producer(manifest):
        if manifest["id"] == t_fail.id:
            raise RuntimeError("boom")
        return {"metrics": {}, "artifacts": {}}

    local.execute([t_ok, t_fail], producer)
    err = capsys.readouterr().err
    assert f"[OK] {t_ok.id} done (" in err
    assert f"[FAIL] {t_fail.id} failed: RuntimeError: boom ->" in err


class _NoRootBackend:
    """Duck-typed remote backend: save/load only, no filesystem root()."""

    def __init__(self) -> None:
        self.docs: dict[tuple[str, str], dict] = {}

    def save(self, kind: str, name: str, doc: dict) -> None:
        self.docs[(kind, name)] = doc

    def load(self, kind: str, name: str) -> dict:
        return self.docs[(kind, name)]

    def load_all(self, kind: str) -> list[dict]:
        return [d for (k, _), d in self.docs.items() if k == kind]


def test_remote_backend_degrades_log_capture_with_a_warning(
    capsys, tmp_path, monkeypatch
):
    # A backend without root() must not silently mkdir a literal "<remote>"
    # dir; the run proceeds, warns once, and leaves execution.log unset so
    # trial_logs stays honest.
    monkeypatch.chdir(tmp_path)
    backend = _NoRootBackend()
    store.use_backend(backend)

    t = _trial("t-s0-remote1")
    rc = local.execute([t], lambda m: {"metrics": {"f1": 1.0}, "artifacts": {}})

    assert rc == 0
    saved = backend.docs[("trials", "t-s0-remote1")]
    assert saved["status"] == "done"
    assert "log" not in saved["execution"]
    err = capsys.readouterr().err
    assert err.count("[WARN] active store backend has no filesystem root") == 1
    assert not (tmp_path / "<remote>").exists()


def test_retried_trial_log_carries_an_attempt_separator_per_run():
    t = _trial("t-s0-retry01")
    producer = lambda m: {"metrics": {}, "artifacts": {}}  # noqa: E731
    local.execute([t], producer)
    local.execute([load(t.id)], producer)

    content = Path(load(t.id).execution["log"]).read_text()
    assert "--- attempt 1 @ " in content
    assert "--- attempt 2 @ " in content


def test_malformed_producer_result_fails_the_trial_not_the_batch():
    # Entry-point producers have no schema validation; a result missing the
    # required keys must land as a failed trial while the batch continues.
    trials = [_trial("t-s0-bad0001"), _trial("t-s0-good001")]

    def producer(manifest):
        if manifest["id"].endswith("bad0001"):
            return {"wrong": True}
        return {"metrics": {"f1": 1.0}, "artifacts": {}}

    rc = local.execute(trials, producer)

    assert rc == 1
    assert load("t-s0-bad0001").status == "failed"
    assert "ProducerResult" in load("t-s0-bad0001").execution["last_error"]
    assert load("t-s0-good001").status == "done"

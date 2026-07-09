"""A registered fixture task for the trainer contract tests.

The trafilatura trainer resolves `spec["task"]` from reishi's registry and calls
its scorer. The trainer is task-agnostic, so its tests don't need any real
task's scorer -- htmlmd's real scorer lives in the mcm-lab workspace now, not in
any service package. This registers a trivial word-overlap scorer so the trainer
tests exercise the extract -> score -> aggregate contract without coupling
physarum to a specific experiment's task.
"""

import pytest

from reishi import store
from reishi.primitives.task import ScoreCounts, Task, register


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    # The default sqlite backend caches its connection in store._backend, so a
    # per-test MCM_STORE change is otherwise ignored after the first store use
    # and trials leak between tests. Point MCM_STORE at a fresh dir and clear
    # the cached backend so each test gets its own store.
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    monkeypatch.setattr(store, "_backend", None)


def _fixture_score(pred: dict, ref: dict) -> ScoreCounts:
    cand = set((pred.get("markdown") or "").split())
    truth = set((ref.get("markdown") or "").split())
    return {
        "tp": len(cand & truth),
        "fp": len(cand - truth),
        "fn": len(truth - cand),
        "exact_match": (pred.get("markdown") or "") == (ref.get("markdown") or ""),
        "invalid": not (pred.get("markdown") or "").strip(),
    }


register(
    Task(
        name="extract-fixture",
        description="trainer-contract test fixture; not a real task",
        output_fields=("markdown",),
        score=_fixture_score,
        codec="text",
    )
)

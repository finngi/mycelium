from typing import Any, Mapping

import pytest

from reishi.primitives.scoring import _render, rescore, run_eval


class FakeTask:
    """A Scorable that records score() calls; bounds nothing to dict shapes."""

    def __init__(self) -> None:
        self.scored: list[tuple[Any, Any]] = []

    def decode(self, raw: str) -> Any:
        return raw

    def score(self, pred: Any, ref: Any) -> Mapping[str, object]:
        self.scored.append((pred, ref))
        return {"ok": 1}

    def aggregate(self, scores: list[Mapping[str, object]]) -> dict:
        return {"n": len(scores), "scores": scores}


def _echo(prompt: str) -> str:
    return prompt


def _rows() -> list[dict]:
    return [
        {"key": "a", "x": "one", "y": "One"},
        {"key": "b", "x": "two", "y": "Two"},
    ]


# The identity run_eval derives for a duck-typed Scorable: its own aggregate.
_FAKE_AGG = f"{FakeTask.aggregate.__module__}.{FakeTask.aggregate.__qualname__}"


def test_run_eval_returns_aggregate_and_scores_once_per_row():
    task = FakeTask()
    result, info = run_eval(task=task, rows=_rows(), generate=_echo)
    assert result == {"n": 2, "scores": [{"ok": 1}, {"ok": 1}]}
    assert len(task.scored) == 2
    # n_eval_rows and the aggregator identity are always known (run_eval counts
    # the rows and holds the aggregate callable); every other K-pinning field
    # is caller-supplied, so it's absent when not passed in.
    assert info == {"n_eval_rows": 2, "aggregator": _FAKE_AGG}


def test_run_eval_rejects_unscored_task():
    class Unscored:
        score = None

        def decode(self, raw: str) -> Any:
            return raw

        def aggregate(self, scores: list[Mapping[str, object]]) -> dict:
            return {}

    with pytest.raises(ValueError, match="no scorer"):
        run_eval(task=Unscored(), rows=_rows(), generate=_echo)


def test_sink_receives_one_record_per_row():
    task = FakeTask()
    records: list[Mapping] = []
    run_eval(task=task, rows=_rows(), generate=_echo, sink=records.append)
    assert records == [
        {"key": "a", "x": "one", "y": "One", "raw": "one"},
        {"key": "b", "x": "two", "y": "Two", "raw": "two"},
    ]


def test_sink_records_are_directly_replayable_by_rescore():
    # The persisted schema must be re-consumable without an adapter: whatever the
    # sink writes during a live run is exactly what rescore() reads back.
    live = FakeTask()
    records: list[Mapping] = []
    run_eval(task=live, rows=_rows(), generate=_echo, sink=records.append)

    replayed = FakeTask()
    result, _info = rescore(task=replayed, rows_with_raw=records)
    assert result["n"] == 2
    assert replayed.scored == [("one", "One"), ("two", "Two")]


def test_str_pred_and_ref_are_unbounded():
    task = FakeTask()
    run_eval(task=task, rows=_rows(), generate=_echo)
    pred, ref = task.scored[0]
    assert pred == "one"
    assert ref == "One"


def test_run_eval_populates_pinning_fields_when_caller_supplies_them():
    task = FakeTask()
    _result, info = run_eval(
        task=task,
        rows=_rows(),
        generate=_echo,
        task_name="htmlmd",
        codec="json",
        scorer_version="trafilatura==1.2.0,spacy==3.7.0",
        dataset_ref="htmlmd-eval",
        dataset_revision="2026-01-01",
        split="test",
    )
    assert info == {
        "n_eval_rows": 2,
        "aggregator": _FAKE_AGG,
        "task": "htmlmd",
        "codec": "json",
        "scorer_version": "trafilatura==1.2.0,spacy==3.7.0",
        "dataset": "htmlmd-eval",
        "dataset_revision": "2026-01-01",
        "split": "test",
    }


def test_rescore_populates_n_eval_rows_and_pinning_fields():
    task = FakeTask()
    records: list[Mapping] = []
    run_eval(task=task, rows=_rows(), generate=_echo, sink=records.append)

    _result, info = rescore(
        task=FakeTask(),
        rows_with_raw=records,
        task_name="htmlmd",
        split="test",
    )
    assert info == {
        "n_eval_rows": 2,
        "aggregator": _FAKE_AGG,
        "task": "htmlmd",
        "split": "test",
    }


def test_aggregator_identity_distinguishes_custom_aggregator():
    from reishi.primitives.task import Task

    def counts_only(scores: list[Mapping[str, object]]) -> dict:
        return {"n": len(scores)}

    scorer = lambda pred, ref: {"ok": 1}  # noqa: E731
    default = Task(name="t", description="", score=scorer)
    custom = Task(name="t", description="", score=scorer, aggregator=counts_only)

    _m, default_info = run_eval(task=default, rows=_rows(), generate=_echo)
    _m, custom_info = run_eval(task=custom, rows=_rows(), generate=_echo)

    # A default Task's fallback is deterministic (field_aggregate inside
    # Task.aggregate), so the generic identity is stable and comparable; a
    # custom aggregator must be distinguishable from it.
    assert default_info["aggregator"].endswith("Task.aggregate")
    assert custom_info["aggregator"].endswith("counts_only")
    assert default_info["aggregator"] != custom_info["aggregator"]


def test_aggregator_override_wins_over_derived_identity():
    _m, info = run_eval(
        task=FakeTask(), rows=_rows(), generate=_echo, aggregator="field_agg==2.0"
    )
    assert info["aggregator"] == "field_agg==2.0"


def test_render_variants():
    assert _render(None, "raw") == "raw"
    assert _render(None, 42) == "42"
    assert _render("Q: {x}", "hi") == "Q: hi"
    assert _render("prefix ", "hi") == "prefix hi"
    # other literal braces (a JSON example) must survive -- no str.format blowup.
    assert _render('e.g. {"a": 1}\n{x}', "hi") == 'e.g. {"a": 1}\nhi'


def test_rescore_replays_persisted_raw_without_inference():
    task = FakeTask()
    rows = [
        {"key": "a", "x": "one", "y": "One", "raw": "R1"},
        {"key": "b", "x": "two", "y": "Two", "raw": "R2"},
    ]
    result, _info = rescore(task=task, rows_with_raw=rows)
    assert result["n"] == 2
    # scoring the persisted raw (not x) proves no live model was invoked.
    assert task.scored == [("R1", "One"), ("R2", "Two")]


def test_run_eval_raising_generate_is_the_live_path():
    task = FakeTask()

    def _boom(prompt: str) -> str:
        raise AssertionError("model called")

    with pytest.raises(AssertionError):
        run_eval(task=task, rows=_rows(), generate=_boom)


def test_aggregator_identity_is_stable_for_callable_instances():
    import functools

    def agg_with_default(scores, top_k=None):
        return {"n": len(scores)}

    class InstanceTask(FakeTask):
        pass

    t = InstanceTask()
    t.aggregator = functools.partial(agg_with_default, top_k=3)  # type: ignore[attr-defined]

    _m, first = run_eval(task=t, rows=_rows(), generate=_echo)
    _m, second = run_eval(task=t, rows=_rows(), generate=_echo)

    # A partial has no __qualname__ and reprs with a memory address; the
    # identity must come from its type so it's equal across runs/processes.
    assert first["aggregator"] == second["aggregator"]
    assert "0x" not in first["aggregator"]
    assert first["aggregator"].endswith("functools.partial")

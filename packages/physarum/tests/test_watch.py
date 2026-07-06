"""'sweep watch' only ever reads already-saved Trial manifests -- these tests
pin how a Trial's recipe name (the only place sweep/trial-number lineage is
recorded) gets parsed back into the shape the dashboard graphs."""

import pytest

import reishi.tasks  # noqa: F401  (populate the task registry)
from reishi.primitives import trial as trial_store
from reishi.primitives.trial import Trial

from physarum import mcm_plugin
from physarum.objective import build_recipe
from physarum.watch import trials_for_sweep


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))


def _save(recipe_name: str, status: str, metrics: dict, trainer: dict, seed: int = 0, created: str = "") -> None:
    # recipe_name must be the bare '<sweep>-t<N>' shape build_recipe actually
    # produces -- the '-s<seed>-<uuid>' suffix lives only on Trial.id (trial.py
    # plan()), never on Trial.recipe itself.
    t = Trial(id=f"{recipe_name}-s{seed}-abc123", recipe=recipe_name, seed=seed, status=status, metrics=metrics,
               spec={"trainer": trainer}, created=created)
    trial_store.save(t)


def test_trials_for_sweep_parses_trial_number_and_sorts(tmp_path):
    _save("my-sweep-t2", "done", {"field_f1": 0.5}, {"include_tables": True})
    _save("my-sweep-t0", "done", {"field_f1": 0.8}, {"include_tables": False})
    _save("my-sweep-t1", "planned", {}, {"include_tables": True})

    rows = trials_for_sweep("my-sweep")
    assert [r["trial"] for r in rows] == [0, 1, 2]
    assert rows[0]["metrics"]["field_f1"] == 0.8
    assert rows[2]["params"] == {"include_tables": True}


def test_trials_for_sweep_ignores_other_sweeps(tmp_path):
    _save("my-sweep-t0", "done", {"field_f1": 0.5}, {})
    _save("other-sweep-t0", "done", {"field_f1": 0.9}, {})

    rows = trials_for_sweep("my-sweep")
    assert len(rows) == 1
    assert rows[0]["metrics"]["field_f1"] == 0.5


def test_build_recipe_naming_matches_trials_for_sweep_parsing(tmp_path):
    # end-to-end: the actual naming contract between objective.build_recipe and
    # watch.trials_for_sweep, not a hand-fabricated recipe string standing in
    # for it -- a naming change on either side should fail this test.
    template = {
        "name": "tpl", "task": "htmlmd", "base_model": None, "dataset": "d",
        "accelerator": "local", "prompt": None, "seeds": 1, "priority": 0, "trainer": {},
    }
    recipe = build_recipe(template, {"trainer.include_tables": True}, "my-sweep", 3)
    [t] = trial_store.plan(recipe)
    t.status, t.metrics = "done", {"field_f1": 0.42}
    trial_store.save(t)

    rows = trials_for_sweep("my-sweep")
    assert [r["trial"] for r in rows] == [3]
    assert rows[0]["metrics"]["field_f1"] == 0.42
    assert rows[0]["params"] == {"include_tables": True}


def test_trials_for_sweep_empty_when_none_saved(tmp_path):
    assert trials_for_sweep("no-such-sweep") == []


def test_sweep_sidecar_reads_what_optimize_writes(tmp_path):
    from reishi import store as reishi_store

    from physarum.watch import _sweep_sidecar

    assert _sweep_sidecar("my-sweep") == {"n_trials": None, "started_at": None}
    reishi_store.save("sweeps", "my-sweep", {"name": "my-sweep", "n_trials": 60, "started_at": "2026-07-06T12:00:00+00:00"})
    assert _sweep_sidecar("my-sweep") == {"n_trials": 60, "started_at": "2026-07-06T12:00:00+00:00"}


def test_trials_for_sweep_hides_trials_from_before_the_current_run_started(tmp_path):
    # Simulates re-running the same sweep name: an old trial from a prior run
    # (created before this run's started_at) must not resurface in the
    # dashboard alongside the new run's trials -- reishi's store never
    # deletes, so this filter is the only thing preventing that.
    _save("my-sweep-t0", "done", {"field_f1": 0.5}, {"a": True}, created="2026-07-06T10:00:00+00:00")
    _save("my-sweep-t0", "done", {"field_f1": 0.9}, {"a": False}, seed=1, created="2026-07-06T12:00:00+00:00")

    rows = trials_for_sweep("my-sweep", started_at="2026-07-06T12:00:00+00:00")
    assert len(rows) == 1
    assert rows[0]["metrics"]["field_f1"] == 0.9

    rows_unfiltered = trials_for_sweep("my-sweep")
    assert len(rows_unfiltered) == 2


def test_sweep_watch_needs_a_sweep_name():
    from reishi.cli.grammar import Command

    assert mcm_plugin.sweep_watch(Command(domain="sweep", action="watch", objects=[])) == 1


def test_watch_verb_registered_and_readonly():
    watch = next(v for v in mcm_plugin.VERBS if v.name == "watch")
    assert watch.readonly is True
    assert ("sweep", "watch") in mcm_plugin.HANDLERS

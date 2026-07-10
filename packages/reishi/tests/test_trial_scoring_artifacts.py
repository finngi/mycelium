"""Trial scoring-provenance + outputs-artifact round-trips, and tolerant load.

These pin the additive contract: a trial can carry persisted raw outputs
(for re-score/replay) and a record of what scored it, and old manifests missing
these keys -- or carrying unknown extra keys -- still load.
"""

from reishi.primitives.trial import Trial, record_scoring


def test_outputs_artifact_roundtrips():
    t = Trial(
        id="t-1",
        recipe_name="r",
        seed=0,
        artifacts={
            "weights": "hf://acme/adapter",
            "outputs": "file:///store/artifacts/t-1/outputs.jsonl",
        },
    )
    back = Trial.from_manifest(t.to_manifest())
    assert back.artifacts == t.artifacts
    assert back.artifacts["outputs"] == "file:///store/artifacts/t-1/outputs.jsonl"


def test_scoring_block_roundtrips():
    t = Trial(
        id="t-2",
        recipe_name="r",
        seed=0,
        scoring={
            "scorer": "extract",
            "scored_at": "2026-07-07T00:00:00+00:00",
            "scored_on": "gpu",
            "source": "live",
        },
    )
    back = Trial.from_manifest(t.to_manifest())
    assert back.scoring == t.scoring


def test_scoring_measurement_key_fields_roundtrip():
    t = Trial(
        id="t-11",
        recipe_name="r",
        seed=0,
        scoring={
            "scorer": "extract",
            "task": "extract",
            "codec": "json",
            "scorer_version": "trafilatura==1.2.0,spacy==3.7.0",
            "dataset": "htmlmd-eval",
            "dataset_revision": "2026-01-01",
            "split": "test",
            "n_eval_rows": 500,
        },
    )
    back = Trial.from_manifest(t.to_manifest())
    assert back.scoring == t.scoring


def test_old_manifest_without_scoring_or_outputs_loads():
    old = {
        "id": "t-3",
        "recipe_name": "r",
        "seed": 0,
        "status": "done",
        "created": "2026-07-01T00:00:00+00:00",
        "metrics": {"f1": 0.9},
        "artifacts": {"weights": "hf://acme/adapter"},
        "spec": {},
        "execution": {},
    }
    t = Trial.from_manifest(old)
    assert t.id == "t-3"
    assert t.scoring == {}
    assert t.observables == {}
    assert "outputs" not in t.artifacts


def test_scorings_list_roundtrips():
    t = Trial(
        id="t-10",
        recipe_name="r",
        seed=0,
        scorings=[
            {"scorer": "extract", "split": "val", "n_eval_rows": 100},
            {"scorer": "extract", "split": "ood", "n_eval_rows": 50},
        ],
    )
    back = Trial.from_manifest(t.to_manifest())
    assert back.scorings == t.scorings


def test_old_manifest_without_scorings_loads():
    old = {
        "id": "t-12",
        "recipe_name": "r",
        "seed": 0,
        "status": "done",
        "created": "2026-07-01T00:00:00+00:00",
        "metrics": {"f1": 0.9},
        "artifacts": {},
        "spec": {},
        "execution": {},
    }
    t = Trial.from_manifest(old)
    assert t.scorings == []


def test_record_scoring_dual_writes_bare_and_namespaced_metrics():
    t = Trial(id="t-13", recipe_name="r", seed=0)
    info = {"scorer": "extract", "split": "val", "n_eval_rows": 100}
    record_scoring(t, {"field_f1": 0.8}, info)
    assert t.metrics == {"field_f1": 0.8, "val/field_f1": 0.8}
    assert t.scoring == info
    assert t.scorings == [info]


def test_record_scoring_without_split_writes_bare_only():
    t = Trial(id="t-14", recipe_name="r", seed=0)
    record_scoring(t, {"field_f1": 0.7}, {"scorer": "extract"})
    assert t.metrics == {"field_f1": 0.7}


def test_record_scoring_second_split_appends_and_bare_reflects_latest():
    t = Trial(id="t-15", recipe_name="r", seed=0)
    val_info = {"scorer": "extract", "split": "val", "n_eval_rows": 100}
    ood_info = {"scorer": "extract", "split": "ood", "n_eval_rows": 50}
    record_scoring(t, {"field_f1": 0.8}, val_info)
    record_scoring(t, {"field_f1": 0.5}, ood_info)

    # both scoring records survive -- the second call appends, it does not replace.
    assert t.scorings == [val_info, ood_info]
    # primary `scoring` tracks the most recent record.
    assert t.scoring == ood_info
    # bare key reflects the latest write; the val-namespaced key is untouched,
    # proving the dual-write never deletes or renames an existing metrics key.
    assert t.metrics == {
        "field_f1": 0.5,
        "val/field_f1": 0.8,
        "ood/field_f1": 0.5,
    }


def test_observables_roundtrips():
    t = Trial(
        id="t-7",
        recipe_name="r",
        seed=0,
        observables={"wall_time_s": 42.0, "artifact_bytes": 1024},
    )
    back = Trial.from_manifest(t.to_manifest())
    assert back.observables == t.observables


def test_old_manifest_without_observables_loads():
    old = {
        "id": "t-8",
        "recipe_name": "r",
        "seed": 0,
        "status": "done",
        "created": "2026-07-01T00:00:00+00:00",
        "metrics": {"f1": 0.9},
        "artifacts": {},
        "spec": {},
        "execution": {},
    }
    t = Trial.from_manifest(old)
    assert t.observables == {}


def test_observables_not_shadowed_by_stale_extra():
    # Same guarantee as test_known_key_not_shadowed_by_stale_extra, for the
    # newly-added observables field.
    t = Trial.from_manifest({"id": "t-9", "recipe_name": "r", "seed": 0})
    t.extra["observables"] = {"wall_time_s": "STALE"}
    t.observables = {"wall_time_s": 1.0}
    assert t.to_manifest()["observables"] == {"wall_time_s": 1.0}


def test_manifest_unknown_key_survives_round_trip():
    m = {
        "id": "t-4",
        "recipe_name": "r",
        "seed": 0,
        "future_field": {"anything": True},
    }
    t = Trial.from_manifest(m)
    assert t.id == "t-4"
    assert t.extra == {"future_field": {"anything": True}}
    # An older checkout loading, then re-saving, must not lose the new field.
    assert t.to_manifest()["future_field"] == {"anything": True}


def test_manifest_key_named_extra_survives_round_trip():
    # A manifest key literally named "extra" must not be swallowed by the
    # unknown-key bucket -- it round-trips like any other unknown key.
    m = {"id": "t-5", "recipe_name": "r", "seed": 0, "extra": {"note": "hi"}}
    t = Trial.from_manifest(m)
    assert t.extra == {"extra": {"note": "hi"}}
    assert t.to_manifest()["extra"] == {"note": "hi"}


def test_known_key_not_shadowed_by_stale_extra():
    # Even if a stale value for a known key somehow lands in `extra`, the live
    # field wins on serialization.
    t = Trial.from_manifest({"id": "t-6", "recipe_name": "r", "seed": 0})
    t.extra["status"] = "STALE"
    t.status = "done"
    assert t.to_manifest()["status"] == "done"

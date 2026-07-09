"""Trial eval-provenance + predictions-artifact round-trips, and tolerant load.

These pin the additive contract: a trial can carry persisted raw predictions
(for re-score/replay) and a record of what scored it, and old manifests missing
these keys -- or carrying unknown extra keys -- still load.
"""

from reishi.primitives.trial import Trial


def test_predictions_artifact_roundtrips():
    t = Trial(
        id="t-1",
        recipe="r",
        seed=0,
        artifacts={
            "weights": "hf://acme/adapter",
            "predictions": "file:///store/artifacts/t-1/predictions.jsonl",
        },
    )
    back = Trial.from_manifest(t.to_manifest())
    assert back.artifacts == t.artifacts
    assert (
        back.artifacts["predictions"] == "file:///store/artifacts/t-1/predictions.jsonl"
    )


def test_eval_block_roundtrips():
    t = Trial(
        id="t-2",
        recipe="r",
        seed=0,
        eval={
            "scorer": "extract",
            "scored_at": "2026-07-07T00:00:00+00:00",
            "placement": "accelerator",
            "source": "live",
        },
    )
    back = Trial.from_manifest(t.to_manifest())
    assert back.eval == t.eval


def test_old_manifest_without_eval_or_predictions_loads():
    old = {
        "id": "t-3",
        "recipe": "r",
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
    assert t.eval == {}
    assert t.observables == {}
    assert "predictions" not in t.artifacts


def test_observables_roundtrips():
    t = Trial(
        id="t-7",
        recipe="r",
        seed=0,
        observables={"wall_time_s": 42.0, "artifact_bytes": 1024},
    )
    back = Trial.from_manifest(t.to_manifest())
    assert back.observables == t.observables


def test_old_manifest_without_observables_loads():
    old = {
        "id": "t-8",
        "recipe": "r",
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
    t = Trial.from_manifest({"id": "t-9", "recipe": "r", "seed": 0})
    t.extra["observables"] = {"wall_time_s": "STALE"}
    t.observables = {"wall_time_s": 1.0}
    assert t.to_manifest()["observables"] == {"wall_time_s": 1.0}


def test_manifest_unknown_key_survives_round_trip():
    m = {
        "id": "t-4",
        "recipe": "r",
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
    m = {"id": "t-5", "recipe": "r", "seed": 0, "extra": {"note": "hi"}}
    t = Trial.from_manifest(m)
    assert t.extra == {"extra": {"note": "hi"}}
    assert t.to_manifest()["extra"] == {"note": "hi"}


def test_known_key_not_shadowed_by_stale_extra():
    # Even if a stale value for a known key somehow lands in `extra`, the live
    # field wins on serialization.
    t = Trial.from_manifest({"id": "t-6", "recipe": "r", "seed": 0})
    t.extra["status"] = "STALE"
    t.status = "done"
    assert t.to_manifest()["status"] == "done"

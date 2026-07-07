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
    assert back.artifacts["predictions"] == "file:///store/artifacts/t-1/predictions.jsonl"


def test_eval_block_roundtrips():
    t = Trial(
        id="t-2",
        recipe="r",
        seed=0,
        eval={
            "scorer": "nameparse",
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
    assert "predictions" not in t.artifacts


def test_manifest_with_unknown_key_is_dropped():
    m = {
        "id": "t-4",
        "recipe": "r",
        "seed": 0,
        "future_field": {"anything": True},
    }
    t = Trial.from_manifest(m)
    assert t.id == "t-4"
    assert not hasattr(t, "future_field")

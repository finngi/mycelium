"""Task customization: custom decoder/aggregator/scorer override the defaults,
and the default path is the JSON codec plus field_aggregate.
"""

from reishi.primitives.task import Task, aggregate, field_aggregate


def test_custom_decoder_returns_non_dict():
    t = Task(name="echo", description="d", decoder=lambda raw: raw.strip().upper())
    assert t.decode("  hi  ") == "HI"


def test_custom_aggregator_returns_arbitrary_dict():
    t = Task(
        name="agg",
        description="d",
        aggregator=lambda scores: {"total": sum(int(s["v"]) for s in scores)},
    )
    assert t.aggregate([{"v": 1}, {"v": 2}, {"v": 3}]) == {"total": 6}


def test_loosely_typed_scorer_end_to_end():
    def scorer(pred: str, ref: str) -> dict:
        return {"correct": 1 if pred == ref else 0}

    t = Task(name="strcmp", description="d", score=scorer, codec="json")
    scores = [t.score(t.decode('{"a":1}'), {"a": 1})]
    assert scores == [{"correct": 1}]


def test_default_path_uses_json_codec_and_field_aggregate():
    def _score(pred: dict, ref: dict):
        cand, truth = set((pred or {}).items()), set((ref or {}).items())
        return {
            "tp": len(cand & truth),
            "fp": len(cand - truth),
            "fn": len(truth - cand),
            "exact_match": pred == ref,
            "invalid": not pred,
        }

    t = Task(name="extract", description="d", output_fields=("a",), score=_score)

    decoded = t.decode('garbage before {"a": 1} trailing')
    assert decoded == {"a": 1}

    scores = [
        _score(t.decode('{"a":1}'), {"a": 1}),
        _score(t.decode("nonsense"), {"a": 1}),
    ]
    metrics = t.aggregate(scores)

    assert metrics == field_aggregate(scores)
    assert metrics == aggregate(scores)
    assert metrics["n"] == 2
    assert metrics["field_precision"] == 1.0
    assert metrics["field_recall"] == 0.5
    assert metrics["invalid_output_rate"] == 0.5


def test_aggregate_alias_is_field_aggregate():
    assert aggregate is field_aggregate


def test_manifest_omits_empty_output_fields():
    bare = Task(name="bare", description="d").to_manifest()
    assert "output_fields" not in bare
    assert bare["codec"] == "json"
    assert bare["scorer"] == "missing"

    full = Task(
        name="full", description="d", output_fields=("a", "b"), score=lambda p, r: {}
    ).to_manifest()
    assert full["output_fields"] == ["a", "b"]
    assert full["scorer"] == "registered"

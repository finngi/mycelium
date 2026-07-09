from reishi.primitives import board
from reishi.primitives.codec import get_codec
from reishi.primitives.trial import Trial


def test_text_codec_is_identity():
    codec = get_codec("text")
    assert codec.decode("# hello") == "# hello"
    assert codec.encode("# hello") == "# hello"


def test_json_codec_still_returns_dict():
    codec = get_codec("json")
    assert codec.decode('{"a": 1}') == {"a": 1}
    assert codec.encode({"a": 1}) == '{"a":1}'


def _done(recipe: str, metrics: dict) -> Trial:
    return Trial(
        id=f"{recipe}-{len(metrics)}-{id(metrics)}",
        recipe=recipe,
        seed=0,
        status="done",
        metrics=metrics,
        spec={"task": "t", "base_model": "m"},  # type: ignore[typeddict-item]
    )


def test_board_reports_scored_vs_trials(monkeypatch):
    trials = [
        _done("all", {"f1": 0.8}),
        _done("all", {"f1": 0.6}),
        _done("some", {"f1": 0.5}),
        _done("some", {}),
        _done("none", {}),
        _done("none", {}),
    ]
    monkeypatch.setattr(board.trial_store, "load_all", lambda: trials)

    rows = {r["recipe"]: r for r in board.build(metric="f1")}

    assert rows["all"]["trials"] == 2
    assert rows["all"]["scored"] == 2
    assert rows["all"]["f1"] == 0.7

    assert rows["some"]["trials"] == 2
    assert rows["some"]["scored"] == 1
    assert rows["some"]["f1"] == 0.5

    assert "none" in rows
    assert rows["none"]["trials"] == 2
    assert rows["none"]["scored"] == 0
    assert rows["none"]["f1"] is None
    assert rows["none"]["f1_min"] is None
    assert rows["none"]["f1_max"] is None


def test_board_skips_non_scalar_metric_leaves(monkeypatch, capsys):
    trials = [
        _done("mix", {"f1": 0.8}),
        _done("mix", {"f1": [0.1, 0.2, 0.3]}),  # a curve -- not a coordinate
        _done("mix", {"f1": {"tp": 1}}),  # a confusion dict -- not a coordinate
        _done("mix", {"f1": True}),  # a bool -- treated as non-scalar
        _done("rich_only", {"f1": {"curve": [1, 2]}}),
    ]
    monkeypatch.setattr(board.trial_store, "load_all", lambda: trials)

    rows = {r["recipe"]: r for r in board.build(metric="f1")}

    # Only the one real number is aggregated; the other three leaves are skipped.
    assert rows["mix"]["trials"] == 4
    assert rows["mix"]["scored"] == 1
    assert rows["mix"]["f1"] == 0.8

    # A recipe with no scalar leaf falls through to the null-aggregate path
    # rather than crashing the whole board.
    assert rows["rich_only"]["scored"] == 0
    assert rows["rich_only"]["f1"] is None

    err = capsys.readouterr().err
    assert "[WARN]" in err
    assert "non-scalar" in err


def test_board_sorts_unscored_last(monkeypatch):
    trials = [
        _done("none", {}),
        _done("high", {"f1": 0.9}),
        _done("low", {"f1": 0.1}),
    ]
    monkeypatch.setattr(board.trial_store, "load_all", lambda: trials)

    order = [r["recipe"] for r in board.build(metric="f1")]
    assert order == ["high", "low", "none"]

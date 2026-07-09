from reishi.primitives import board
from reishi.primitives.trial import Trial


def _done(
    recipe: str, metrics: dict, eval_info: dict | None = None, seed: int = 0
) -> Trial:
    return Trial(
        id=f"{recipe}-{seed}-{id(metrics)}",
        recipe=recipe,
        seed=seed,
        status="done",
        metrics=metrics,
        eval=eval_info or {},
        spec={"task": "t", "base_model": "m"},  # type: ignore[typeddict-item]
    )


def test_board_warns_once_when_group_mixes_dataset_revision(monkeypatch, capsys):
    trials = [
        _done("r", {"f1": 0.8}, {"dataset_revision": "v1"}, seed=0),
        _done("r", {"f1": 0.6}, {"dataset_revision": "v2"}, seed=1),
    ]
    monkeypatch.setattr(board.trial_store, "load_all", lambda: trials)

    board.build(metric="f1")

    err = capsys.readouterr().err
    assert err.count("[WARN]") == 1
    assert "recipe 'r' mixes measurement keys (dataset_revision)" in err


def test_board_warns_with_every_mismatched_field_listed(monkeypatch, capsys):
    trials = [
        _done("r", {"f1": 0.8}, {"scorer": "extract", "eval_n": 100}, seed=0),
        _done("r", {"f1": 0.6}, {"scorer": "extract-v2", "eval_n": 50}, seed=1),
    ]
    monkeypatch.setattr(board.trial_store, "load_all", lambda: trials)

    board.build(metric="f1")

    err = capsys.readouterr().err
    assert err.count("[WARN]") == 1
    assert "scorer" in err
    assert "eval_n" in err


def test_board_silent_when_keys_match(monkeypatch, capsys):
    key = {"dataset_revision": "v1", "split": "test", "eval_n": 100}
    trials = [
        _done("r", {"f1": 0.8}, dict(key), seed=0),
        _done("r", {"f1": 0.6}, dict(key), seed=1),
    ]
    monkeypatch.setattr(board.trial_store, "load_all", lambda: trials)

    board.build(metric="f1")

    assert capsys.readouterr().err == ""


def test_board_silent_when_eval_info_entirely_absent(monkeypatch, capsys):
    trials = [
        _done("r", {"f1": 0.8}, {}, seed=0),
        _done("r", {"f1": 0.6}, {}, seed=1),
    ]
    monkeypatch.setattr(board.trial_store, "load_all", lambda: trials)

    board.build(metric="f1")

    assert capsys.readouterr().err == ""


def test_board_silent_when_one_trial_has_no_eval_info_at_all(monkeypatch, capsys):
    # Absent is not a conflict: only pinned-and-differing values are.
    trials = [
        _done("r", {"f1": 0.8}, {"dataset_revision": "v1"}, seed=0),
        _done("r", {"f1": 0.6}, {}, seed=1),
    ]
    monkeypatch.setattr(board.trial_store, "load_all", lambda: trials)

    board.build(metric="f1")

    assert capsys.readouterr().err == ""


def test_board_ignores_key_of_trials_that_did_not_contribute_a_value(
    monkeypatch, capsys
):
    # A trial with no metric for this board's metric never enters the average,
    # so its eval info shouldn't be compared against the trials that did.
    trials = [
        _done("r", {"f1": 0.8}, {"dataset_revision": "v1"}, seed=0),
        _done("r", {}, {"dataset_revision": "v2"}, seed=1),
    ]
    monkeypatch.setattr(board.trial_store, "load_all", lambda: trials)

    board.build(metric="f1")

    assert capsys.readouterr().err == ""

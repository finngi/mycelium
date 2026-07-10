"""Comparison round-trips: the manifest/dataclass contract mirrors Trial's --
unknown top-level keys survive a load/save cycle, and a key literally named
"extra" isn't swallowed by the carry-over field of the same name.
"""

import pytest

from reishi import store
from reishi.primitives.comparison import Comparison, load, load_all, record, save


def test_comparison_roundtrips():
    c = Comparison(
        id="cmp-1",
        trial_a="t-a",
        trial_b="t-b",
        winner="a",
        judge="gpt-5",
        created="2026-07-01T00:00:00+00:00",
        scoring={"task": "extract", "dataset": "htmlmd-eval", "split": "test"},
    )
    back = Comparison.from_manifest(c.to_manifest())
    assert back == c


def test_comparison_unknown_key_preserved():
    c = Comparison(
        id="cmp-1", trial_a="t-a", trial_b="t-b", winner="tie", judge="human"
    )
    m = c.to_manifest() | {"from_the_future": True}
    loaded = Comparison.from_manifest(m)
    assert loaded.to_manifest()["from_the_future"] is True


def test_comparison_literal_extra_key_preserved():
    m = {
        "id": "cmp-1",
        "trial_a": "t-a",
        "trial_b": "t-b",
        "winner": "b",
        "judge": "human",
        "created": "",
        "scoring": {},
        "extra": "not-swallowed",
    }
    loaded = Comparison.from_manifest(m)
    assert loaded.extra == {"extra": "not-swallowed"}
    assert loaded.to_manifest()["extra"] == "not-swallowed"


def test_comparison_known_key_never_shadowed_by_stale_extra():
    c = Comparison(id="cmp-1", trial_a="t-a", trial_b="t-b", winner="a", judge="human")
    m = c.to_manifest() | {"winner": "b"}
    loaded = Comparison.from_manifest(m)
    assert loaded.winner == "b"
    assert loaded.to_manifest()["winner"] == "b"


def test_record_rejects_unknown_winner():
    with pytest.raises(ValueError, match="winner"):
        record("t-a", "t-b", winner="nope", judge="human")


def test_record_builds_a_valid_comparison():
    c = record("t-a", "t-b", winner="a", judge="gpt-5")
    assert c.trial_a == "t-a"
    assert c.trial_b == "t-b"
    assert c.winner == "a"
    assert c.created


def test_comparison_save_load_roundtrips_through_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    store.use_backend(store.LocalFilesystemBackend())
    try:
        c = record("t-a", "t-b", winner="tie", judge="human")
        save(c)
        assert load(c.id) == c
        assert [x.id for x in load_all()] == [c.id]
    finally:
        store.use_backend(store.LocalFilesystemBackend())

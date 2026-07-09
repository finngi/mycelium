"""query()/stream() -- reserved StorageBackend surface (docs/design/pr-plan.md).

Additive by construction: every backend that only ever implemented
save/load/load_all keeps working unchanged, whether or not it inherits the
StorageBackend protocol default.
"""

import json

import pytest

from reishi import store
from reishi.store.base import StorageBackend


@pytest.fixture(params=["fs", "sqlite"])
def backend(request, tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    b = (
        store.LocalFilesystemBackend()
        if request.param == "fs"
        else store.SqliteBackend()
    )
    store.use_backend(b)
    yield b
    store.use_backend(store.LocalFilesystemBackend())


# --- protocol default: a backend that never heard of query/stream ---


class MinimalBackend(StorageBackend):
    """Implements only the pre-existing three methods; query/stream must
    come entirely from the protocol's default (load_all-backed) bodies."""

    def __init__(self):
        self.data: dict[tuple[str, str], dict] = {}

    def save(self, kind, name, manifest):
        self.data[(kind, name)] = dict(manifest)

    def load(self, kind, name):
        return self.data[(kind, name)]

    def load_all(self, kind, *, tolerant=True):
        return [v for (k, _), v in self.data.items() if k == kind]


def test_protocol_default_stream_falls_back_to_load_all():
    b = MinimalBackend()
    b.save("trials", "a", {"id": "a", "status": "done"})
    b.save("trials", "b", {"id": "b", "status": "planned"})
    assert sorted(m["id"] for m in b.stream("trials")) == ["a", "b"]


def test_protocol_default_query_falls_back_to_load_all():
    b = MinimalBackend()
    b.save("trials", "a", {"id": "a", "status": "done"})
    b.save("trials", "b", {"id": "b", "status": "planned"})
    assert [m["id"] for m in b.query("trials", status="done")] == ["a"]


def test_facade_query_and_stream_work_against_a_backend_with_neither():
    class NoQueryBackend:
        """Duck-typed like enoki's PostgresBackend: matches the pre-reservation
        surface only, no inheritance from StorageBackend."""

        def __init__(self):
            self.data = {}

        def save(self, kind, name, manifest):
            self.data[(kind, name)] = dict(manifest)

        def load(self, kind, name):
            return self.data[(kind, name)]

        def load_all(self, kind, *, tolerant=True):
            return [v for (k, _), v in self.data.items() if k == kind]

    store.use_backend(NoQueryBackend())
    store.save("trials", "a", {"id": "a", "status": "done"})
    store.save("trials", "b", {"id": "b", "status": "planned"})
    assert sorted(m["id"] for m in store.stream("trials")) == ["a", "b"]
    assert [m["id"] for m in store.query("trials", status="done")] == ["a"]
    store.use_backend(store.LocalFilesystemBackend())


# --- filesystem backend: genuine laziness ---


def test_fs_stream_reads_one_file_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    fs = store.LocalFilesystemBackend()
    fs.save("trials", "a-aaa111", {"id": "a"})
    fs.save("trials", "b-bbb222", {"id": "b"})

    reads = []
    real_read_text = __import__("pathlib").Path.read_text

    def counting_read_text(self, *args, **kwargs):
        reads.append(self.name)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr("pathlib.Path.read_text", counting_read_text)

    it = fs.stream("trials")
    first = next(it)
    assert first["id"] == "a"
    assert reads == ["a-aaa111.json"]  # second file untouched until pulled


def test_fs_stream_yields_good_manifests_before_a_later_corrupt_one(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    fs = store.LocalFilesystemBackend()
    fs.save("trials", "a-good111", {"id": "a"})
    (tmp_path / "trials" / "z-broken.json").write_text("{not json")

    out = list(fs.stream("trials"))
    assert [m["id"] for m in out] == ["a"]


# --- query filter correctness, both backends ---


def test_query_matches_single_equality_filter(backend):
    store.save("trials", "a", {"id": "a", "status": "done", "seed": 0})
    store.save("trials", "b", {"id": "b", "status": "planned", "seed": 0})
    assert [m["id"] for m in store.query("trials", status="done")] == ["a"]


def test_query_matches_multiple_filters_as_and(backend):
    store.save("trials", "a", {"id": "a", "status": "done", "seed": 0})
    store.save("trials", "b", {"id": "b", "status": "done", "seed": 1})
    assert [m["id"] for m in store.query("trials", status="done", seed=0)] == ["a"]


def test_query_no_filters_returns_everything(backend):
    store.save("trials", "a", {"id": "a"})
    store.save("trials", "b", {"id": "b"})
    assert sorted(m["id"] for m in store.query("trials")) == ["a", "b"]


def test_query_no_match_returns_empty(backend):
    store.save("trials", "a", {"id": "a", "status": "done"})
    assert store.query("trials", status="nope") == []


def test_query_only_returns_requested_kind(backend):
    store.save("trials", "a", {"id": "a", "status": "done"})
    store.save("datasets", "d", {"id": "d", "status": "done"})
    assert [m["id"] for m in store.query("trials", status="done")] == ["a"]


def test_query_rejects_unsafe_filter_key(backend):
    with pytest.raises(store.StoreError):
        store.query("trials", **{"a.b": 1})


def test_stream_yields_all_manifests_of_kind(backend):
    store.save("trials", "a", {"id": "a"})
    store.save("trials", "b", {"id": "b"})
    assert sorted(m["id"] for m in store.stream("trials")) == ["a", "b"]


def test_stream_is_a_lazy_iterator_not_a_list(backend):
    store.save("trials", "a", {"id": "a"})
    result = store.stream("trials")
    assert not isinstance(result, list)
    assert list(result) == [{"id": "a"}]


def test_query_is_identical_across_backends(tmp_path, monkeypatch):
    manifest_a = {"id": "a", "status": "done", "seed": 3}
    manifest_b = {"id": "b", "status": "planned", "seed": 3}

    monkeypatch.setenv("MCM_STORE", str(tmp_path / "fs"))
    store.use_backend(store.LocalFilesystemBackend())
    store.save("trials", "a", manifest_a)
    store.save("trials", "b", manifest_b)
    from_fs = store.query("trials", status="done")

    monkeypatch.setenv("MCM_STORE", str(tmp_path / "sql"))
    store.use_backend(store.SqliteBackend())
    store.save("trials", "a", manifest_a)
    store.save("trials", "b", manifest_b)
    from_sql = store.query("trials", status="done")

    assert from_fs == from_sql == [manifest_a]
    store.use_backend(store.LocalFilesystemBackend())


def test_sqlite_query_uses_json_extract_not_full_scan():
    # sqlite3.Connection is a C type and can't be instance- or
    # class-monkeypatched, so this checks the SQL-backed shape structurally:
    # json1's json_extract in a WHERE clause is what makes query() a genuine
    # in-database filter instead of a Python-side load_all + filter.
    import inspect

    from reishi.store.sqlite import SqliteBackend

    assert "json_extract" in inspect.getsource(SqliteBackend.query)
    assert "WHERE" in inspect.getsource(SqliteBackend.query)


def test_sqlite_query_handles_null_filter_value(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    sq = store.SqliteBackend()
    sq.save("trials", "a", {"id": "a", "parent": None})
    sq.save("trials", "b", {"id": "b", "parent": "a"})
    assert [m["id"] for m in sq.query("trials", parent=None)] == ["a"]


def test_manifest_dump_still_valid_json(tmp_path, monkeypatch):
    # Sanity check that query() doesn't change what's on disk.
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    store.use_backend(store.LocalFilesystemBackend())
    store.save("trials", "a", {"id": "a", "status": "done"})
    store.query("trials", status="done")
    raw = (tmp_path / "trials" / "a.json").read_text()
    assert json.loads(raw) == {"id": "a", "status": "done"}
    store.use_backend(store.LocalFilesystemBackend())

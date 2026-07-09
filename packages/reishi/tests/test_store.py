"""Store-seam contract: the guarantees callers rely on regardless of backend.

The contract tests run against every backend; backend-specific behaviour
(atomic temp files, corrupt-file tolerance, db path resolution, default
selection) is tested separately. Every test pins MCM_STORE to a tmp location so
the real ~/.mcm store is never touched.
"""

import json

import pytest

from reishi import store


@pytest.fixture(params=["fs", "sqlite"])
def backend(request, tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    b = (
        store.LocalFilesystemBackend()
        if request.param == "fs"
        else store.SqliteBackend()
    )
    store.use_backend(b)
    yield request.param
    store.use_backend(store.LocalFilesystemBackend())


# --- contract: identical across every backend ---


def test_save_load_roundtrip(backend):
    store.save("trials", "demo-s0-abc123", {"id": "demo-s0-abc123", "seed": 0})
    assert store.load("trials", "demo-s0-abc123")["seed"] == 0


def test_load_missing_raises(backend):
    with pytest.raises(FileNotFoundError):
        store.load("trials", "nope")


@pytest.mark.parametrize("bad", ["../escape", "a/b", ".", "..", "", "with space"])
def test_unsafe_names_rejected(backend, bad):
    with pytest.raises(store.StoreError):
        store.save("trials", bad, {})


@pytest.mark.parametrize("bad", ["../escape", "a/b", ".", "..", "", "with space"])
def test_unsafe_kinds_rejected(backend, bad):
    with pytest.raises(store.StoreError):
        store.save(bad, "t-s0-aaa111", {})


def test_load_all_returns_only_its_kind(backend):
    store.save("trials", "a-s0-aaa111", {"id": "a"})
    store.save("trials", "b-s0-bbb222", {"id": "b"})
    store.save("datasets", "d-010101", {"name": "d"})
    assert sorted(m["id"] for m in store.load_all("trials")) == ["a", "b"]


def test_save_overwrites_same_key(backend):
    store.save("trials", "x-s0-aaa111", {"id": "x", "status": "planned"})
    store.save("trials", "x-s0-aaa111", {"id": "x", "status": "done"})
    assert store.load("trials", "x-s0-aaa111")["status"] == "done"
    assert len(store.load_all("trials")) == 1


def test_manifest_is_identical_across_backends(tmp_path, monkeypatch):
    manifest = {"id": "t", "seed": 3, "metrics": {"f1": 0.9}, "nested": {"a": [1, 2]}}

    monkeypatch.setenv("MCM_STORE", str(tmp_path / "fs"))
    store.use_backend(store.LocalFilesystemBackend())
    store.save("trials", "t-s0-aaa111", manifest)
    from_fs = store.load("trials", "t-s0-aaa111")

    monkeypatch.setenv("MCM_STORE", str(tmp_path / "sql"))
    store.use_backend(store.SqliteBackend())
    store.save("trials", "t-s0-aaa111", manifest)
    from_sql = store.load("trials", "t-s0-aaa111")

    assert from_fs == from_sql == manifest
    store.use_backend(store.LocalFilesystemBackend())


def test_backends_persist_byte_identical_docs(tmp_path, monkeypatch):
    # Not just equal-after-load: the serialised form on disk is the same string,
    # so the two backends can never quietly diverge in what they persist.
    manifest = {"id": "t", "note": "café", "nested": {"a": [1, 2]}}

    monkeypatch.setenv("MCM_STORE", str(tmp_path / "fs"))
    store.use_backend(store.LocalFilesystemBackend())
    store.save("trials", "t-s0-aaa111", manifest)
    file_doc = (tmp_path / "fs" / "trials" / "t-s0-aaa111.json").read_text()

    monkeypatch.setenv("MCM_STORE", str(tmp_path / "sql"))
    sq = store.SqliteBackend()
    store.use_backend(sq)
    store.save("trials", "t-s0-aaa111", manifest)
    db_doc = sq._db.execute(
        "SELECT doc FROM manifests WHERE kind = ? AND name = ?",
        ("trials", "t-s0-aaa111"),
    ).fetchone()[0]

    assert file_doc == db_doc
    store.use_backend(store.LocalFilesystemBackend())


def test_backend_swap_is_honored():
    class MemoryBackend:
        def __init__(self):
            self.data = {}

        def save(self, kind, name, manifest):
            self.data[(kind, name)] = manifest

        def load(self, kind, name):
            return self.data[(kind, name)]

        def load_all(self, kind, *, tolerant=True):
            return [v for (k, _), v in self.data.items() if k == kind]

    store.use_backend(MemoryBackend())
    store.save("trials", "mem", {"id": "mem"})
    assert store.load("trials", "mem") == {"id": "mem"}
    assert store.root() == store.Path("<remote>")
    store.use_backend(store.LocalFilesystemBackend())


# --- default selection (sqlite is opt-out) ---


def test_default_backend_is_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    monkeypatch.delenv("MCM_STORE_BACKEND", raising=False)
    store._backend = None  # force the lazy default to re-resolve
    store.save("trials", "t-s0-aaa111", {"id": "t"})
    assert (tmp_path / "store.db").is_file()
    store.use_backend(store.LocalFilesystemBackend())


def test_backend_env_selects_fs(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    monkeypatch.setenv("MCM_STORE_BACKEND", "fs")
    store._backend = None
    store.save("trials", "t-s0-aaa111", {"id": "t"})
    assert (tmp_path / "trials" / "t-s0-aaa111.json").exists()
    store.use_backend(store.LocalFilesystemBackend())


def test_unknown_backend_env_raises(monkeypatch):
    monkeypatch.setenv("MCM_STORE_BACKEND", "bogus")
    store._backend = None
    with pytest.raises(store.StoreError):
        store.save("trials", "t-s0-aaa111", {"id": "t"})
    store.use_backend(store.LocalFilesystemBackend())


# --- sqlite specifics ---


def test_sqlite_db_lives_inside_the_mcm_store_dir(tmp_path, monkeypatch):
    # MCM_STORE always names a directory; the db is store.db inside it, whether
    # or not the directory exists yet (no dir-vs-file ambiguity).
    monkeypatch.setenv("MCM_STORE", str(tmp_path / "not-created-yet"))
    b = store.SqliteBackend()
    assert b.root() == tmp_path / "not-created-yet"
    assert (b.root() / "store.db").exists()


# --- filesystem specifics (the layout oyster's git-as-queue depends on) ---


def test_fs_save_is_atomic_leaves_no_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    store.use_backend(store.LocalFilesystemBackend())
    store.save("datasets", "orgs-040726", {"name": "orgs-040726"})
    assert list((tmp_path / "datasets").glob("*.tmp")) == []
    store.use_backend(store.LocalFilesystemBackend())


def test_fs_load_all_skips_corrupt_file_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    store.use_backend(store.LocalFilesystemBackend())
    store.save("trials", "good-s0-aaa111", {"id": "good"})
    (tmp_path / "trials" / "broken.json").write_text("{not json")
    assert [m["id"] for m in store.load_all("trials")] == ["good"]
    store.use_backend(store.LocalFilesystemBackend())


def test_fs_load_all_skips_undecodable_file_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    store.use_backend(store.LocalFilesystemBackend())
    store.save("trials", "good-s0-aaa111", {"id": "good"})
    (tmp_path / "trials" / "notutf8.json").write_bytes(b"\xff\xfe\x00 not valid utf-8")
    assert [m["id"] for m in store.load_all("trials")] == ["good"]
    store.use_backend(store.LocalFilesystemBackend())


def test_fs_load_all_strict_reraises_on_corrupt(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    store.use_backend(store.LocalFilesystemBackend())
    store.save("trials", "good-s0-aaa111", {"id": "good"})
    (tmp_path / "trials" / "broken.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        store.load_all("trials", tolerant=False)
    store.use_backend(store.LocalFilesystemBackend())


# --- artifact store root ---


def test_artifact_root_default_and_override(monkeypatch):
    monkeypatch.delenv("MCM_ARTF_STORE", raising=False)
    assert store.artifact_root() == store.Path.home() / ".mcm" / "artifacts"
    monkeypatch.setenv("MCM_ARTF_STORE", "/tmp/blobs")
    assert store.artifact_root() == store.Path("/tmp/blobs")

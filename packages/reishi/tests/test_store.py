"""Store-seam contract: the guarantees callers rely on regardless of backend.

Every test pins MCM_STORE to a tmp dir so the real ~/.mcm/store is never touched.
"""

import json

import pytest

from mcm import store


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    store.use_backend(store.LocalFilesystemBackend())
    yield
    store.use_backend(store.LocalFilesystemBackend())


def test_save_load_roundtrip():
    store.save("trials", "demo-s0-abc123", {"id": "demo-s0-abc123", "seed": 0})
    assert store.load("trials", "demo-s0-abc123")["seed"] == 0


def test_load_missing_raises():
    with pytest.raises(FileNotFoundError):
        store.load("trials", "nope")


@pytest.mark.parametrize("bad", ["../escape", "a/b", ".", "..", "", "with space"])
def test_unsafe_names_rejected(bad):
    with pytest.raises(store.StoreError):
        store.save("trials", bad, {})


def test_save_is_atomic_leaves_no_tmp(tmp_path):
    store.save("datasets", "orgs-040726", {"name": "orgs-040726"})
    leftovers = list((tmp_path / "datasets").glob("*.tmp"))
    assert leftovers == []


def test_load_all_skips_corrupt_file_by_default(tmp_path):
    store.save("trials", "good-s0-aaa111", {"id": "good"})
    (tmp_path / "trials" / "broken.json").write_text("{not json")
    loaded = store.load_all("trials")
    assert [m["id"] for m in loaded] == ["good"]


def test_load_all_strict_reraises_on_corrupt(tmp_path):
    store.save("trials", "good-s0-aaa111", {"id": "good"})
    (tmp_path / "trials" / "broken.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        store.load_all("trials", tolerant=False)


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

"""Manifest store: a StorageBackend seam.

Everything mcm knows about trials, datasets, and boards is a manifest reached
through three calls -- save / load / load_all over a (kind, name) key. Backends
plug in behind one Protocol so swapping the store never touches callers:

  - SqliteBackend            default; a single ~/.mcm/store/store.db (MCM_STORE_BACKEND=sqlite)
  - LocalFilesystemBackend   one JSON file per manifest (MCM_STORE_BACKEND=fs); oyster pins this
  - PostgresBackend          the cloud store of record, contributed by an executor via mcm.plugins

Manifests are small JSON truth records; the big blobs they reference by URI
(adapters, checkpoints, datasets) live in the separate artifact store rooted at
artifact_root() (MCM_ARTF_STORE). The CLI stays a view over manifests, never
over live state.
"""

import os
from collections.abc import Mapping
from pathlib import Path

from reishi.store.base import StorageBackend, StoreError, safe_name
from reishi.store.filesystem import LocalFilesystemBackend
from reishi.store.sqlite import SqliteBackend

# Back-compat alias: some callers referenced the private name before the split.
_safe_name = safe_name

__all__ = [
    "StorageBackend",
    "StoreError",
    "LocalFilesystemBackend",
    "SqliteBackend",
    "use_backend",
    "save",
    "load",
    "load_all",
    "root",
    "artifact_root",
]

_backend: StorageBackend | None = None


def _default_backend() -> StorageBackend:
    name = os.environ.get("MCM_STORE_BACKEND", "sqlite").strip().lower()
    if name == "sqlite":
        return SqliteBackend()
    if name == "fs":
        return LocalFilesystemBackend()
    raise StoreError(f"unknown MCM_STORE_BACKEND {name!r} (want 'sqlite' or 'fs')")


def _active() -> StorageBackend:
    # Lazy: importing reishi.store must not create ~/.mcm/store/store.db as a side
    # effect (keeps imports hermetic and tests that pin a backend clean).
    global _backend
    if _backend is None:
        _backend = _default_backend()
    return _backend


def use_backend(backend: StorageBackend) -> None:
    global _backend
    _backend = backend


def save(kind: str, name: str, manifest: Mapping[str, object]) -> None:
    _active().save(kind, name, manifest)


def load(kind: str, name: str) -> dict:
    return _active().load(kind, name)


def load_all(kind: str, *, tolerant: bool = True) -> list[dict]:
    return _active().load_all(kind, tolerant=tolerant)


def root() -> Path:
    # Location of the local store; only meaningful for the local backends
    # (the CLI `info` view). Remote backends report a placeholder.
    getter = getattr(_active(), "root", None)
    return getter() if getter else Path("<remote>")


def artifact_root() -> Path:
    """Root for locally-staged artifacts (adapters, checkpoints, datasets).

    The blob counterpart to the manifest store: manifests hold URIs pointing
    here (or at gs://, hf://). Executors write under artifact_root()/<trial-id>
    and upload from there; the URI that wins is recorded back in the manifest.
    """
    return Path(os.environ.get("MCM_ARTF_STORE", Path.home() / ".mcm" / "artifacts"))

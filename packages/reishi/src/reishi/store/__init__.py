"""Module-level facade over a pluggable manifest StorageBackend.

Manifests are small JSON records keyed by (kind, name), reached through
save / load / load_all. The active backend is selected once from
MCM_STORE_BACKEND ('sqlite' default, 'fs' for one JSON file per manifest)
and can be replaced with use_backend(). Blobs referenced by manifest URIs
live under artifact_root(), not here.
"""

import os
from collections.abc import Mapping
from pathlib import Path

from reishi.store.base import StorageBackend, StoreError, safe_name
from reishi.store.filesystem import LocalFilesystemBackend
from reishi.store.sqlite import SqliteBackend

# Back-compat alias for the pre-split private name.
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
    # Lazy so importing reishi.store never creates ~/.mcm/store/store.db as a
    # side effect.
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
    # Remote backends expose no local root, so fall back to a placeholder.
    getter = getattr(_active(), "root", None)
    return getter() if getter else Path("<remote>")


def artifact_root() -> Path:
    """Root for locally-staged artifact blobs (adapters, checkpoints, datasets).

    Manifests reference these by URI; the blobs themselves live here (or at
    gs://, hf://), separate from the manifest store.
    """
    return Path(os.environ.get("MCM_ARTF_STORE", Path.home() / ".mcm" / "artifacts"))

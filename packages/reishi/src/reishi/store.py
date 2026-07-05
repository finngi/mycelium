"""Manifest store: a StorageBackend seam.

Everything mcm knows about trials, datasets, and boards is a manifest reached
through three calls -- save / load / load_all over a (kind, name) key. The
local-filesystem backend is the default; a Cloud SQL Postgres store of record
and any gs:// blob backend plug in behind the same Protocol via mcm.plugins, so
swapping the store never touches callers. The CLI stays a view over these
manifests, never over live state.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Protocol

# Manifest names become path components (local) or keys (remote); a name with a
# separator or traversal segment could escape the store root, so reject it.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class StoreError(Exception):
    pass


def _safe_name(name: str) -> str:
    if name in (".", "..") or not _SAFE_NAME.fullmatch(name):
        raise StoreError(f"unsafe manifest name: {name!r}")
    return name


class StorageBackend(Protocol):
    def save(self, kind: str, name: str, manifest: dict) -> None: ...
    def load(self, kind: str, name: str) -> dict: ...
    def load_all(self, kind: str, *, tolerant: bool = True) -> list[dict]: ...


class LocalFilesystemBackend:
    def root(self) -> Path:
        return Path(os.environ.get("MCM_STORE", Path.home() / ".mcm" / "store"))

    def _dir(self, kind: str) -> Path:
        d = self.root() / kind
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, kind: str, name: str, manifest: dict) -> None:
        path = self._dir(kind) / f"{_safe_name(name)}.json"
        # Write-then-rename: a crashed or concurrent write can never leave a
        # half-written manifest that load_all would choke on.
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, path)

    def load(self, kind: str, name: str) -> dict:
        path = self._dir(kind) / f"{_safe_name(name)}.json"
        if not path.exists():
            raise FileNotFoundError(f"no {kind} manifest named '{name}' in {self.root()}")
        return json.loads(path.read_text())

    def load_all(self, kind: str, *, tolerant: bool = True) -> list[dict]:
        out = []
        for p in sorted(self._dir(kind).glob("*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except (json.JSONDecodeError, OSError) as e:
                if not tolerant:
                    raise
                print(f"[WARN] skipping unreadable {kind} manifest {p.name}: {e}", file=sys.stderr)
        return out


_backend: StorageBackend = LocalFilesystemBackend()


def use_backend(backend: StorageBackend) -> None:
    global _backend
    _backend = backend


def save(kind: str, name: str, manifest: dict) -> None:
    _backend.save(kind, name, manifest)


def load(kind: str, name: str) -> dict:
    return _backend.load(kind, name)


def load_all(kind: str, *, tolerant: bool = True) -> list[dict]:
    return _backend.load_all(kind, tolerant=tolerant)


def root() -> Path:
    # Location of the local store; only meaningful for the filesystem backend
    # (the CLI `info` view). Remote backends report a placeholder.
    getter = getattr(_backend, "root", None)
    return getter() if getter else Path("<remote>")

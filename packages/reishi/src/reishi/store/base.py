"""Store contract shared by every manifest backend.

A manifest is small JSON keyed by (kind, name). Both halves become path
components (fs) or primary-key values (sqlite), so a value with a separator
or traversal segment could escape the store root -- reject it here once for
every backend rather than per-backend.
"""

import json
import re
from collections.abc import Mapping
from typing import Protocol

_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class StoreError(Exception):
    pass


def _safe(value: str, what: str) -> str:
    if value in (".", "..") or not _SAFE_KEY.fullmatch(value):
        raise StoreError(f"unsafe manifest {what}: {value!r}")
    return value


def safe_name(name: str) -> str:
    return _safe(name, "name")


def safe_kind(kind: str) -> str:
    return _safe(kind, "kind")


def dump_doc(manifest: Mapping[str, object]) -> str:
    """Serialise a manifest to the one on-disk form shared by every backend:
    indented, utf-8 (non-ASCII verbatim), one trailing newline. Byte-identical
    whichever backend writes it."""
    return json.dumps(dict(manifest), indent=2, ensure_ascii=False) + "\n"


class StorageBackend(Protocol):
    def save(self, kind: str, name: str, manifest: Mapping[str, object]) -> None: ...
    def load(self, kind: str, name: str) -> dict: ...
    def load_all(self, kind: str, *, tolerant: bool = True) -> list[dict]: ...

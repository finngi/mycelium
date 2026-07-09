"""Store contract shared by every manifest backend.

A manifest is small JSON keyed by (kind, name). Both halves become path
components (fs) or primary-key values (sqlite), so a value with a separator
or traversal segment could escape the store root -- reject it here once for
every backend rather than per-backend.
"""

import json
import re
from collections.abc import Iterator, Mapping
from typing import Protocol

_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_FILTER_KEY = re.compile(r"^[A-Za-z0-9_]+$")


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


def safe_filter_key(key: str) -> str:
    # query() filters are top-level equality only; a dotted/bracketed key
    # would silently ask a JSON-path-aware backend (sqlite) to traverse
    # nested fields the fs/default backends can never honor.
    if not _SAFE_FILTER_KEY.fullmatch(key):
        raise StoreError(f"unsafe query filter key: {key!r}")
    return key


def matches(manifest: Mapping[str, object], filters: Mapping[str, object]) -> bool:
    return all(manifest.get(k) == v for k, v in filters.items())


def dump_doc(manifest: Mapping[str, object]) -> str:
    """Serialise a manifest to the one on-disk form shared by every backend:
    indented, utf-8 (non-ASCII verbatim), one trailing newline. Byte-identical
    whichever backend writes it."""
    return json.dumps(dict(manifest), indent=2, ensure_ascii=False) + "\n"


class StorageBackend(Protocol):
    def save(self, kind: str, name: str, manifest: Mapping[str, object]) -> None: ...
    def load(self, kind: str, name: str) -> dict: ...
    def load_all(self, kind: str, *, tolerant: bool = True) -> list[dict]: ...

    def query(self, kind: str, **filters: object) -> list[dict]:
        """Reserved (docs/design/pr-plan.md): default falls back to load_all
        so a backend that only ever implemented the pre-existing three
        methods still satisfies the protocol unchanged. Subclass StorageBackend
        and override for a genuinely lazy/native filter."""
        for key in filters:
            safe_filter_key(key)
        return [m for m in self.load_all(kind) if matches(m, filters)]

    def stream(self, kind: str) -> Iterator[dict]:
        """Reserved (docs/design/pr-plan.md): default yields from load_all,
        i.e. still materializes everything first. Override for a backend
        where large trial sets must never be loaded all at once."""
        yield from self.load_all(kind)

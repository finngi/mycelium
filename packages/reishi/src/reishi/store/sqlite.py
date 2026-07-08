"""Default local manifest backend: one sqlite db, one table of JSON docs.

Same (kind, name) -> JSON contract as the filesystem backend, so callers and
manifests are byte-identical across the two. Chosen as the default because it
gives transactional writes, a single portable file, and the same relational
shape as the eventual Postgres store of record. The doc stays an opaque JSON
blob (not shredded into columns) so tolerant readers keep ignoring unknown keys
across versions -- the same forward/backward compatibility the JSON files had.

Not for oyster: a single binary db cannot be a git-as-queue (see filesystem.py).
"""

import json
import os
import sqlite3
import sys
from collections.abc import Mapping
from pathlib import Path

from reishi.store.base import dump_doc, safe_kind, safe_name


def _db_path() -> Path:
    # MCM_STORE always names a directory (its historical meaning, and what the
    # filesystem backend and oyster use); the db lives inside it. No dir-vs-file
    # ambiguity, so the resolved path never depends on what exists yet.
    root = Path(os.environ.get("MCM_STORE", Path.home() / ".mcm" / "store"))
    return root / "store.db"


class SqliteBackend:
    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else _db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self._path, isolation_level=None)  # autocommit
        self._db.execute("PRAGMA journal_mode=WAL")  # concurrent readers + one writer
        self._db.execute("PRAGMA busy_timeout=5000")  # wait, don't fail, on a locked db
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS manifests "
            "(kind TEXT NOT NULL, name TEXT NOT NULL, doc TEXT NOT NULL, "
            "PRIMARY KEY (kind, name))"
        )

    def root(self) -> Path:
        # The store directory, not the db file inside it: root() means "the
        # local store location" across all backends (the fs backend returns
        # MCM_STORE directly), and callers like oyster's gitstore run `git -C
        # root()` against it -- a file path there fails silently.
        return self._path.parent

    def save(self, kind: str, name: str, manifest: Mapping[str, object]) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO manifests(kind, name, doc) VALUES (?, ?, ?)",
            (safe_kind(kind), safe_name(name), dump_doc(manifest)),
        )

    def load(self, kind: str, name: str) -> dict:
        row = self._db.execute(
            "SELECT doc FROM manifests WHERE kind = ? AND name = ?",
            (safe_kind(kind), safe_name(name)),
        ).fetchone()
        if row is None:
            raise FileNotFoundError(f"no {kind} manifest named '{name}' in {self._path}")
        return json.loads(row[0])

    def load_all(self, kind: str, *, tolerant: bool = True) -> list[dict]:
        out = []
        for name, doc in self._db.execute(
            "SELECT name, doc FROM manifests WHERE kind = ? ORDER BY name", (safe_kind(kind),)
        ):
            try:
                out.append(json.loads(doc))
            except json.JSONDecodeError:
                if not tolerant:
                    raise
                print(f"[WARN] skipping unreadable {kind} manifest '{name}' in {self._path}",
                      file=sys.stderr)
        return out

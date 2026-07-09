"""One-JSON-file-per-manifest backend.

Kept for the executors whose mechanism is coupled to the physical layout --
oyster's git-as-queue makes a claim by committing a single changed file, which
only works with one file per manifest. It is no longer the default (sqlite is);
select it with MCM_STORE_BACKEND=fs or pin it explicitly via use_backend().
"""

import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from reishi.store.base import dump_doc, safe_kind, safe_name


class LocalFilesystemBackend:
    def root(self) -> Path:
        return Path(os.environ.get("MCM_STORE", Path.home() / ".mcm" / "store"))

    def _dir(self, kind: str) -> Path:
        d = self.root() / safe_kind(kind)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, kind: str, name: str, manifest: Mapping[str, object]) -> None:
        path = self._dir(kind) / f"{safe_name(name)}.json"
        # Write-then-rename so a crashed or concurrent write can never leave a
        # half-written manifest that load_all would choke on. mkstemp gives a
        # unique temp name per write: two writers to the same manifest in one
        # process would otherwise share a pid-derived name and clobber each
        # other's temp before os.replace.
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(dump_doc(manifest))
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def load(self, kind: str, name: str) -> dict:
        path = self._dir(kind) / f"{safe_name(name)}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no {kind} manifest named '{name}' in {self.root()}"
            )
        return json.loads(path.read_text())

    def load_all(self, kind: str, *, tolerant: bool = True) -> list[dict]:
        out = []
        for p in sorted(self._dir(kind).glob("*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                if not tolerant:
                    raise
                print(
                    f"[WARN] skipping unreadable {kind} manifest {p.name}: {e}",
                    file=sys.stderr,
                )
        return out

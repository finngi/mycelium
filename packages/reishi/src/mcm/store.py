"""Manifest store: local directory now, gs://xapien-mcm-train later.

Everything mcm knows about trials, datasets, and boards is a JSON manifest
under this root; the CLI is a view over these files, never over live state.
"""

import json
import os
from pathlib import Path


def root() -> Path:
    return Path(os.environ.get("MCM_STORE", Path.home() / ".mcm" / "store"))


def _dir(kind: str) -> Path:
    d = root() / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(kind: str, name: str, manifest: dict) -> Path:
    path = _dir(kind) / f"{name}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return path


def load(kind: str, name: str) -> dict:
    path = _dir(kind) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no {kind} manifest named '{name}' in {root()}")
    return json.loads(path.read_text())


def load_all(kind: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(_dir(kind).glob("*.json"))]

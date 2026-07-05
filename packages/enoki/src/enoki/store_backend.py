"""Postgres StorageBackend: the store of record once trials run in the
cluster instead of on a laptop.

Kept out of mcm core so the contract layer stays free of database clients --
enoki is the one place allowed to know Postgres exists, same rule that
already applies to Ray. The driver swaps this in via reishi.store.use_backend()
before it plans or claims any trial; mcm's primitives never change.

One JSONB table for every kind (trials/datasets/recipes): the manifest stays
exactly the shape mcm already defines, and Postgres adds the transactional
and concurrent-claim guarantees a directory of files can't.
"""

import json
import os

_DDL = """
CREATE TABLE IF NOT EXISTS manifests (
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    manifest JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (kind, name)
);
"""


class PostgresBackend:
    def __init__(self, dsn: str):
        import psycopg  # lazy: only paid for when this backend is actually selected

        self._psycopg = psycopg
        self._dsn = dsn
        with self._connect() as conn:
            conn.execute(_DDL)

    @classmethod
    def from_env(cls, var: str = "MCM_PG_DSN") -> "PostgresBackend":
        dsn = os.environ.get(var)
        if not dsn:
            raise RuntimeError(f"{var} is not set; cannot construct a PostgresBackend")
        return cls(dsn)

    def _connect(self):
        return self._psycopg.connect(self._dsn, autocommit=True)

    def save(self, kind: str, name: str, manifest: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO manifests (kind, name, manifest, updated_at) "
                "VALUES (%s, %s, %s, now()) "
                "ON CONFLICT (kind, name) DO UPDATE "
                "SET manifest = EXCLUDED.manifest, updated_at = now()",
                (kind, name, json.dumps(manifest)),
            )

    def load(self, kind: str, name: str) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT manifest FROM manifests WHERE kind = %s AND name = %s",
                (kind, name),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"no {kind} manifest named '{name}' in Postgres store")
        return row[0]

    def load_all(self, kind: str, *, tolerant: bool = True) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT manifest FROM manifests WHERE kind = %s ORDER BY name",
                (kind,),
            ).fetchall()
        return [r[0] for r in rows]

    def claim_next(self, kind: str, claimant: str) -> dict | None:
        """Atomically hand one 'planned' manifest to `claimant`, highest
        recipe.priority first, ties broken by name. SKIP LOCKED means a
        second enoki pod racing for the same row sees the next candidate
        instead of blocking on or double-claiming this one.

        The SELECT and UPDATE must share one transaction: on an autocommit
        connection each execute() is its own implicit transaction, so a
        FOR UPDATE lock taken by the SELECT is released the instant it
        returns -- before this method ever reaches the UPDATE -- and a
        second claimant sees the same 'planned' row as still free. Verified
        directly against Postgres; the two-statement version double-claims.
        """
        with self._connect() as conn, conn.transaction():
            row = conn.execute(
                "SELECT name, manifest FROM manifests "
                "WHERE kind = %s AND manifest->>'status' = 'planned' "
                "ORDER BY COALESCE((manifest #>> '{spec,priority}')::int, 0) DESC, name ASC "
                "FOR UPDATE SKIP LOCKED LIMIT 1",
                (kind,),
            ).fetchone()
            if row is None:
                return None
            name, manifest = row
            manifest["status"] = "running"
            manifest.setdefault("execution", {})["runner"] = claimant
            conn.execute(
                "UPDATE manifests SET manifest = %s, updated_at = now() "
                "WHERE kind = %s AND name = %s",
                (json.dumps(manifest), kind, name),
            )
            return manifest

"""Live integration tests for PostgresBackend against a real Postgres.

Needs a reachable database: set MCM_PG_TEST_DSN, or these tests are skipped.
Local dev: `brew install postgresql@16 && brew services start postgresql@16`,
then `createdb mcm_enoki_test` and export
MCM_PG_TEST_DSN=postgresql:///mcm_enoki_test
"""

import os
import threading

import pytest

psycopg = pytest.importorskip("psycopg")

from enoki.store_backend import PostgresBackend  # noqa: E402

DSN = os.environ.get("MCM_PG_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="MCM_PG_TEST_DSN not set; no live Postgres to test against")


@pytest.fixture
def backend():
    b = PostgresBackend(DSN)
    with b._connect() as conn:
        conn.execute("TRUNCATE manifests")
    return b


def test_save_load_roundtrip(backend):
    backend.save("trials", "t1", {"id": "t1", "status": "planned", "priority": 0})
    assert backend.load("trials", "t1")["status"] == "planned"


def test_load_missing_raises(backend):
    with pytest.raises(FileNotFoundError):
        backend.load("trials", "nope")


def test_save_upserts_existing_name(backend):
    backend.save("trials", "t1", {"id": "t1", "status": "planned"})
    backend.save("trials", "t1", {"id": "t1", "status": "done"})
    assert backend.load("trials", "t1")["status"] == "done"


def test_load_all_scoped_to_kind_and_ordered(backend):
    backend.save("trials", "b", {"id": "b"})
    backend.save("trials", "a", {"id": "a"})
    backend.save("datasets", "z", {"id": "z"})
    assert [m["id"] for m in backend.load_all("trials")] == ["a", "b"]


def test_claim_next_returns_none_when_nothing_planned(backend):
    backend.save("trials", "t1", {"id": "t1", "status": "done"})
    assert backend.claim_next("trials", "runner-a") is None


def test_claim_next_picks_highest_priority(backend):
    backend.save("trials", "low", {"id": "low", "status": "planned", "spec": {"priority": 0}})
    backend.save("trials", "high", {"id": "high", "status": "planned", "spec": {"priority": 5}})
    claimed = backend.claim_next("trials", "runner-a")
    assert claimed["id"] == "high"
    assert claimed["status"] == "running"
    assert claimed["execution"]["runner"] == "runner-a"
    # It's gone from the planned pool now.
    assert backend.claim_next("trials", "runner-b")["id"] == "low"


def test_claim_next_ties_broken_by_name(backend):
    backend.save("trials", "b", {"id": "b", "status": "planned"})
    backend.save("trials", "a", {"id": "a", "status": "planned"})
    assert backend.claim_next("trials", "runner-a")["id"] == "a"


def test_claim_next_is_exclusive_under_concurrency(backend):
    """The whole reason claim_next exists: two racing claimants must never
    both walk away with the same trial. A barrier forces all threads to
    call claim_next at the same instant instead of hoping the OS scheduler
    happens to interleave them -- an unsynchronized race can pass by luck
    even when the underlying locking is broken (it did, once, here)."""
    backend.save("trials", "only", {"id": "only", "status": "planned"})

    results = []
    lock = threading.Lock()
    barrier = threading.Barrier(10)

    def race(n):
        b = PostgresBackend(DSN)
        barrier.wait()
        result = b.claim_next("trials", f"runner-{n}")
        if result is not None:
            with lock:
                results.append(n)

    threads = [threading.Thread(target=race, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 1, f"expected exactly one winner, got {results}"

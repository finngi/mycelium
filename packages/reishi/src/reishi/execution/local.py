"""Local executor: runs trials sequentially, in-process, one producer call
per trial. No queue, no claim/heartbeat -- see oyster (mesh) and enoki
(KubeRay) for executors that need that machinery; this one is for a single
laptop running its own recipe.
"""

import socket
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from reishi import store
from reishi.execution.contract import Producer
from reishi.primitives import trial as trial_store
from reishi.primitives.trial import Trial


class _Tee:
    """Forwards writes to every stream in `streams` -- output still reaches
    the real stdout/stderr while also landing in the trial's log file."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log_root() -> Path | None:
    # A backend without a filesystem root (e.g. Postgres) makes store.root()
    # return the REMOTE_ROOT sentinel; mkdir would happily create a literal
    # "<remote>/logs" directory, so treat it as "logs not capturable here".
    root = store.root()
    return None if root == store.REMOTE_ROOT else root


def _log_path(log_root: Path, trial_id: str) -> Path:
    path = log_root / "logs" / f"{trial_id}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def execute(trials: list[Trial], producer: Producer) -> int:
    runner = f"local:{socket.gethostname()}"
    exit_code = 0
    log_root = _log_root()
    if log_root is None and trials:
        print(
            "[WARN] active store backend has no filesystem root -> "
            "trial logs not captured",
            file=sys.stderr,
        )

    for t in trials:
        t.status = "running"
        t.execution["runner"] = runner
        t.execution["claimed_at"] = _now()
        t.execution["attempt"] = t.execution.get("attempt", 0) + 1
        log_path = _log_path(log_root, t.id) if log_root is not None else None
        if log_path is not None:
            t.execution["log"] = str(log_path)
        # Saved before the producer runs: a crash mid-trial must leave a
        # visible running-orphan on disk, never a silent gap.
        trial_store.save(t)

        if log_path is not None:
            with log_path.open("a") as log_file:
                log_file.write(f"--- attempt {t.execution['attempt']} @ {_now()} ---\n")

        start = time.monotonic()
        try:
            if log_path is not None:
                with log_path.open("a") as log_file:
                    with (
                        redirect_stdout(_Tee(sys.stdout, log_file)),
                        redirect_stderr(_Tee(sys.stderr, log_file)),
                    ):
                        result = producer(t.to_manifest())
            else:
                result = producer(t.to_manifest())
            # Entry-point producers carry no runtime schema; a malformed
            # result must fail THIS trial, not crash the batch below.
            if not isinstance(result, dict) or not {"metrics", "artifacts"} <= set(
                result
            ):
                got = (
                    sorted(result)
                    if isinstance(result, dict)
                    else type(result).__name__
                )
                raise ValueError(
                    f"producer returned an invalid ProducerResult (got: {got}; "
                    "needs 'metrics' and 'artifacts')"
                )
        except Exception as e:
            if log_path is not None:
                with log_path.open("a") as log_file:
                    log_file.write(traceback.format_exc())
            t.status = "failed"
            t.execution["last_error"] = f"{type(e).__name__}: {e}"
            t.execution["finished_at"] = _now()
            trial_store.save(t)
            exit_code = 1
            print(
                f"[FAIL] {t.id} failed: {type(e).__name__}: {e} -> "
                f"see mcm logs {t.id[:8]}",
                file=sys.stderr,
            )
            continue

        wall_time_s = round(time.monotonic() - start, 3)
        t.metrics.update(result["metrics"])
        t.artifacts = result["artifacts"]
        t.observables.update(result.get("observables", {}))
        # The executor's outer clock is authoritative for run cost: this
        # overwrites any wall_time_s the producer itself reported.
        t.observables["wall_time_s"] = wall_time_s
        t.status = "done"
        t.execution["finished_at"] = _now()
        trial_store.save(t)
        print(f"[OK] {t.id} done ({wall_time_s}s)", file=sys.stderr)

    return exit_code

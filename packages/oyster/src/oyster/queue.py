"""The scheduler: which trial does this machine run next, and who owns what.

All state is trial manifests in the mcm store; every transition here is a
manifest edit published as a commit. Ordering is (priority desc, created
asc, id) -- priority jumps the queue, ties are FIFO.
"""

import sys
from datetime import datetime, timezone

import mcm.tasks  # noqa: F401  (populate the task registry)
from mcm.primitives.trial import Trial
from mcm.primitives import trial as trial_store

from oyster import footprint, gitstore

MAX_ATTEMPTS = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fits(t: Trial, budget_gb: float, accelerators: set[str]) -> tuple[bool, str]:
    """Can THIS machine run this trial? Returns (ok, reason-if-not)."""
    if t.status != "planned":
        return False, f"status is {t.status}"
    if t.spec.get("accelerator") not in accelerators:
        return False, f"needs {t.spec.get('accelerator')}, this machine runs {sorted(accelerators) or 'nothing'}"
    if t.execution.get("attempt", 0) >= MAX_ATTEMPTS:
        return False, f"exhausted {MAX_ATTEMPTS} attempts"
    est = footprint.estimate_gb(t.spec)
    if est > budget_gb:
        return False, f"est. {est:.1f}GB > budget {budget_gb:.0f}GB"
    return True, ""


def eligible(budget_gb: float, accelerators: set[str]) -> list[Trial]:
    """Claimable trials for this machine, best-first."""
    ts = [t for t in trial_store.load_all() if fits(t, budget_gb, accelerators)[0]]
    return sorted(ts, key=lambda t: (-t.spec.get("priority", 0), t.created, t.id))


def claim(t: Trial, runner: str) -> Trial | None:
    """Atomically take ownership. Returns the claimed trial (work with THIS
    object, not the pre-claim one), or None: lost the race, re-pick."""
    gitstore.sync()
    current = trial_store.load(t.id)
    if current.status != "planned":
        return None
    current.status = "running"
    current.execution = {
        "runner": runner,
        "claimed_at": _now(),
        "heartbeat": _now(),
        "attempt": current.execution.get("attempt", 0) + 1,
    }
    trial_store.save(current)
    if not gitstore.publish(f"claim {current.id} on {runner}"):
        return None
    return current


def heartbeat(t: Trial) -> None:
    t.execution["heartbeat"] = _now()
    trial_store.save(t)
    gitstore.publish(f"heartbeat {t.id}")


def finish(t: Trial, metrics: dict, artifacts: dict) -> None:
    t.status, t.metrics, t.artifacts = "done", metrics, artifacts
    t.execution["finished_at"] = _now()
    trial_store.save(t)
    gitstore.publish(f"done {t.id}: {metrics}")


def fail(t: Trial, error: str) -> None:
    # Back to planned while attempts remain -- the mesh retries elsewhere.
    attempt = t.execution.get("attempt", 0)
    t.execution["last_error"] = error
    t.status = "failed" if attempt >= MAX_ATTEMPTS else "planned"
    trial_store.save(t)
    gitstore.publish(f"fail {t.id} (attempt {attempt}): {error[:80]}")


def requeue(trial_id: str) -> Trial:
    """Yank a trial back to planned (load retraction / manual reassignment)."""
    t = trial_store.resolve(trial_id)
    t.status = "planned"
    t.execution.pop("runner", None)
    t.execution.pop("heartbeat", None)
    trial_store.save(t)
    gitstore.publish(f"requeue {t.id}")
    return t


def requeue_stale(timeout_min: float) -> list[str]:
    """Reaper: a running trial whose heartbeat went stale is a dead runner
    (closed lid, killed service). Requeue while attempts remain, else fail."""
    gitstore.sync()
    now = datetime.now(timezone.utc)
    actions = []
    for t in trial_store.load_all():
        if t.status != "running":
            continue
        hb = t.execution.get("heartbeat") or t.execution.get("claimed_at")
        if hb and (now - datetime.fromisoformat(hb)).total_seconds() < timeout_min * 60:
            continue
        runner = t.execution.get("runner", "unknown")
        if t.execution.get("attempt", 0) >= MAX_ATTEMPTS:
            t.status = "failed"
            t.execution["last_error"] = f"lost on {runner}, attempts exhausted"
            actions.append(f"[FAIL] {t.id}: lost on {runner}, {MAX_ATTEMPTS} attempts used")
        else:
            t.status = "planned"
            t.execution.pop("runner", None)
            actions.append(f"[REQUEUE] {t.id}: heartbeat stale on {runner}")
        trial_store.save(t)
    if actions:
        gitstore.publish(f"reaper: {len(actions)} stale trial(s)")
        for a in actions:
            print(a, file=sys.stderr)
    return actions

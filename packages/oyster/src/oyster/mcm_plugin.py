"""mcm plugin: the `mesh` domain. With oyster installed, the one mcm CLI grows
mesh vocabulary -- same grammar, same canonical echo, same -o json.

    mcm mesh                 # queue counts + this machine's fit (read-only default)
    mcm next                 # > mcm mesh next     what this machine would claim
    mcm work [--max N]       # > mcm mesh work     claim-and-train loop
    mcm drain / undrain      # > mcm mesh drain    retract/rejoin this machine
    mcm requeue <trial-id>   # > mcm mesh requeue  yank a trial back to planned
    mcm reap [--timeout-min] # > mcm mesh reap     requeue stale-heartbeat trials
"""

import os
import sys
from pathlib import Path

from reishi.cli.grammar import Command, Verb
from reishi.cli.output import emit

from oyster import footprint, gitstore, machine, queue, trainers, worker

DOMAINS = ("mesh",)
VERBS = (
    Verb("work", home="mesh", readonly=False),
    Verb("drain", home="mesh", readonly=False),
    Verb("undrain", home="mesh", readonly=False),
    Verb("requeue", home="mesh", readonly=False),
    Verb("reap", home="mesh", readonly=False),
    Verb("next", home="mesh", readonly=True),
)


def _flag_value(flags: list[str], name: str) -> str | None:
    if name in flags:
        i = flags.index(name)
        if i + 1 < len(flags):
            return flags[i + 1]
    return None


def mesh_status(cmd: Command) -> int:
    from reishi.primitives import trial as trial_store

    gitstore.sync()
    by_status: dict[str, int] = {}
    for t in trial_store.load_all():
        by_status[t.status] = by_status.get(t.status, 0) + 1
    emit({
        "trials": by_status,
        "machine": machine.name(),
        "mem_budget_gb": machine.mem_budget_gb(),
        "busy": machine.is_busy(),
        "trainers": sorted(trainers.supported()) or None,
    }, cmd.flags)
    return 0


def mesh_next(cmd: Command) -> int:
    gitstore.sync()
    accelerators = trainers.supported()
    if not accelerators:
        # Claiming needs a trainer, but the dry run stays useful without one:
        # show the mlx queue rather than hiding it behind an empty registry.
        print("[WARN] no trainer installed -> showing the mlx queue; nothing is actually claimable here",
              file=sys.stderr)
        accelerators = {"mlx"}
    cands = queue.eligible(machine.mem_budget_gb(), accelerators)
    rows = [
        {"id": t.id, "priority": t.spec.get("priority", 0),
         "est_gb": round(footprint.estimate_gb(t.spec), 1), "created": t.created}
        for t in cands[:10]
    ]
    emit(rows, cmd.flags, columns=["id", "priority", "est_gb", "created"])
    return 0


def mesh_work(cmd: Command) -> int:
    max_trials = _flag_value(cmd.flags, "--max")
    return worker.run(max_trials=int(max_trials) if max_trials else None)


def mesh_drain(cmd: Command) -> int:
    # Label first (assignment level: no job lands here at all), busy file
    # second (execution level: catches jobs assigned before the label came off).
    if machine.set_ready(False):
        print("[OK] `ready` label removed -> no new jobs will be assigned here", file=sys.stderr)
    else:
        print("[WARN] could not toggle `ready` label (gh/repo/runner unreachable) "
              "-> relying on the busy-file gate only", file=sys.stderr)
    machine.BUSY_FILE.touch()
    print(f"[OK] {machine.BUSY_FILE} created -> in-flight worker stops after the current trial",
          file=sys.stderr)
    return 0


def mesh_undrain(cmd: Command) -> int:
    machine.BUSY_FILE.unlink(missing_ok=True)
    if machine.set_ready(True):
        print("[OK] `ready` label restored -> job assignment re-enabled", file=sys.stderr)
    else:
        print("[WARN] could not restore `ready` label (gh/repo/runner unreachable) "
              "-> restore it manually or set OYSTER_REPO", file=sys.stderr)
    print(f"[OK] {machine.BUSY_FILE} removed -> claiming re-enabled", file=sys.stderr)
    return 0


def mesh_requeue(cmd: Command) -> int:
    if not cmd.objects:
        print("[FAIL] mcm mesh requeue needs a trial id (prefix ok)", file=sys.stderr)
        return 1
    t = queue.requeue(cmd.objects[0])
    print(f"[OK] {t.id} -> planned (attempt {t.execution.get('attempt', 0)} "
          f"of {queue.MAX_ATTEMPTS})", file=sys.stderr)
    return 0


def mesh_reap(cmd: Command) -> int:
    timeout = float(_flag_value(cmd.flags, "--timeout-min") or 90.0)
    actions = queue.requeue_stale(timeout_min=timeout)
    print(f"[OK] {len(actions)} stale trial(s) handled", file=sys.stderr)
    return 0


HANDLERS = {
    ("mesh", "list"): mesh_status,
    ("mesh", "next"): mesh_next,
    ("mesh", "work"): mesh_work,
    ("mesh", "drain"): mesh_drain,
    ("mesh", "undrain"): mesh_undrain,
    ("mesh", "requeue"): mesh_requeue,
    ("mesh", "reap"): mesh_reap,
}

# Inside an oyster checkout the committed store/ is the queue; an explicit
# MCM_STORE anywhere else is honored untouched.
if "MCM_STORE" not in os.environ and (Path.cwd() / "store" / "trials").is_dir():
    os.environ["MCM_STORE"] = str(Path.cwd() / "store")

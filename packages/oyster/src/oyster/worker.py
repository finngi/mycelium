"""Claim -> train -> record, until drained or nothing fits.

One trial at a time: on a single unified-memory GPU two trainings fight for
the same VRAM and compute, so serial is faster than parallel AND avoids the
OOM. The drain check sits between trials, so `mcm drain` retracts a machine
gracefully: current trial finishes, nothing more claimed.
"""

import sys
import traceback

from oyster import gitstore, machine, queue, trainers


def run(max_trials: int | None = None) -> int:
    if machine.training_process_running():
        print(
            "[FAIL] a training process is already running here -> refusing to start",
            file=sys.stderr,
        )
        return 1

    done = 0
    while max_trials is None or done < max_trials:
        if machine.is_busy():
            print(
                f"[INFO] drained ({machine.BUSY_FILE} present) -> stopping after {done} trial(s)",
                file=sys.stderr,
            )
            break

        gitstore.sync()
        cands = queue.eligible(machine.mem_budget_gb(), trainers.supported())
        if not cands:
            print(
                f"[INFO] nothing claimable for this machine -> done ({done} trial(s))",
                file=sys.stderr,
            )
            break

        t = queue.claim(cands[0], machine.name())
        if t is None:
            continue  # lost the race; re-sync and re-pick

        spec_line = f"{t.id} ({t.spec.get('base_model') or 'from-scratch'}, prio {t.spec.get('priority', 0)})"
        print(f"[RUN] {spec_line}", file=sys.stderr)
        try:
            trainer = trainers.get(t.spec["accelerator"])
            result = trainer(t.to_manifest())
            queue.finish(t, result.get("metrics", {}), result.get("artifacts", {}))
            print(f"[OK] {t.id}", file=sys.stderr)
        except Exception as e:
            queue.fail(t, f"{e}\n{traceback.format_exc()}")
            print(f"[FAIL] {t.id}: {e}", file=sys.stderr)
        done += 1
    return 0

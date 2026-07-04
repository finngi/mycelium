"""Trainers, keyed by accelerator ('trainers' not 'adapters' -- an adapter
is the LoRA artifact a trial produces).

A trainer is a callable: (trial_manifest: dict) -> {"metrics": ..., "artifacts": ...}.
It must call mat.queue.heartbeat(trial) periodically or the reaper will
requeue the trial mid-run. The MLX LoRA trainer lands here; until then the
registry is empty, so workers claim nothing.
"""

TRAINERS: dict[str, object] = {}


def supported() -> set[str]:
    return set(TRAINERS)


def get(accelerator: str):
    if accelerator not in TRAINERS:
        known = ", ".join(sorted(TRAINERS)) or "none yet"
        raise KeyError(f"no trainer for '{accelerator}' (installed: {known})")
    return TRAINERS[accelerator]

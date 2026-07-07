"""Trainers, keyed by accelerator ('trainers' not 'adapters' -- an adapter
is the LoRA artifact a trial produces).

A trainer is a callable: (trial_manifest: TrialManifest) -> TrainerResult.
It must call oyster.queue.heartbeat(trial) periodically or the reaper will
requeue the trial mid-run.
"""

import sys

from oyster.trainers.contract import Trainer

TRAINERS: dict[str, Trainer] = {}

try:
    from oyster.trainers.mlx_lora import train as _mlx_train

    TRAINERS["mlx"] = _mlx_train
except ImportError as e:
    # Only swallow the absence of the optional mlx/mlx-lm stack itself (darwin+arm64
    # only, per pyproject platform marker) -- a real import bug inside mlx_lora.py
    # should surface loudly, not silently shrink the trainer registry.
    if not (e.name or "").startswith("mlx"):
        raise
    print(f"[WARN] mlx trainer unavailable ({e}) -> this machine can't claim mlx trials", file=sys.stderr)


def supported() -> set[str]:
    return set(TRAINERS)


def get(accelerator: str) -> Trainer:
    if accelerator not in TRAINERS:
        known = ", ".join(sorted(TRAINERS)) or "none yet"
        raise KeyError(f"no trainer for '{accelerator}' (installed: {known})")
    return TRAINERS[accelerator]

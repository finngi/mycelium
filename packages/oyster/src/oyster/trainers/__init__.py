"""Producers, keyed by the recipe's runtime ('trainers' not 'adapters' -- an
adapter is the LoRA artifact a trial produces).

A producer is a callable: (trial_manifest: TrialManifest) -> ProducerResult.
oyster's own worker calls oyster.queue.heartbeat(trial) periodically while a
producer runs, or the reaper will requeue the trial mid-run -- that
heartbeating is oyster's queue requirement, not part of the Producer contract
itself.
"""

import sys

from oyster.trainers.contract import Producer

TRAINERS: dict[str, Producer] = {}

try:
    from oyster.trainers.mlx_lora import train as _mlx_train

    TRAINERS["mlx"] = _mlx_train
except ImportError as e:
    # Only swallow the absence of the optional mlx/mlx-lm stack itself (darwin+arm64
    # only, per pyproject platform marker) -- a real import bug inside mlx_lora.py
    # should surface loudly, not silently shrink the trainer registry.
    if not (e.name or "").startswith("mlx"):
        raise
    print(
        f"[WARN] mlx trainer unavailable ({e}) -> this machine can't claim mlx trials",
        file=sys.stderr,
    )


def supported() -> set[str]:
    return set(TRAINERS)


def get(runtime: str) -> Producer:
    if runtime not in TRAINERS:
        known = ", ".join(sorted(TRAINERS)) or "none yet"
        raise KeyError(f"no trainer for '{runtime}' (installed: {known})")
    return TRAINERS[runtime]

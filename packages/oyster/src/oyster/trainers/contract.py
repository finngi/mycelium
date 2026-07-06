"""Trainer-contract types, kept out of __init__.py to avoid a cycle: __init__
imports the trainer modules (mlx_lora.py, ...) that also need these types."""

from collections.abc import Callable
from typing import TypedDict

from reishi.primitives.trial import TrialArtifacts, TrialManifest


class TrainerResult(TypedDict):
    metrics: dict  # task-specific; nameparse-style tasks land Task.aggregate's AggregateMetrics here
    artifacts: TrialArtifacts


Trainer = Callable[[TrialManifest], TrainerResult]

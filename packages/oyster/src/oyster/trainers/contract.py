"""Trainer-contract types, kept out of __init__.py to avoid a cycle: __init__
imports the trainer modules (mlx_lora.py, ...) that also need these types."""

from collections.abc import Callable
from typing import Any, TypedDict

from reishi.primitives.trial import TrialArtifacts, TrialManifest


class TrainerResult(TypedDict):
    metrics: dict[
        str, Any
    ]  # shape is task-defined, so it stays untyped rather than a fixed schema
    artifacts: TrialArtifacts


Trainer = Callable[[TrialManifest], TrainerResult]

"""Trainer-contract types.

Duplicated in oyster's trainers/contract.py: the executors don't depend on each
other, so each keeps its own copy of the contract that Trial's fields imply.
"""

from collections.abc import Callable
from typing import TypedDict

from reishi.primitives.trial import TrialArtifacts, TrialManifest


class TrainerResult(TypedDict):
    metrics: dict  # task-specific; field-scored tasks store AggregateMetrics here
    artifacts: TrialArtifacts


Trainer = Callable[[TrialManifest], TrainerResult]

"""Trainer-contract types.

Mirrors oyster's oyster/trainers/contract.py -- enoki and oyster don't
depend on each other (reishi has no dependency on either sibling repo,
and the siblings don't depend on one another), so each executor keeps its
own copy of the contract reishi.primitives.trial's Trial fields imply.
"""

from collections.abc import Callable
from typing import TypedDict

from reishi.primitives.trial import TrialArtifacts, TrialManifest


class TrainerResult(TypedDict):
    metrics: dict  # task-specific; nameparse-style tasks land Task.aggregate's AggregateMetrics here
    artifacts: TrialArtifacts


Trainer = Callable[[TrialManifest], TrainerResult]

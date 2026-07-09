"""A registered fixture task for the driver tests.

reishi ships no tasks and no deployment is installed in enoki's test env, so
load_tasks() finds nothing here -- the driver is task-agnostic anyway (these
tests fake the trainer). This registers a trivial task so the recipe's task
name resolves, without coupling enoki to any real experiment's task.
"""

from reishi.primitives.task import ScoreCounts, Task, register


def _fixture_score(pred: dict, ref: dict) -> ScoreCounts:
    cand = set((pred or {}).items())
    truth = set((ref or {}).items())
    return {
        "tp": len(cand & truth),
        "fp": len(cand - truth),
        "fn": len(truth - cand),
        "exact_match": pred == ref,
        "invalid": not pred,
    }


register(
    Task(
        name="fixture",
        description="driver-contract test fixture; not a real task",
        output_fields=("x",),
        score=_fixture_score,
    )
)

"""A registered fixture task for the scheduler tests.

reishi ships no tasks and no deployment is installed in oyster's test env, so
load_tasks() finds nothing here -- but Recipe.validate() (called by trial.plan)
requires the recipe's task to resolve. The scheduler is task-agnostic (these
tests fake the trainer), so this registers a trivial task purely so the recipe
validates, without coupling oyster to any real experiment's task.
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
        description="scheduler-contract test fixture; not a real task",
        output_fields=("x",),
        score=_fixture_score,
    )
)

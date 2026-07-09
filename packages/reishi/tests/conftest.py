"""A registered fixture task for the contract tests.

reishi ships no tasks of its own, but Recipe.validate() checks the task name
against the registry, so the primitive tests need some task registered to
resolve against. This provides a trivial one.
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
        description="contract-test fixture; not a real task",
        output_fields=("x",),
        score=_fixture_score,
    )
)

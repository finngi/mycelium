"""A registered fixture task for the contract tests.

reishi ships no tasks of its own, but Recipe.validate() checks the task name
against the registry, so the primitive tests need some task registered to
resolve against. This provides a trivial one.
"""

from reishi.primitives.task import ScoreCounts, Task, register


def _fixture_score(pred: dict, gold: dict) -> ScoreCounts:
    cand = set((pred or {}).items())
    ref = set((gold or {}).items())
    return {
        "tp": len(cand & ref),
        "fp": len(cand - ref),
        "fn": len(ref - cand),
        "exact_match": pred == gold,
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

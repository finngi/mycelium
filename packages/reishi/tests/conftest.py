"""A registered fixture task for the contract tests.

reishi ships no tasks -- real ones live in deployment workspaces (e.g. mcm-lab).
But the primitive tests need *some* registered task to resolve, since
Recipe.validate() checks the name against the registry. This registers a trivial
one so the contract tests exercise recipe/trial/experiment_submit without
reishi depending on any real experiment's task.
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

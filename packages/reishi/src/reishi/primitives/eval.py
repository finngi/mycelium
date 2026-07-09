"""Eval loop: render prompt -> generate -> decode -> score -> aggregate.

generate is injected as a callback, so this module imports no accelerator
libraries and runs no inference itself. Rows are read only as x/y/key and the
task only through the Scorable protocol, so nothing here is bound to a concrete
row or task type.
"""

from collections.abc import Iterable, Mapping
from typing import Any, Callable, Literal, Protocol, runtime_checkable

GenerateFn = Callable[[str], str]  # prompt -> raw text; injected by executor / replay
Placement = Literal["cpu", "accelerator", "local"]


@runtime_checkable
class Scorable(Protocol):
    """The subset of Task that run_eval actually calls."""

    def decode(self, raw: str) -> Any: ...
    def score(self, pred: Any, ref: Any) -> Mapping[str, object]: ...
    def aggregate(self, scores: list[Mapping[str, object]]) -> dict: ...


def _render(prompt: str | None, x: Any) -> str:
    if prompt is None:
        return x if isinstance(x, str) else str(x)
    # `{x}` lets a template place the input; otherwise treat prompt as a prefix.
    # Targeted replace, not str.format: a prompt may carry other literal braces
    # (a JSON example), which .format would choke on.
    return prompt.replace("{x}", str(x)) if "{x}" in prompt else prompt + str(x)


def run_eval(
    *,
    task: Scorable,
    rows: Iterable[Mapping],
    generate: GenerateFn,
    prompt: str | None = None,
    sink: Callable[[Mapping], None] | None = None,
) -> dict:
    if not callable(getattr(task, "score", None)):
        raise ValueError(
            "task has no scorer (Task.score is None) -- nothing to run_eval"
        )
    scores: list[Mapping[str, object]] = []
    for row in rows:
        raw = generate(_render(prompt, row["x"]))
        if sink:
            # Persist in the same schema run_eval consumes (key/x/y/raw) so a
            # predictions artifact is directly replayable by rescore(), no adapter.
            sink({"key": row["key"], "x": row["x"], "y": row["y"], "raw": raw})
        scores.append(task.score(task.decode(raw), row["y"]))
    return task.aggregate(scores)


def rescore(
    *,
    task: Scorable,
    rows_with_raw: Iterable[Mapping],
    prompt: str | None = None,
) -> dict:
    """Re-score persisted outputs without re-running inference.

    Each row already carries its model output under `raw`; replaying it lets a
    scorer change (new metric, fixed bug) rescore a trial for free, with no GPU.
    The replayed `generate` ignores its prompt and returns the row's stored raw.
    """
    rows = list(rows_with_raw)  # materialize: replay and the loop share these rows
    replay = iter(row["raw"] for row in rows)
    return run_eval(
        task=task,
        rows=rows,
        generate=lambda _prompt: next(replay),
        prompt=prompt,
    )

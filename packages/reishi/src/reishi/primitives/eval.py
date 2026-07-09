"""Eval loop: render prompt -> generate -> decode -> score -> aggregate.

generate is injected as a callback, so this module imports no accelerator
libraries and runs no inference itself. Rows are read only as x/y/key and the
task only through the Scorable protocol, so nothing here is bound to a concrete
row or task type.
"""

from collections.abc import Iterable, Mapping
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from reishi.primitives.trial import EvalInfo

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
    task_name: str | None = None,
    codec: str | None = None,
    scorer_version: str | None = None,
    dataset_ref: str | None = None,
    dataset_revision: str | None = None,
    split: str | None = None,
) -> tuple[dict, EvalInfo]:
    """Run the eval loop; return (aggregate metrics, measurement-key info).

    The K-pinning keyword-only args (task_name, codec, scorer_version,
    dataset_ref, dataset_revision, split) are optional and caller-supplied:
    run_eval only sees rows and a Scorable, so it can't know a Task's name or
    a Dataset's ref itself -- a caller that has those objects passes the
    values through; one it doesn't know is simply left absent. eval_n is the
    one K field run_eval always knows, since it counts the rows it scores.
    """
    if not callable(getattr(task, "score", None)):
        raise ValueError(
            "task has no scorer (Task.score is None) -- nothing to run_eval"
        )
    scores: list[Mapping[str, object]] = []
    eval_n = 0
    for row in rows:
        raw = generate(_render(prompt, row["x"]))
        if sink:
            # Persist in the same schema run_eval consumes (key/x/y/raw) so a
            # predictions artifact is directly replayable by rescore(), no adapter.
            sink({"key": row["key"], "x": row["x"], "y": row["y"], "raw": raw})
        scores.append(task.score(task.decode(raw), row["y"]))
        eval_n += 1
    info: EvalInfo = {"eval_n": eval_n}
    if task_name is not None:
        info["task"] = task_name
    if codec is not None:
        info["codec"] = codec
    if scorer_version is not None:
        info["scorer_version"] = scorer_version
    if dataset_ref is not None:
        info["dataset"] = dataset_ref
    if dataset_revision is not None:
        info["dataset_revision"] = dataset_revision
    if split is not None:
        info["split"] = split
    return task.aggregate(scores), info


def rescore(
    *,
    task: Scorable,
    rows_with_raw: Iterable[Mapping],
    prompt: str | None = None,
    task_name: str | None = None,
    codec: str | None = None,
    scorer_version: str | None = None,
    dataset_ref: str | None = None,
    dataset_revision: str | None = None,
    split: str | None = None,
) -> tuple[dict, EvalInfo]:
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
        task_name=task_name,
        codec=codec,
        scorer_version=scorer_version,
        dataset_ref=dataset_ref,
        dataset_revision=dataset_revision,
        split=split,
    )

"""Scoring loop: render prompt -> generate -> decode -> score -> aggregate.

generate is injected as a callback, so this module imports no runtime
libraries and runs no inference itself. Rows are read only as x/y/key and the
task only through the Scorable protocol, so nothing here is bound to a concrete
row or task type.
"""

from collections.abc import Iterable, Mapping
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from reishi.primitives.trial import ScoringInfo

GenerateFn = Callable[[str], str]  # prompt -> raw text; injected by executor / replay
ScoredOn = Literal["cpu", "gpu", "tpu"]


@runtime_checkable
class Scorable(Protocol):
    """The subset of Task that run_eval actually calls."""

    def decode(self, raw: str) -> Any: ...
    def score(self, pred: Any, ref: Any) -> Mapping[str, object]: ...
    def aggregate(self, scores: list[Mapping[str, object]]) -> dict: ...


def _aggregator_identity(task: Scorable) -> str:
    """Best-effort identity of the aggregator that will roll up the scores.

    A Task carries an optional custom aggregator as `.aggregator` (its
    `.aggregate` falls back to field_aggregate when that is None, so the
    generic "Task.aggregate" identity is stable for every default Task); a
    duck-typed Scorable's identity is its own aggregate method.
    """
    agg = getattr(task, "aggregator", None)
    if agg is None:
        agg = task.aggregate
    fn = getattr(agg, "__func__", agg)
    qualname = getattr(fn, "__qualname__", None)
    if qualname is None:
        # Callable instances (partial, __call__ objects) have no __qualname__
        # and their repr carries a memory address -- unstable across runs, so
        # it would trip mixed-K warnings. Identify by the callable's type;
        # differently-configured instances of one type collide, which is what
        # the explicit `aggregator` override exists for.
        fn = type(fn)
        qualname = fn.__qualname__
    module = getattr(fn, "__module__", "")
    return f"{module}.{qualname}" if module else qualname


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
    aggregator: str | None = None,
) -> tuple[dict, ScoringInfo]:
    """Run the eval loop; return (aggregate metrics, measurement-key info).

    The K-pinning keyword-only args (task_name, codec, scorer_version,
    dataset_ref, dataset_revision, split) are optional and caller-supplied:
    run_eval only sees rows and a Scorable, so it can't know a Task's name or
    a Dataset's ref itself -- a caller that has those objects passes the
    values through; one it doesn't know is simply left absent. n_eval_rows
    and the aggregator identity are the two K fields run_eval always knows
    itself (it counts the rows and holds the aggregate callable); pass
    `aggregator` only to override the derived identity.
    """
    if not callable(getattr(task, "score", None)):
        raise ValueError(
            "task has no scorer (Task.score is None) -- nothing to run_eval"
        )
    scores: list[Mapping[str, object]] = []
    n_eval_rows = 0
    for row in rows:
        raw = generate(_render(prompt, row["x"]))
        if sink:
            # Persist in the same schema run_eval consumes (key/x/y/raw) so an
            # outputs artifact is directly replayable by rescore(), no adapter.
            sink({"key": row["key"], "x": row["x"], "y": row["y"], "raw": raw})
        scores.append(task.score(task.decode(raw), row["y"]))
        n_eval_rows += 1
    info: ScoringInfo = {
        "n_eval_rows": n_eval_rows,
        "aggregator": aggregator or _aggregator_identity(task),
    }
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
    aggregator: str | None = None,
) -> tuple[dict, ScoringInfo]:
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
        aggregator=aggregator,
    )

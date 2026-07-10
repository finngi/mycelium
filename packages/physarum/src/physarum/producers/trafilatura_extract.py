"""'cpu' runtime producer: scores one trafilatura config against a
recipe's eval dataset -- no gradient step, no model.

Satisfies the Producer contract (trial manifest in, metrics + artifacts out),
so "training" here is deterministic extraction-parameter search.
"""

import json
import time
from pathlib import Path
from typing import Any

import trafilatura

from reishi.primitives import task as task_registry
from reishi.primitives import dataset as dataset_registry
from reishi.primitives.trial import TrialManifest

from physarum.objective import ProducerResult

# The boolean knobs trafilatura.extract() accepts; any other hparams key
# (besides n_eval_rows) is rejected below rather than silently ignored by extract().
_EXTRACT_PARAMS = (
    "favor_precision",
    "favor_recall",
    "include_comments",
    "include_tables",
    "include_images",
    "include_formatting",
    "include_links",
    "deduplicate",
    "fast",  # trafilatura's replacement for the deprecated no_fallback flag
)


def _load_rows(uri: str) -> list[dict]:
    rows = []
    with open(Path(uri)) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def train(trial_manifest: TrialManifest) -> ProducerResult:
    spec = trial_manifest["spec"]
    task_obj = task_registry.get(spec["task"])
    if task_obj.score is None:
        raise ValueError(
            f"task '{task_obj.name}' has no scorer registered; cpu producer can't eval it"
        )

    hparams_cfg = dict(spec.get("hparams", {}))
    n_eval_rows = hparams_cfg.pop("n_eval_rows", None)
    extract_kwargs = {
        k: hparams_cfg.pop(k) for k in _EXTRACT_PARAMS if k in hparams_cfg
    }
    if hparams_cfg:
        raise ValueError(
            f"cpu trafilatura producer: unknown hparams keys {sorted(hparams_cfg)}"
        )

    eval_dataset = spec.get("eval_dataset")
    if eval_dataset is None:
        raise ValueError(
            "cpu trafilatura producer needs a recipe with 'eval_dataset' set"
        )
    ds = dataset_registry.load(eval_dataset)
    rows = _load_rows(ds.uri)
    if n_eval_rows is not None:
        rows = rows[:n_eval_rows]
    if not rows:
        raise ValueError(
            f"cpu trafilatura producer: eval set is empty (dataset '{eval_dataset}', "
            f"n_eval_rows={n_eval_rows})"
        )

    t0 = time.time()
    scores = []
    for row in rows:
        html = row.get("html") or ""
        try:
            md = (
                trafilatura.extract(html, output_format="markdown", **extract_kwargs)
                or ""
            )
        except Exception:
            md = ""
        pred = {"markdown": md}
        ref = {
            "html": html,
            "markdown": row.get("markdown", ""),
            "converter": row.get("converter", ""),
        }
        scores.append(task_obj.score(pred, ref))

    metrics: dict[str, Any] = {
        **task_registry.aggregate(scores),
        "backend": "cpu",
        "extractor": "trafilatura",
        "params": extract_kwargs,
        "n_rows": len(rows),
        "wall_s": round(time.time() - t0, 2),
    }
    return {"metrics": metrics, "artifacts": {}}

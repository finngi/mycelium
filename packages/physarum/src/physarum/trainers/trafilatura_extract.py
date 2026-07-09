"""'local' accelerator trainer: one Optuna-suggested trafilatura config,
scored against a recipe's dataset -- no gradient step, no model.

Same Trainer contract as oyster's mlx_lora (trial manifest in, metrics +
artifacts out), so this "training" is deterministic extraction-parameter
search rather than fine-tuning. No heartbeat thread: unlike mlx_lora's
multi-hour cluster run, one trial here is a bounded in-process CPU loop with
no reaper watching it.
"""

import json
import time
from pathlib import Path
from typing import Any

import trafilatura

from reishi.primitives import task as task_registry
from reishi.primitives import dataset as dataset_registry
from reishi.primitives.trial import TrialManifest

from physarum.objective import TrainerResult

# The only knobs trafilatura.extract() itself takes as booleans -- the sweep's
# search_space is validated (Sweep.validate) to only contain "trainer.*" keys,
# and every one of those must be one of these or eval_n, so a typo'd param
# name fails loudly here rather than silently no-opping inside extract().
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


def train(trial_manifest: TrialManifest) -> TrainerResult:
    spec = trial_manifest["spec"]
    task_obj = task_registry.get(spec["task"])
    if task_obj.score is None:
        raise ValueError(
            f"task '{task_obj.name}' has no scorer registered; local trainer can't eval it"
        )

    trainer_cfg = dict(spec.get("trainer", {}))
    eval_n = trainer_cfg.pop("eval_n", None)
    extract_kwargs = {
        k: trainer_cfg.pop(k) for k in _EXTRACT_PARAMS if k in trainer_cfg
    }
    if trainer_cfg:
        raise ValueError(
            f"local trafilatura trainer: unknown trainer keys {sorted(trainer_cfg)}"
        )

    ds = dataset_registry.load(spec["dataset"])
    rows = _load_rows(ds.uri)
    if eval_n is not None:
        rows = rows[:eval_n]
    if not rows:
        raise ValueError(
            f"local trafilatura trainer: eval set is empty (dataset '{spec['dataset']}', eval_n={eval_n})"
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
        gold = {
            "html": html,
            "markdown": row.get("markdown", ""),
            "converter": row.get("converter", ""),
        }
        scores.append(task_obj.score(pred, gold))

    metrics: dict[str, Any] = {
        **task_registry.aggregate(scores),
        "backend": "local",
        "extractor": "trafilatura",
        "params": extract_kwargs,
        "n_rows": len(rows),
        "wall_s": round(time.time() - t0, 2),
    }
    return {"metrics": metrics, "artifacts": {}}

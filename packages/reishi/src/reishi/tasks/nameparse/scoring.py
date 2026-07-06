"""Per-example field scorer for nameparse, ported from mycelium's eval_common.

field_pairs() excludes PresentedName -- it's the input echoed back, not a
parsed field, so counting it would inflate every prediction's score by one
free true positive regardless of parse quality.
"""

import json

from reishi.primitives.task import ScoreCounts


def _field_pairs(d: dict) -> set:
    return {(k, json.dumps(v, ensure_ascii=False, sort_keys=True)) for k, v in d.items() if k != "PresentedName"}


def score(pred: dict, gold: dict) -> ScoreCounts:
    """One example's contribution to the trial's aggregate (see task.aggregate()).

    An empty/falsy pred (codec failed to decode a JSON object at all) counts
    every gold field as a false negative and none as a false positive --
    there's no prediction to blame a field mismatch on, only a missing one.
    """
    if not pred:
        gold_pairs = _field_pairs(gold)
        return {"tp": 0, "fp": 0, "fn": len(gold_pairs), "exact_match": False, "invalid": True}

    gold_pairs = _field_pairs(gold)
    pred_pairs = _field_pairs(pred)
    return {
        "tp": len(gold_pairs & pred_pairs),
        "fp": len(pred_pairs - gold_pairs),
        "fn": len(gold_pairs - pred_pairs),
        "exact_match": pred == gold,
        "invalid": False,
    }

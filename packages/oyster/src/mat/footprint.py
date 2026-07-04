"""Conservative unified-memory footprint per trial.

Deliberately over-estimates: MLX's own peak-memory logging under-reports
true Metal pressure (OOMs occur well below the logged peak), so the guard
must err toward leaving a trial for a bigger machine, never toward an OOM
kill mid-run.
"""

import re

# MLX LoRA keeps base weights frozen -> small optimizer/activation overhead.
# HF full fine-tunes carry AdamW states + activations -> much larger.
_MULTIPLIER = {"mlx": 3.0, "hf": 6.0}
_BYTES_PER_PARAM_FP16 = 2

_PARAM_HINTS_M = {
    "google/byt5-small": 300.0,
    "google/mt5-small": 300.0,
    "google-t5/t5-small": 60.0,
    "google/gemma-3-270m": 270.0,
}

_PESSIMISTIC_PARAMS_M = 1000.0


def params_m(model_id: str | None) -> float:
    """Best-effort parameter count (millions) from a model id string.

    Parses a trailing size token (0.5B, 360M, 270m...); falls back to a hint
    table, then pessimistic so unknown models are treated as large.
    """
    if not model_id:
        return _PESSIMISTIC_PARAMS_M
    if model_id in _PARAM_HINTS_M:
        return _PARAM_HINTS_M[model_id]
    m = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])", model_id)
    if m:
        val = float(m.group(1))
        return val * 1000.0 if m.group(2).lower() == "b" else val
    return _PESSIMISTIC_PARAMS_M


def estimate_gb(spec: dict) -> float:
    """Peak unified-memory estimate (GB) for one trial, from its frozen spec."""
    trainer = spec.get("trainer", {})
    if spec.get("base_model") is None and "params_m" in trainer:
        p = float(trainer["params_m"])  # from-scratch: arch declares its size
    else:
        p = params_m(spec.get("base_model"))
    mult = _MULTIPLIER.get(trainer.get("backend", "mlx"), _MULTIPLIER["hf"])
    return p * 1e6 * _BYTES_PER_PARAM_FP16 * mult / 1e9

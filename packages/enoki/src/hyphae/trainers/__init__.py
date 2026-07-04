"""Trainers, keyed by the recipe's accelerator.

A trainer is a callable: (trial_manifest: dict) -> dict of metrics plus
artifact URIs. Named 'trainers' not 'adapters' — in this domain an adapter
is the LoRA artifact a trial produces. TRL/PEFT (l4, h100) and XLA/JAX
(v5e) implementations land here; until then the registry is empty and the
driver fails cleanly.
"""

TRAINERS: dict[str, object] = {}


def get(accelerator: str):
    if accelerator not in TRAINERS:
        known = ", ".join(sorted(TRAINERS)) or "none yet"
        raise KeyError(f"no trainer for '{accelerator}' (installed: {known})")
    return TRAINERS[accelerator]

"""Trainer adapters, keyed by the recipe's accelerator.

An adapter is a callable: (trial_manifest: dict) -> dict of metrics plus
artifact URIs. TRL/PEFT (l4, h100) and XLA/JAX (v5e) implementations land
here; until then the registry is empty and the driver fails cleanly.
"""

ADAPTERS: dict[str, object] = {}


def get(accelerator: str):
    if accelerator not in ADAPTERS:
        known = ", ".join(sorted(ADAPTERS)) or "none yet"
        raise KeyError(f"no trainer adapter for '{accelerator}' (installed: {known})")
    return ADAPTERS[accelerator]

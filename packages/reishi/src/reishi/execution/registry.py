"""Producers, keyed by the recipe's runtime, discovered from the
`mcm.producers` entry-point group (entry name = runtime name). Mirrors
oyster.trainers's shape (`supported()`/`get()`) so a caller doesn't care
whether the producer that materializes came from oyster, enoki, or physarum.

Discovery runs lazily on first call and is cached -- importing this module
must not touch entry-point metadata as a side effect.
"""

import sys
from importlib.metadata import entry_points

from reishi.execution.contract import Producer

_producers: dict[str, Producer] | None = None


def _discover() -> dict[str, Producer]:
    producers: dict[str, Producer] = {}
    for ep in entry_points(group="mcm.producers"):
        try:
            producers[ep.name] = ep.load()
        except ImportError as e:
            # Unlike oyster/trainers/__init__.py's mlx guard, this registry
            # can't check e.name against a single known-optional dependency --
            # it discovers arbitrary runtimes -- so it swallows ImportError
            # broadly and lets any other exception surface as a real bug.
            print(
                f"[WARN] producer '{ep.name}' unavailable ({e}) -> this machine "
                f"can't run {ep.name} trials",
                file=sys.stderr,
            )
    return producers


def _producers_map() -> dict[str, Producer]:
    global _producers
    if _producers is None:
        _producers = _discover()
    return _producers


def supported() -> set[str]:
    return set(_producers_map())


def get(runtime: str) -> Producer:
    producers = _producers_map()
    if runtime not in producers:
        known = ", ".join(sorted(producers)) or "none yet"
        raise KeyError(f"no producer for '{runtime}' (installed: {known})")
    return producers[runtime]

"""In-cluster entrypoint: recipe -> trials -> trainer adapter -> manifests.

This module and the mcm CLI are the only places allowed to know Ray exists.
Ray is imported lazily so the driver can plan trials on a laptop without the
cluster dependency group installed.
"""

import sys

import mcm.tasks  # noqa: F401  (populate the task registry)
from mcm.primitives import trial
from mcm.primitives.recipe import Recipe

from hyphae import adapters


def run(recipe_path: str) -> int:
    recipe = Recipe.from_yaml(recipe_path)
    trials = trial.plan(recipe)
    for t in trials:
        trial.save(t)
    print(f"[OK] planned {len(trials)} trial(s) for recipe '{recipe.name}'", file=sys.stderr)

    try:
        adapters.get(recipe.accelerator)
    except KeyError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    # Adapter execution (one Ray task per trial) lands with the first adapter.
    raise NotImplementedError


def main() -> int:
    if len(sys.argv) != 2:
        print("[FAIL] usage: python -m hyphae.driver <recipe.yaml>", file=sys.stderr)
        return 2
    return run(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())

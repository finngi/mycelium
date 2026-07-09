"""In-cluster entrypoint: recipe -> trials -> trainer -> manifests.

Ray is imported lazily so the driver can plan trials on a laptop without the
cluster dependency group installed.
"""

import os
import sys

from reishi import store
from reishi.primitives import trial
from reishi.primitives.recipe import Recipe
from reishi.primitives.trial import TrialManifest
from reishi.tasks import load_tasks

from enoki import trainers
from enoki.trainers.contract import Trainer, TrainerResult


def _use_cluster_store() -> None:
    # No MCM_PG_DSN -> keep the default local filesystem store.
    dsn = os.environ.get("MCM_PG_DSN")
    if not dsn:
        return
    from enoki.store_backend import PostgresBackend

    store.use_backend(PostgresBackend(dsn))


def _make_trainer_call(trainer_fn: Trainer, accelerator: str) -> Trainer:
    """Route the training call onto a GPU-holding Ray worker rather than the
    CPU-only head node. Falls back to calling trainer_fn directly when Ray
    isn't installed (a laptop, or tests that monkeypatch TRAINERS)."""
    try:
        import ray
    except ImportError:
        return trainer_fn

    if not ray.is_initialized():
        ray.init()

    num_gpus = trainers.TRAINER_GPUS.get(accelerator, 0)
    remote_fn = ray.remote(num_gpus=num_gpus)(trainer_fn)

    def _call(manifest: TrialManifest) -> TrainerResult:
        return ray.get(remote_fn.remote(manifest))

    return _call


def run(recipe_path: str) -> int:
    # Populate the task registry (via mcm.tasks entry points) so the recipe's
    # task name resolves.
    load_tasks()
    _use_cluster_store()
    recipe = Recipe.from_yaml(recipe_path)

    # Resolve the trainer BEFORE persisting anything: a RayJob retry re-runs
    # this entrypoint, and trial.plan mints fresh ids each time, so saving
    # first would leave orphaned `planned` trials accumulating per attempt.
    try:
        trainer_fn = trainers.get(recipe.accelerator)
    except KeyError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    call_trainer = _make_trainer_call(trainer_fn, recipe.accelerator)

    trials = trial.plan(recipe)
    for t in trials:
        trial.save(t)
    print(
        f"[OK] planned {len(trials)} trial(s) for recipe '{recipe.name}'",
        file=sys.stderr,
    )

    failures = 0
    for t in trials:
        t.status = "running"
        trial.save(t)
        try:
            result = call_trainer(t.to_manifest())
            t.metrics = result.get("metrics", {})
            t.artifacts = result.get("artifacts", {})
            t.status = "done"
            print(f"[OK] trial '{t.id}' done", file=sys.stderr)
        except Exception as e:
            failures += 1
            t.execution = {**t.execution, "last_error": str(e)}
            t.status = "failed"
            print(f"[FAIL] trial '{t.id}' failed: {e}", file=sys.stderr)
        trial.save(t)

    return 1 if failures else 0


def main() -> int:
    if len(sys.argv) != 2:
        print("[FAIL] usage: python -m enoki.driver <recipe.yaml>", file=sys.stderr)
        return 2
    return run(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())

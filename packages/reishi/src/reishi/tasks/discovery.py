"""load_tasks(): import every installed deployment's task package.

reishi ships no tasks -- a deployment defines its own and advertises the package
holding them via an `mcm.tasks` entry point (mirroring the `mcm.plugins` group
executors already use). load_tasks() resolves that group and imports each
package and every module under it, so each task's register(Task(...)) runs.

Entry points, not an import-path scan, because the same call has to work in two
places that don't share a startup path: the local `mcm` CLI and the in-cluster
driver (`python -m enoki.driver`), which never imports a deployment's __init__.
Installed metadata is visible to both; a filesystem walk relative to a caller is
not.

Import errors are NOT swallowed. A task that won't import is a broken experiment
definition, not an optional extra -- unlike a third-party mcm.plugins entry,
which the CLI degrades to a [WARN] because a deployment doesn't own it. A
first-party task that vanished silently would resurface later as a misleading
"unknown task" from get(); failing here points at the real cause.
"""

import importlib
import pkgutil
from importlib.metadata import entry_points
from types import ModuleType

_GROUP = "mcm.tasks"


def _reraise(_name: str) -> None:
    raise


def _import_submodules(package: ModuleType) -> None:
    search_paths = getattr(package, "__path__", None)
    if search_paths is None:
        return  # a single module, not a package -- loading it already registered it
    for info in pkgutil.walk_packages(
        search_paths, package.__name__ + ".", onerror=_reraise
    ):
        importlib.import_module(info.name)


def load_tasks() -> None:
    for ep in entry_points(group=_GROUP):
        _import_submodules(ep.load())

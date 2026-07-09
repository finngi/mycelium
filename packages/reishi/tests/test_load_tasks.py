"""load_tasks(): resolve mcm.tasks entry points, import their packages so the
tasks under them self-register."""

import importlib
from types import SimpleNamespace

import pytest

from reishi.primitives import task
from reishi.tasks import discovery, load_tasks

_TASK_SRC = (
    "from reishi.primitives.task import Task, register\n"
    "register(Task(name={name!r}, description='d', output_fields=('x',)))\n"
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    # register() writes a module-global; without this, adv-* names leak into the
    # rest of the session and trip the duplicate-name guard on re-registration.
    saved = dict(task._REGISTRY)
    try:
        yield
    finally:
        task._REGISTRY.clear()
        task._REGISTRY.update(saved)


def _make_pkg(root, dotted: str):
    pkg = root
    for part in dotted.split("."):
        pkg = pkg / part
        pkg.mkdir(exist_ok=True)
        (pkg / "__init__.py").touch()
    return pkg


@pytest.fixture
def importable(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    return tmp_path


def _entry_point(dotted: str):
    return SimpleNamespace(name=dotted, load=lambda: importlib.import_module(dotted))


@pytest.fixture
def fake_group(monkeypatch):
    eps: list = []
    monkeypatch.setattr(
        discovery, "entry_points", lambda group: eps if group == "mcm.tasks" else []
    )
    return eps


def test_load_tasks_imports_the_advertised_package(importable, fake_group):
    pkg = _make_pkg(importable, "adv_a")
    (pkg / "a_task.py").write_text(_TASK_SRC.format(name="adv-a"))
    fake_group.append(_entry_point("adv_a"))
    load_tasks()
    assert task.get("adv-a").name == "adv-a"


def test_load_tasks_recurses_into_subpackages(importable, fake_group):
    pkg = _make_pkg(importable, "adv_b.deep")
    (pkg / "b_task.py").write_text(_TASK_SRC.format(name="adv-b-deep"))
    fake_group.append(_entry_point("adv_b"))
    load_tasks()
    assert task.get("adv-b-deep").name == "adv-b-deep"


def test_load_tasks_is_idempotent(importable, fake_group):
    pkg = _make_pkg(importable, "adv_c")
    (pkg / "c_task.py").write_text(_TASK_SRC.format(name="adv-c"))
    fake_group.append(_entry_point("adv_c"))
    load_tasks()
    load_tasks()  # sys.modules dedupes, so register()'s dup guard never fires
    assert task.get("adv-c").name == "adv-c"


def test_a_broken_task_module_fails_loud(importable, fake_group):
    pkg = _make_pkg(importable, "adv_d")
    (pkg / "broken.py").write_text("this is not valid python !!!\n")
    fake_group.append(_entry_point("adv_d"))
    with pytest.raises(SyntaxError):
        load_tasks()

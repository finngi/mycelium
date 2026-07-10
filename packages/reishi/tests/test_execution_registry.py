"""registry: mcm.producers entry-point discovery, per-entry degrade-on-
ImportError, and the supported()/get() surface oyster.trainers established."""

from types import SimpleNamespace

import pytest

from reishi.execution import registry


def _entry_point(name: str, loader):
    return SimpleNamespace(name=name, load=loader)


@pytest.fixture(autouse=True)
def isolated_registry():
    registry._producers = None
    yield
    registry._producers = None


def test_supported_discovers_loadable_entries(monkeypatch):
    def fake_train(manifest):
        return {"metrics": {}, "artifacts": {}}

    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda group: (
            [_entry_point("cpu", lambda: fake_train)]
            if group == "mcm.producers"
            else []
        ),
    )
    assert registry.supported() == {"cpu"}
    assert registry.get("cpu") is fake_train


def test_get_unknown_runtime_lists_installed(monkeypatch):
    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda group: [_entry_point("cpu", lambda: lambda m: m)],
    )
    with pytest.raises(KeyError, match="cpu"):
        registry.get("mlx")


def test_get_on_empty_registry_says_none_yet(monkeypatch):
    monkeypatch.setattr(registry, "entry_points", lambda group: [])
    with pytest.raises(KeyError, match="none yet"):
        registry.get("cpu")


def test_failing_entry_point_degrades_with_warning_and_keeps_others(
    monkeypatch, capsys
):
    def bad_loader():
        raise ImportError("no module named 'mlx'")

    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda group: [
            _entry_point("mlx", bad_loader),
            _entry_point("cpu", lambda: lambda m: m),
        ],
    )
    assert registry.supported() == {"cpu"}
    err = capsys.readouterr().err
    assert "[WARN] producer 'mlx' unavailable" in err
    assert "can't run mlx trials" in err


def test_non_import_error_from_a_loaded_entry_point_surfaces():
    def bad_loader():
        raise RuntimeError("a real bug, not a missing dependency")

    registry._producers = None
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            registry, "entry_points", lambda group: [_entry_point("cpu", bad_loader)]
        )
        with pytest.raises(RuntimeError, match="a real bug"):
            registry.supported()


def test_discovery_runs_once_and_is_cached(monkeypatch):
    calls = []

    def tracking_entry_points(group):
        calls.append(group)
        return [_entry_point("cpu", lambda: lambda m: m)]

    monkeypatch.setattr(registry, "entry_points", tracking_entry_points)
    registry.supported()
    registry.get("cpu")
    assert calls == ["mcm.producers"]

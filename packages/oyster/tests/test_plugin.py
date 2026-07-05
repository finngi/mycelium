"""oyster's face inside the mcm CLI: vocabulary registers cleanly and the
drain path leads with the assignment-level label."""

import pytest

from mcm.cli import grammar
from mcm.cli.grammar import Command, canonicalize

from oyster import machine, mcm_plugin


@pytest.fixture(autouse=True)
def mesh_grammar(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(machine, "BUSY_FILE", tmp_path / "busy")
    domains, plurals, verbs = grammar.DOMAINS, dict(grammar._PLURALS), dict(grammar.VERBS)
    grammar.extend(mcm_plugin.DOMAINS, mcm_plugin.VERBS)
    yield
    grammar.DOMAINS = domains
    grammar._PLURALS.clear(), grammar._PLURALS.update(plurals)
    grammar.VERBS.clear(), grammar.VERBS.update(verbs)


def test_mesh_vocabulary_canonicalizes():
    assert (canonicalize(["work"]).domain, canonicalize(["work"]).action) == ("mesh", "work")
    assert canonicalize(["mesh"]).action == "list"
    cmd = canonicalize(["requeue", "trial-1"])
    assert (cmd.domain, cmd.action, cmd.objects) == ("mesh", "requeue", ["trial-1"])


def test_every_verb_has_a_handler():
    for v in mcm_plugin.VERBS:
        assert ("mesh", v.name) in mcm_plugin.HANDLERS
    assert ("mesh", "list") in mcm_plugin.HANDLERS


def test_drain_leads_with_label_then_busy_file(monkeypatch):
    calls = []
    monkeypatch.setattr(machine, "set_ready", lambda ready: calls.append(ready) or True)
    mcm_plugin.mesh_drain(Command(domain="mesh", action="drain"))
    assert calls == [False] and machine.BUSY_FILE.exists()
    mcm_plugin.mesh_undrain(Command(domain="mesh", action="undrain"))
    assert calls == [False, True] and not machine.BUSY_FILE.exists()


def test_drain_still_sets_busy_file_when_label_unreachable(monkeypatch):
    monkeypatch.setattr(machine, "set_ready", lambda ready: False)
    mcm_plugin.mesh_drain(Command(domain="mesh", action="drain"))
    assert machine.BUSY_FILE.exists()

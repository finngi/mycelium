"""The grammar's plugin surface: extend() grows the vocabularies under the
same disjointness law the built-ins live by."""

import pytest

from reishi.cli import grammar
from reishi.cli.grammar import GrammarError, Verb, canonicalize


@pytest.fixture(autouse=True)
def pristine_grammar():
    domains, plurals, verbs = grammar.DOMAINS, dict(grammar._PLURALS), dict(grammar.VERBS)
    yield
    grammar.DOMAINS = domains
    grammar._PLURALS.clear(), grammar._PLURALS.update(plurals)
    grammar.VERBS.clear(), grammar.VERBS.update(verbs)


def test_extend_adds_domain_and_verbs():
    grammar.extend(("mesh",), (Verb("work", home="mesh", readonly=False),))
    cmd = canonicalize(["work"])
    assert (cmd.domain, cmd.action) == ("mesh", "work")
    assert canonicalize(["mesh"]).action == "list"  # read-only default applies


def test_extend_rejects_verb_colliding_with_domain():
    with pytest.raises(GrammarError, match="collides"):
        grammar.extend((), (Verb("trial", home="mesh", readonly=True),))


def test_extend_rejects_domain_colliding_with_verb():
    with pytest.raises(GrammarError, match="collides"):
        grammar.extend(("run",), ())


def test_extend_is_idempotent_for_known_domains():
    grammar.extend(("mesh",), ())
    grammar.extend(("mesh",), ())
    assert grammar.DOMAINS.count("mesh") == 1


def test_extend_rejects_verb_colliding_with_builtin_verb():
    # Without this check, VERBS.update() below would silently overwrite the
    # built-in 'list' verb with the plugin's -- no error, just wrong behavior.
    with pytest.raises(GrammarError, match="collides"):
        grammar.extend((), (Verb("list", home="mesh", readonly=True),))
    assert grammar.VERBS["list"].home is None  # built-in survives


def test_extend_rejects_verb_shadowed_by_alias():
    # canonicalize() resolves 'get'/'ls' to describe/list before ever
    # consulting VERBS, so a plugin verb named 'get' would install without
    # error yet be permanently unreachable. That must be rejected, not silent.
    with pytest.raises(GrammarError, match="collides"):
        grammar.extend((), (Verb("get", home="mesh", readonly=True),))
    with pytest.raises(GrammarError, match="collides"):
        grammar.extend((), (Verb("ls", home="mesh", readonly=True),))

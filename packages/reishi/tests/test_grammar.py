import pytest

from reishi.cli.grammar import DOMAINS, VERBS, GrammarError, canonicalize


CANONICALIZATION = [
    # typed -> (domain, action, objects)
    (["trials"], ("trial", "list", [])),
    (["trial", "list"], ("trial", "list", [])),
    (["trial", "7f3a-x"], ("trial", "describe", ["7f3a-x"])),
    (["logs", "7f3a-x"], ("trial", "logs", ["7f3a-x"])),
    (["run", "example.yaml"], ("recipe", "run", ["example.yaml"])),
    (["submit", "extract-prompted"], ("experiment", "submit", ["extract-prompted"])),
    (["dataset", "build", "sample"], ("dataset", "build", ["sample"])),
    (["board"], ("board", "list", [])),
    (["tasks"], ("task", "list", [])),
    (["get", "trial", "7f3a-x"], ("trial", "describe", ["7f3a-x"])),
    # order tolerance: classification is by vocabulary, not position
    (["example.yaml", "run"], ("recipe", "run", ["example.yaml"])),
]


@pytest.mark.parametrize("typed,expected", CANONICALIZATION)
def test_canonicalization(typed, expected):
    cmd = canonicalize(typed)
    assert (cmd.domain, cmd.action, cmd.objects) == expected


def test_flags_split_from_head():
    cmd = canonicalize(["submit", "exp-1", "--seeds", "3", "-o", "json"])
    assert cmd.objects == ["exp-1"]
    assert cmd.flags == ["--seeds", "3", "-o", "json"]


def test_bare_mcm_is_status():
    cmd = canonicalize([])
    assert cmd.domain is None and cmd.action is None


def test_omitted_action_never_mutates():
    for domain in DOMAINS:
        cmd = canonicalize([domain])
        assert VERBS[cmd.action].readonly


def test_verb_without_home_requires_domain():
    with pytest.raises(GrammarError, match="needs a domain"):
        canonicalize(["list"])


def test_two_domains_rejected():
    with pytest.raises(GrammarError, match="two domains"):
        canonicalize(["trial", "dataset"])


def test_two_actions_rejected():
    with pytest.raises(GrammarError, match="two actions"):
        canonicalize(["run", "submit", "x.yaml"])


def test_bare_word_object_rejected():
    with pytest.raises(GrammarError, match="not a domain"):
        canonicalize(["foo"])


def test_vocabularies_disjoint():
    plurals = {d + "s" for d in DOMAINS}
    assert not set(VERBS) & (set(DOMAINS) | plurals)


def test_canonical_echo_string():
    cmd = canonicalize(["trials"])
    assert cmd.canonical() == "mcm trial list"

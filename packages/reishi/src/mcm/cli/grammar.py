"""mcm <domain> <action> <objects> --<parameters> — all segments optional.

Tokens are classified by vocabulary lookup, not position, so segments may be
omitted or reordered. This only stays unambiguous while DOMAINS and VERBS are
disjoint and object refs can't collide with either (they must contain a
hyphen, dot, slash, underscore, or digit).
"""

import re
from dataclasses import dataclass, field

DOMAINS = ("task", "dataset", "recipe", "trial", "experiment", "board")
_PLURALS = {d + "s": d for d in DOMAINS}

_OBJECT_REF = re.compile(r"[-./_\d]")


@dataclass(frozen=True)
class Verb:
    name: str
    home: str | None  # domain inferred when omitted; None -> domain required
    readonly: bool


VERBS = {
    v.name: v
    for v in (
        Verb("run", home="recipe", readonly=False),
        Verb("submit", home="experiment", readonly=False),
        Verb("build", home="dataset", readonly=False),
        Verb("stop", home="trial", readonly=False),
        Verb("logs", home="trial", readonly=True),
        Verb("show", home="board", readonly=True),
        Verb("list", home=None, readonly=True),
        Verb("describe", home=None, readonly=True),
    )
}
_VERB_ALIASES = {"get": "describe", "ls": "list"}

_overlap = set(VERBS) & (set(DOMAINS) | set(_PLURALS))
assert not _overlap, f"grammar vocabularies must stay disjoint: {_overlap}"


class GrammarError(Exception):
    pass


@dataclass
class Command:
    domain: str | None
    action: str | None
    objects: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def canonical(self) -> str:
        parts = ["mcm", self.domain or "", self.action or "", *self.objects, *self.flags]
        return " ".join(p for p in parts if p)


def canonicalize(argv: list[str]) -> Command:
    head: list[str] = []
    flags: list[str] = []
    for i, tok in enumerate(argv):
        if tok.startswith("-"):
            flags = argv[i:]
            break
        head.append(tok)

    domain: str | None = None
    action: str | None = None
    objects: list[str] = []

    for tok in head:
        word = _VERB_ALIASES.get(tok, tok)
        if word in DOMAINS or word in _PLURALS:
            resolved = _PLURALS.get(word, word)
            if domain and domain != resolved:
                raise GrammarError(f"two domains given: '{domain}' and '{resolved}'")
            domain = resolved
        elif word in VERBS:
            if action:
                raise GrammarError(f"two actions given: '{action}' and '{word}'")
            action = word
        elif _OBJECT_REF.search(tok):
            objects.append(tok)
        else:
            raise GrammarError(
                f"'{tok}' is not a domain, an action, or a valid object ref "
                f"(object names must contain a hyphen, dot, slash, underscore, or digit)"
            )

    # Omitted action defaults must be read-only: never mutate on less typing.
    if action is None:
        if objects:
            action = "describe"
        elif domain:
            action = "list"

    if domain is None and action is not None:
        home = VERBS[action].home
        if home is None:
            raise GrammarError(f"'{action}' needs a domain: one of {', '.join(DOMAINS)}")
        domain = home

    return Command(domain=domain, action=action, objects=objects, flags=flags)

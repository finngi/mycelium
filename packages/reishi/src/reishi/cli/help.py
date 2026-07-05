"""Help text generated from the live vocabularies — never hand-maintained."""

from reishi.cli.grammar import DOMAINS, VERBS


def render(handlers: dict) -> str:
    lines = [
        "mcm — reishi: experiment contract layer for small-model training",
        "",
        "usage: mcm <domain> <action> <objects> --<parameters>",
        "",
        "Every segment is optional; tokens are classified by vocabulary, so",
        "shorthand canonicalizes deterministically (echoed on stderr):",
        "  mcm                    status overview",
        "  mcm trials             -> mcm trial list",
        "  mcm trial <id>         -> mcm trial describe <id>",
        "  mcm logs <id>          -> mcm trial logs <id>",
        "  mcm run <recipe.yaml>  -> mcm recipe run  (local, in-process)",
        "  mcm submit <exp>       -> mcm experiment submit  (RayJob)",
        "",
        f"domains: {', '.join(DOMAINS)}  (plurals accepted)",
        "",
        "actions:",
    ]
    for v in sorted(VERBS.values(), key=lambda v: v.name):
        home = f"home: {v.home}" if v.home else "domain required"
        kind = "read-only" if v.readonly else "mutating"
        lines.append(f"  {v.name:<10} {kind:<10} {home}")
    lines += [
        "  (aliases: get -> describe, ls -> list)",
        "",
        "implemented commands:",
    ]
    for domain, action in sorted(handlers):
        lines.append(f"  mcm {domain} {action}")
    lines += [
        "",
        "flags: -o json on any command (canonical echo stays on stderr);",
        "       --plan on recipe run; --metric <name> on board show",
        "",
        "store: ~/.mcm/store, override with MCM_STORE",
    ]
    return "\n".join(lines)

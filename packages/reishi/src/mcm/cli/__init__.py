import sys

from mcm.cli.grammar import GrammarError, canonicalize


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    try:
        cmd = canonicalize(argv)
    except GrammarError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 2

    from mcm.cli import commands

    if cmd.domain is None and cmd.action is None:
        return commands.status(cmd)

    # Canonical echo on stderr: users learn the full grammar from shorthand,
    # agents get an unambiguous record, and -o json stdout stays clean.
    print(f"> {cmd.canonical()}", file=sys.stderr)

    handler = commands.HANDLERS.get((cmd.domain, cmd.action))
    if handler is None:
        print(f"[FAIL] 'mcm {cmd.domain} {cmd.action}' is not implemented", file=sys.stderr)
        return 2

    try:
        return handler(cmd)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 1

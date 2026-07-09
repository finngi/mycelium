import sys

from reishi.cli.grammar import GrammarError, canonicalize

_plugins_loaded = False


def _load_plugins(handlers: dict) -> None:
    """Load `mcm.plugins` entry points and merge each module's DOMAINS, VERBS,
    and HANDLERS. A failed plugin warns on stderr rather than breaking the CLI."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    from importlib.metadata import entry_points

    from reishi.cli import grammar

    for ep in entry_points(group="mcm.plugins"):
        try:
            mod = ep.load()
            grammar.extend(
                getattr(mod, "DOMAINS", ()), tuple(getattr(mod, "VERBS", ()))
            )
            handlers.update(getattr(mod, "HANDLERS", {}))
        except Exception as e:
            print(
                f"[WARN] mcm plugin '{ep.name}' failed to load: {type(e).__name__}: {e}",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    from reishi.cli import commands

    _load_plugins(commands.HANDLERS)

    if "-h" in argv or "--help" in argv or argv[:1] == ["help"]:
        from reishi.cli.help import render

        print(render(commands.HANDLERS))
        return 0

    # Ordered after the help short-circuit: load_tasks() fails loud on a broken
    # task, but `--help` must render regardless.
    from reishi.tasks import load_tasks

    try:
        load_tasks()
    except Exception as e:
        print(f"[FAIL] task loading failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    try:
        cmd = canonicalize(argv)
    except GrammarError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 2

    if cmd.domain is None and cmd.action is None:
        return commands.status(cmd)

    # Echo on stderr, not stdout, so -o json output stays clean while the user
    # still sees the resolved command.
    print(f"> {cmd.canonical()}", file=sys.stderr)

    handler = commands.HANDLERS.get((cmd.domain, cmd.action))
    if handler is None:
        print(
            f"[FAIL] 'mcm {cmd.domain} {cmd.action}' is not implemented",
            file=sys.stderr,
        )
        return 2

    try:
        return handler(cmd)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 1

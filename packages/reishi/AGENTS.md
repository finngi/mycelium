# AGENTS.md — mcm-reishi

Guidance for AI coding agents (Claude Code, GitHub Copilot, Cursor, Codex,
etc.) working in this repo. `CLAUDE.md` in this repo is just `@AGENTS.md` —
this file is the canonical source, tool-specific files import it.

## What this repo is

mcm (reishi) is the experiment contract layer for small-model training on
KubeRay. It defines what a training run *means* — reproducible, scored
identically, comparable on one board — regardless of what executes it.
mcm itself never imports Ray, MLX, or any accelerator-specific dependency;
execution lives in sibling repos.

## Repo shape

Single package, `src/reishi/`, installed with `uv`. No path dependencies —
this is the one repo in the mcm family with no dependency on the others.
Sibling repos (`../mcm-enoki`, `../mcm-oyster`) depend on this one via a
local path dependency and extend its CLI via `mcm.plugins` entry points.

| Primitive | What it is |
|---|---|
| `Task` | Output schema + codec + constrained decoder + scorer. |
| `Dataset` | Versioned `gs://` prefix + card + leak contract (`dataset.leaks()`). |
| `Recipe` | Declarative model x dataset x prompt x trainer spec. |
| `Trial` | One recipe x seed execution — a manifest, not a log line. |
| `Board` | Aggregation over trial manifests; computed, never stored as truth. |

`reishi.store` is a `StorageBackend` Protocol: `LocalFilesystemBackend` is
the default, executors swap in their own (e.g. enoki's `PostgresBackend`)
via `store.use_backend()` — reishi's primitives never know which one is
active.

## Conventions

1. **Grammar is closed and disjoint**, enforced by `tests/test_grammar.py`:
   domains and verbs are separate closed vocabularies, every verb has one
   home domain, an omitted action defaults to something read-only.
   Plugins extend the grammar via `grammar.extend()`, which raises on any
   collision — never resolve a collision by silently letting one side win.
2. **`-o json` everywhere.** Canonical-form echo goes to stderr so stdout
   stays parseable.
3. **Plugins own their domain.** `mcm.plugins` entry points contribute
   `DOMAINS`, `VERBS`, `HANDLERS` — a broken plugin degrades to a `[WARN]`
   on stderr, never a dead CLI.
4. **No emojis in tool output.** ASCII status indicators only (`[OK]`,
   `[FAIL]`, `[WARN]`, `[INFO]`, `->`).
5. **Comments: only when the *why* is non-obvious.** Never narrate *what*
   the code does.

## Working in this repo

```
uv venv && uv pip install -e . --group dev
uv run pytest -q
uv run reishi-admin tasks     # dev-only CLI; deployments expose this as `mcm`
uvx ruff check .               # lint (required in CI)
```

`MCM_STORE` overrides the manifest store root (default `~/.mcm/store`).

## Sibling repos

- [mcm-enoki](../mcm-enoki) — KubeRay (cloud) execution: `l4`, `h100`, `v5e`.
- [mcm-oyster](../mcm-oyster) — self-hosted mesh execution (currently `mlx` on Apple Silicon).

Both must be checked out as siblings on disk (`uv.sources` path
dependency); neither is a git submodule of this repo.

See `CONTRIBUTING.md` for commit conventions and the PR/CI gate.

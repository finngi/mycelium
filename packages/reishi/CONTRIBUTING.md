# Contributing

## Setup

From the repo root — this package is one member of the `mycelium` uv
workspace, not a standalone checkout:

```bash
uv sync --all-extras           # or --group dev, for a lighter install
```

## Before opening a PR

```bash
uvx ruff check .
uvx ruff format --check .
uv run pytest packages/reishi -q
```

Scope pytest to `packages/reishi`, never a bare `uv run pytest -q` from the
root: reishi's task registry is global process state, and a pooled session
across the workspace's test suites cross-contaminates (reishi and oyster
both register a task named `fixture`). All three commands run in CI
(`.github/workflows/ci.yml`, one `lint` job plus a per-package `test`
matrix) and are required to merge.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) —
`feat:`, `fix:`, `refactor:`, `test:`, `chore:`, etc. — since
[release-please](https://github.com/googleapis/release-please) reads
these to decide the version bump. The whole workspace ships as one
`mcm-mycelium` distribution on one prerelease counter (`0.0.0-a.N`), so a
`feat`/`fix` anywhere in `packages/*` bumps every package's version and
`CHANGELOG.md` together — there's no separate reishi release. A commit
that isn't a `feat`/`fix` (docs, chore, test-only) doesn't trigger one.

## Pull requests

`main` is protected: PRs need CI green to merge. This is a solo-owner
project, so there's no required-reviewer count — self-merge once checks
pass.

## Depended on by

`packages/enoki` and `packages/oyster` depend on this package as workspace
members (root `pyproject.toml`'s `[tool.uv.sources]`) and extend its CLI
via `mcm.plugins` — a change to a primitive's manifest shape or to
`reishi.store`'s contract is worth checking against both before merging.
See this package's `AGENTS.md` primitives table for what's ratified.

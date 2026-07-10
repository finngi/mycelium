# Contributing

## Setup

From the repo root — this package is one member of the `mycelium` uv
workspace, no sibling checkout needed:

```bash
uv sync --all-extras           # or --group dev, for a lighter install
```

The root `dev` dependency-group (synced by default) already lists `oyster`
and `trafilatura` directly — kept there rather than behind physarum's own
`mlx`/`cpu` extras (see the root `pyproject.toml`'s `[dependency-groups]`
comment) — so both the `mlx` and `cpu` sweep runtimes are exercisable out
of the box, not just the in-process scheduling logic against a stub
`Producer`. `mlx` sweeps still need Apple Silicon hardware to actually run:
oyster's own mlx/mlx-lm deps are platform-gated, not extras-gated.

## Before opening a PR

```bash
uvx ruff check .
uvx ruff format --check .
uv run pytest packages/physarum -q
```

Scope pytest to `packages/physarum`, not a bare `uv run pytest -q` from the
root — reishi's task registry and each plugin's conftest mutate shared
global state, so a pooled session across the workspace cross-contaminates.
All three commands run in CI (`.github/workflows/ci.yml`, one `lint` job
plus a per-package `test` matrix) and are required to merge.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) —
`feat:`, `fix:`, `refactor:`, `test:`, `chore:`, etc. — since
[release-please](https://github.com/googleapis/release-please) reads
these to decide the version bump. The whole workspace ships as one
`mcm-mycelium` distribution on one prerelease counter (`0.0.0-a.N`) —
there's no separate physarum release.

## Pull requests

`main` is protected: PRs need CI green to merge. Solo-owner project, so
there's no required-reviewer count — self-merge once checks pass.

## Depends on

- `packages/reishi` — required workspace member, for `Task`/`Dataset`/
  `Recipe`/`Trial` and `reishi.store`.
- `packages/oyster` — optional; only needed to actually run a sweep whose
  template targets `runtime: mlx` (`_resolve_producer` in `mcm_plugin.py`).
  `enoki` isn't wired into `_resolve_producer` yet — `mlx` and `cpu` are the
  only runtimes physarum can dispatch to today.

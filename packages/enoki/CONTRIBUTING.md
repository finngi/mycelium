# Contributing

## Setup

From the repo root — this package is one member of the `mycelium` uv
workspace, no sibling checkout needed:

```
uv sync --all-extras           # or --group dev, for a lighter install
```

The `cluster` dependency group (torch/ray/psycopg/transformers/peft/
accelerate/google-cloud-storage) is only needed to actually run a
trainer — from `packages/enoki`, `uv sync --group cluster` when working on
`trainers.train_l4` or `store_backend.PostgresBackend` locally. CI never
installs it: `uv sync --all-extras` (what CI runs) never pulls in a
dependency group, and `enoki.trainers`/`enoki.store_backend` both import
their heavy deps lazily, so lint/test stay meaningful without it.

## Before opening a PR

```
uvx ruff check .
uvx ruff format --check .
uv run pytest packages/enoki -q
```

Scope pytest to `packages/enoki`, not a bare `uv run pytest -q` from the
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
there's no separate enoki release.

## Pull requests

`main` is protected: PRs need CI green to merge. Solo-owner project, so
there's no required-reviewer count — self-merge once checks pass.

## Training image

`jobs/Dockerfile` builds the cluster training image; `jobs/rayjob.yaml` is
the RayJob template `mcm experiment submit` fills in. Changes to either
need a real submit against the `train` namespace to verify — CI doesn't
do this (no cluster credentials), so treat it as manual verification
before merging.

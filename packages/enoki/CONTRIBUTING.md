# Contributing

## Setup

Requires `../mcm-reishi` checked out as a sibling directory (`uv.sources`
path dependency).

```
uv venv && uv pip install -e . --group dev
uv run pytest -q
```

The `cluster` dependency group (torch/ray/transformers/peft/accelerate/
google-cloud-storage) is only needed to actually run a trainer — install
it with `uv pip install -e . --group cluster` when working on
`trainers.train_l4` or `store_backend.PostgresBackend` locally. CI never
installs it: `enoki.trainers` and `enoki.store_backend` both import their
heavy deps lazily, so lint/test stay meaningful without it.

## Before opening a PR

```
uvx ruff check .
uv run pytest -q
```

Both run in CI (`.github/workflows/ci.yml`) and are required to merge.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) —
`feat:`, `fix:`, `refactor:`, `test:`, `chore:`, etc. — since
[release-please](https://github.com/googleapis/release-please) reads
these to decide version bumps and generate `CHANGELOG.md`.

## Pull requests

`main` is protected: PRs need CI green to merge. Solo-owner project, so
there's no required-reviewer count — self-merge once checks pass.

## Training image

`jobs/Dockerfile` builds the cluster training image; `jobs/rayjob.yaml` is
the RayJob template `mcm experiment submit` fills in. Changes to either
need a real submit against the `train` namespace to verify — CI doesn't
do this (no cluster credentials), so treat it as manual verification
before merging.

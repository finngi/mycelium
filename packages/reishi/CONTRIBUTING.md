# Contributing

## Setup

```
uv venv && uv pip install -e . --group dev
uv run pytest -q
```

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
these to decide version bumps and generate `CHANGELOG.md`. A commit that
isn't a `feat`/`fix` (docs, chore, test-only) doesn't trigger a release.

## Pull requests

`main` is protected: PRs need CI green to merge. This is a solo-owner
project, so there's no required-reviewer count — self-merge once checks
pass.

## Sibling repos

mcm-enoki and mcm-oyster depend on this repo via a local path
(`../mcm-reishi`), so keep them checked out next to each other on disk.
CI in those repos checks this repo out alongside via
`actions/checkout` against `${{ github.repository_owner }}/mcm-reishi`.

# Contributing

## Setup

```
uv venv && uv pip install -e . --group dev
uv run pytest -q
```

Requires `../mcm-reishi` checked out as a sibling directory (`uv.sources`
path dependency). `../mcm-oyster` is also required if you're exercising an
`mlx` sweep end-to-end rather than just the scheduling logic (`mlx` extra).

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

## Sibling repos

Depends on mcm-reishi via a local path (`../mcm-reishi`), and optionally
mcm-oyster (`../mcm-oyster`, for `mlx` sweeps) — keep them checked out next
to each other on disk. CI checks reishi out alongside via `actions/checkout`
against `${{ github.repository_owner }}/mcm-reishi`.

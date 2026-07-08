# Contributing

## Setup

```
uv venv && uv pip install -e . --group dev
uv run pytest -q
```

Requires `../mcm-reishi` checked out as a sibling directory (`uv.sources`
path dependency).

## Before opening a PR

```
uvx ruff check .
uv run pytest -q
```

Both run in CI (`.github/workflows/ci.yml`) and are required to merge.
`herder.yml` / `reaper.yml` / `worker.yml` are the mesh's own scheduled
orchestration jobs, not part of the PR gate — they can only be verified
against a live runner (see `scripts/join_network.sh`).

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) —
`feat:`, `fix:`, `refactor:`, `test:`, `chore:`, etc. — since
[release-please](https://github.com/googleapis/release-please) reads
these to decide version bumps and generate `CHANGELOG.md`.

## Pull requests

`main` is protected: PRs need CI green to merge. Solo-owner project, so
there's no required-reviewer count — self-merge once checks pass.

`store/` (the manifest queue) is committed to this repo and can change on
every claim/train/requeue — a PR touching it is normal, not a sign
something's wrong.

## Joining the mesh

`OYSTER_REPO=finngi/mycelium ./scripts/join_network.sh`
on an Apple Silicon Mac. Needs a `gh`-authenticated session (or a manually
minted runner registration token) and `HF_TOKEN` set as a repo secret
before `worker.yml` can push/resume checkpoints.

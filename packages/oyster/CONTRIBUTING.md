# Contributing

## Setup

From the repo root — this package is one member of the `mycelium` uv
workspace, no sibling checkout needed:

```bash
uv sync --all-extras           # or --group dev, for a lighter install
```

## Before opening a PR

```bash
uvx ruff check .
uvx ruff format --check .
uv run pytest packages/oyster -q
```

Scope pytest to `packages/oyster`, not a bare `uv run pytest -q` from the
root — reishi's task registry and each plugin's conftest mutate shared
global state, so a pooled session across the workspace cross-contaminates.
All three commands run in CI (`.github/workflows/ci.yml`, one `lint` job
plus a per-package `test` matrix) and are the only checks required to
merge.

`herder.yml`/`worker.yml`/`reaper.yml`/`mesh-sync.yml` are **not** part of
that gate, and right now they aren't part of anything else either: they
live under `packages/oyster/.github/workflows/`, a path GitHub Actions
does not read (only the repo-root `.github/workflows/` is live, and it
holds none of these four). That's a real gap left by the monorepo
migration, not a design choice — see `AGENTS.md` for the full note. Don't
describe these as the mesh's live scheduled orchestration until they've
been moved to the repo root and re-verified against a runner.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) —
`feat:`, `fix:`, `refactor:`, `test:`, `chore:`, etc. — since
[release-please](https://github.com/googleapis/release-please) reads
these to decide the version bump. The whole workspace ships as one
`mcm-mycelium` distribution on one prerelease counter (`0.0.0-a.N`) —
there's no separate oyster release.

## Pull requests

`main` is protected: PRs need CI green to merge. Solo-owner project, so
there's no required-reviewer count — self-merge once checks pass.

By design, `store/` — the mcm manifest store — is meant to be committed
inside this package, since the queue-is-the-repo design (see `AGENTS.md`)
depends on every claim/train/requeue being a commit here. In this
checkout `packages/oyster/store/` doesn't currently exist (`git log --
packages/oyster/store` is empty), so don't expect to see it churn in a
diff yet — that's the same migration gap as the workflows above, not
something you broke.

## Joining the mesh

Not currently possible end to end: with the mesh workflows sitting outside
`.github/workflows/` and no committed `store/`, a runner has nothing to
claim from. `scripts/join_network.sh` (`OYSTER_REPO=finngi/mycelium
./scripts/join_network.sh` on an Apple Silicon Mac, `gh`-authenticated,
`HF_TOKEN` set as a repo secret) is still the right onboarding script once
the gap above is closed — don't advertise it as working today.

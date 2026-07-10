# AGENTS.md — oyster

oyster is mcm's no-cloud execution layer: self-hosted runners pull-claim
trials instead of having GitHub push jobs at them. The claim/heartbeat
mechanism is machine-agnostic; only the one trainer wired up so far
(`runtime: mlx`) happens to require Apple Silicon. By design the mcm
manifest store lives as `store/` committed inside this package — the queue
IS the repo, and a claim is a commit — see `queue.py`/`gitstore.py`'s
docstrings before changing anything there; the pull-based, git-atomic
design is load-bearing, not incidental. That design is currently not wired
up in this checkout — see the gap below before assuming it's live.

**Post-migration gap, unresolved:** the mesh's own orchestration —
`herder.yml`, `worker.yml`, `reaper.yml`, `mesh-sync.yml` — still lives
under `packages/oyster/.github/workflows/`, a path GitHub Actions does not
read (it only picks up workflows from the repo-root `.github/workflows/`,
which currently holds only `ci.yml`/`codeql.yml`/`publish.yml`/
`release-please.yml`/`security.yml`). No trial has been claimed through
these since the monorepo migration, and `packages/oyster/store/` — the
committed queue — doesn't exist in this checkout either (confirm with
`git log -- packages/oyster/store` before assuming otherwise). Don't
describe the mesh as currently operating; if you're asked to revive it,
the fix is moving those four workflow files to the repo root (checking
their `runs-on`/path filters still make sense there) and re-establishing
`store/`.

## Package shape

Depends on [`packages/reishi`](../reishi/AGENTS.md) as a workspace member
(root `pyproject.toml`'s `[tool.uv.sources]`) — no sibling checkout needed.
No CLI of its own: installing it registers the `mesh` domain into mcm's
CLI via the `mcm.plugins` entry point (`src/oyster/mcm_plugin.py`).

`ci.yml` (lint + test on GitHub-hosted runners) is the one workflow
actually required for PRs to merge — don't fold checks into the mesh
orchestration files above even once they're moved back to the root.

## Conventions

- The `~/.mycelium-*` paths (`~/.mycelium-busy`, `~/.mycelium-runner-config`)
  are a fleet contract with machines already onboarded via
  `join_network.sh` — renaming them orphans the fleet, not just this package.
- `mcm drain`/`mcm undrain` toggle two layers in order: the `ready`
  runner label (assignment) then the busy file (execution). Keep that
  order when touching `machine.py`.
- No emojis in tool/CLI output. ASCII status indicators only.
- Comments: only when the *why* is non-obvious — this package's existing
  comments (e.g. `queue.py`'s claim-race handling) are the calibration
  for how much is enough.

## Working in this package

```bash
uv sync --all-extras           # from the repo root
uv run pytest packages/oyster -q
uvx ruff check .
```

`mcm_plugin.py` points `MCM_STORE` at `./store` only when a `store/trials`
directory already exists at the CWD (the committed-queue case above); absent
that, it falls through to reishi's own default (`~/.mcm/store`, sqlite by
default) same as any other package.

## Sibling packages

- [`packages/reishi`](../reishi/AGENTS.md) — the contract layer this package executes against.
- [`packages/enoki`](../enoki/AGENTS.md) — the other executor (KubeRay, cloud GPU/TPU).

See `CONTRIBUTING.md` for commit conventions, the PR/CI gate, and how to
join the mesh as a runner.

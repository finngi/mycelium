# AGENTS.md — mcm-oyster

Guidance for AI coding agents working in this repo. `CLAUDE.md` in this
repo is just `@AGENTS.md` — this file is the canonical source.

## What this repo is

oyster is mcm's no-cloud execution layer: self-hosted runners pull-claim
trials instead of having GitHub push jobs at them. The claim/heartbeat
mechanism is machine-agnostic; only the one trainer wired up so far
(`runtime: mlx`) happens to require Apple Silicon. `store/` (the mcm
manifest store) is committed
to this repo — the queue IS the repo, and a claim is a commit. See the
README's "Scheduling: pull, not push" section before changing anything
in `queue.py`/`gitstore.py`; the pull-based, git-atomic design is load
-bearing, not incidental.

## Repo shape

Depends on [mcm-reishi](https://github.com/finngi/mycelium/tree/main/packages/reishi)
via a local path (`uv.sources`, `../mcm-reishi`) — check it out as a
sibling directory. No CLI of its own: installing it registers the `mesh`
domain into mcm's CLI via the `mcm.plugins` entry point
(`src/oyster/mcm_plugin.py`).

Three GitHub Actions workflows are the mesh's own orchestration, not CI:

- `herder.yml` — scheduled, wakes `min(planned, ready-idle)` workers
- `worker.yml` — runs on `[self-hosted, macOS, mlx, ready]`, claims and trains
- `reaper.yml` — scheduled, requeues stale-heartbeat trials

`ci.yml` (lint + test on GitHub-hosted runners) is separate and is the
one required for PRs to merge — don't fold checks into the orchestration
workflows above.

## Conventions

- The `~/.mycelium-*` paths (`~/.mycelium-busy`, `~/.mycelium-runner-config`)
  are a fleet contract with machines already onboarded via
  `join_network.sh` — renaming them orphans the fleet, not just this repo.
- `mcm drain`/`mcm undrain` toggle two layers in order: the `ready`
  runner label (assignment) then the busy file (execution). Keep that
  order when touching `machine.py`.
- No emojis in tool/CLI output. ASCII status indicators only.
- Comments: only when the *why* is non-obvious — this repo's existing
  comments (e.g. `queue.py`'s claim-race handling) are the calibration
  for how much is enough.

## Working in this repo

```
uv venv && uv pip install -e . --group dev
uv run pytest -q
uvx ruff check .
```

`MCM_STORE` defaults to `./store` inside this checkout.

## Sibling repos

- [mcm-reishi](https://github.com/finngi/mycelium/tree/main/packages/reishi) — the contract layer this repo executes against.
- [mcm-enoki](https://github.com/finngi/mycelium/tree/main/packages/enoki) — the other executor (KubeRay, cloud GPU/TPU).

See `CONTRIBUTING.md` for commit conventions, the PR/CI gate, and how to
join the mesh as a runner.

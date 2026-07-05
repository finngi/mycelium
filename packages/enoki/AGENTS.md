# AGENTS.md — mcm-enoki

Guidance for AI coding agents working in this repo. `CLAUDE.md` in this
repo is just `@AGENTS.md` — this file is the canonical source.

## What this repo is

enoki is mcm's KubeRay (cloud GPU/TPU) execution layer: `l4`, `h100`,
`v5e` accelerators. `mcm experiment submit` templates `jobs/rayjob.yaml`
and applies it with `kubectl` against the `train` namespace on GKE; the
image built from `jobs/Dockerfile` runs `enoki.driver` as the RayJob
entrypoint. Only `l4` has a verified node selector/toleration today.

## Repo shape

Depends on [mcm-reishi](https://github.com/finngi/mcm-reishi)
via a local path (`uv.sources`, `../mcm-reishi`) — check it out as a
sibling directory. Two dependency groups:

- `dev` — base package + pytest. This is all CI installs.
- `cluster` — torch/ray/transformers/peft/accelerate/google-cloud-storage,
  pinned against the training image's CUDA 12.4 base. Only the training
  image (`jobs/Dockerfile`) installs this; GitHub-hosted CI runners don't
  have a GPU, so lint/test never pull it in.

`enoki.trainers` and `enoki.store_backend` both import their heavy/
optional deps lazily inside the function that needs them — importing the
module itself must stay safe with only `dev` installed. Keep that
invariant when adding a trainer or backend.

## Conventions

- Comments: only when the *why* is non-obvious (see `pyproject.toml`'s
  torch/transformers pins for the pattern — each one records a real
  failure mode it prevents, not just "pinned for stability").
- No emojis in tool/CLI output. ASCII status indicators only.
- `driver.run()` swaps in the Postgres store via `MCM_PG_DSN` when set;
  with no DSN it falls back to mcm's local filesystem store, same as a
  laptop run. Never assume one or the other in new code — check the env.

## Working in this repo

```
uv venv && uv pip install -e . --group dev
uv run pytest -q
uvx ruff check .
```

Add `--group cluster` to actually exercise a trainer locally (needs a
GPU to be useful, but will install/import on CPU too).

## Sibling repos

- [mcm-reishi](https://github.com/finngi/mcm-reishi) — the contract layer this repo executes against.
- [mcm-oyster](https://github.com/finngi/mcm-oyster) — the other executor (self-hosted Mac mesh, `mlx`).

See `CONTRIBUTING.md` for commit conventions and the PR/CI gate.

# AGENTS.md — enoki

enoki is mcm's KubeRay (cloud GPU/TPU) execution layer: `l4`, `h100`,
`v5e` runtimes. `mcm experiment submit` templates `jobs/rayjob.yaml`
and applies it with `kubectl` against the `train` namespace on GKE; the
image built from `jobs/Dockerfile` runs `enoki.driver` as the RayJob
entrypoint. Only `l4` has a verified node selector/toleration today.

## Package shape

Depends on [`packages/reishi`](../reishi/AGENTS.md) as a workspace member
(root `pyproject.toml`'s `[tool.uv.sources]`) — no sibling checkout needed.
Two dependency groups, scoped to this package:

- `dev` — base package + pytest. This is all CI installs.
- `cluster` — torch/ray/psycopg/transformers/peft/accelerate/google-cloud-storage,
  pinned against the training image's CUDA 12.4 base. Only the training
  image (`jobs/Dockerfile`) installs this; `uv sync --all-extras` (what CI
  runs) never pulls in a dependency group, so lint/test never see it even
  on a GPU-less runner.

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
  with no DSN it falls back to reishi's own default store backend (sqlite,
  unless `MCM_STORE_BACKEND=fs`), same as a laptop run. Never assume one or
  the other in new code — check the env.

## Working in this package

```
uv sync --all-extras           # from the repo root
uv run pytest packages/enoki -q
uvx ruff check .
```

From `packages/enoki`, add `--group cluster` to actually exercise a trainer
locally (needs a GPU to be useful, but will install/import on CPU too).

## Sibling packages

- [`packages/reishi`](../reishi/AGENTS.md) — the contract layer this package executes against.
- [`packages/oyster`](../oyster/AGENTS.md) — the other executor (self-hosted mesh, currently `mlx` on Apple Silicon).

See `CONTRIBUTING.md` for commit conventions and the PR/CI gate.

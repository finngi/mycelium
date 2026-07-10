# AGENTS.md — mycelium

mycelium is the mcm family monorepo: the experiment contract layer for
small-model training plus its execution and search layers, packaged as one
`uv` workspace. Each package has its own `AGENTS.md` — read it before
working inside that package.

| Package | What it is |
|---|---|
| [`packages/reishi`](packages/reishi/AGENTS.md) | The contract layer: Task, Dataset, Recipe, Trial, Board primitives. Depends on none of the others. |
| [`packages/enoki`](packages/enoki/AGENTS.md) | KubeRay (cloud) execution layer for mcm recipes. |
| [`packages/oyster`](packages/oyster/AGENTS.md) | Self-hosted mesh execution (GH Actions transport, pull-based claims). |
| [`packages/physarum`](packages/physarum/AGENTS.md) | Optuna hyperparameter-search adaptor for mcm sweeps. |

enoki, oyster, and physarum depend on reishi and extend its CLI via
`mcm.plugins` entry points; reishi never imports them.

Published to PyPI as one distribution, `mcm-mycelium`, whose wheel ships all
four import packages inline. Extras are module-named (`enoki`, `oyster`,
`physarum`, `all`); `enoki`'s extra is deliberately empty — its KubeRay/cluster
stack is pinned against the training image's own CUDA base and installed
there, not via a pip extra.

## Working in this repo

```plaintext
uv sync --all-extras
uv run pytest packages/<package> -q
uvx ruff check .
uvx ruff format --check .
```

Run pytest per package, never pooled across the workspace: reishi's task
registry and each plugin's conftest mutate shared global state, so a pooled
session cross-contaminates (reishi and oyster both register a task named
`fixture`). CI mirrors this with a per-package matrix.

## Conventions

- ASCII status indicators only in tool output (`[OK]`, `[FAIL]`, `[WARN]`,
  `[INFO]`, `->`) — output is parsed and grepped, never just read.
- Comment only when the *why* is non-obvious; never narrate what the code
  does.
- CLI stdout stays parseable (`-o json` everywhere); human-facing echo goes
  to stderr.

Tool-specific agent configs (`.claude/`, `.codex/`) and beads issue
tracking are local-only and deliberately untracked.

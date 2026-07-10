# AGENTS.md — reishi

reishi is the experiment contract layer for small-model training: it defines
what a training run *means* — reproducible, scored identically, comparable
on one board — independent of what executes it. It imports no accelerator
or executor code (no Ray, no MLX); execution lives in sibling packages
(`packages/enoki`, `packages/oyster`), which depend on reishi and extend its
CLI via `mcm.plugins` entry points.

## Package shape

`src/reishi/` is the one workspace member with no dependency on the others
(root `pyproject.toml`'s `[tool.uv.sources]`).

| Primitive | What it is |
|---|---|
| `Task` | Output schema + codec + constrained decoder + scorer. |
| `Dataset` | Versioned `gs://` prefix + card + leak contract (`dataset.leaks()`). |
| `Recipe` | model x dataset x prompt x `hparams`; `runtime` (`cpu`/`mlx`/`l4`/`h100`/`v5e`) selects the executor. |
| `Trial` | One recipe x seed execution — a manifest, not a log line. |
| `Board` | Aggregation over trial manifests; computed, never stored as truth. |

`reishi.store` is a `StorageBackend` Protocol selected by `MCM_STORE_BACKEND`:
`sqlite` (default) or `fs` (`LocalFilesystemBackend`, one JSON file per
manifest). Executors swap in their own via `store.use_backend()` — e.g.
enoki's `PostgresBackend` — so reishi's primitives never know which one is
active.

## Conventions

1. **Grammar is closed and disjoint**, enforced by `tests/test_grammar.py`:
   domains and verbs are separate closed vocabularies, every verb has one
   home domain, an omitted action defaults to something read-only.
   Plugins extend the grammar via `grammar.extend()`, which raises on any
   collision — never resolve a collision by silently letting one side win.
2. **`-o json` everywhere.** Canonical-form echo goes to stderr so stdout
   stays parseable.
3. **Plugins own their domain.** `mcm.plugins` entry points contribute
   `DOMAINS`, `VERBS`, `HANDLERS` — a broken plugin degrades to a `[WARN]`
   on stderr, never a dead CLI.
4. **No emojis in tool output.** ASCII status indicators only (`[OK]`,
   `[FAIL]`, `[WARN]`, `[INFO]`, `->`).
5. **Comments: only when the *why* is non-obvious.** Never narrate *what*
   the code does.

## Working in this package

```bash
uv sync --all-extras           # from the repo root
uv run pytest packages/reishi -q
uv run reishi-admin tasks      # dev-only CLI; deployments expose this as `mcm`
uvx ruff check .
```

`MCM_STORE` overrides the manifest store root (default `~/.mcm/store`).

## Depended on by

- [`packages/enoki`](../enoki/AGENTS.md) — KubeRay (cloud) execution: `l4`, `h100`, `v5e`.
- [`packages/oyster`](../oyster/AGENTS.md) — self-hosted mesh execution (currently `mlx` on Apple Silicon).

See `CONTRIBUTING.md` for commit conventions and the PR/CI gate.

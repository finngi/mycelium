# AGENTS.md — physarum

physarum is the Optuna hyperparameter-search adaptor for
[reishi](../reishi/AGENTS.md): one Optuna-suggested point in a search space
becomes one ordinary `Recipe`,
planned and run through the exact same `Producer` contract oyster and enoki
already implement. physarum defines no execution of its own — it is a
scheduler over recipes that already work.

v1 scope, deliberately: **in-process execution only, no mid-producer
pruning, no claim semantics.** (Post-hoc pruning works: Optuna can mark a
completed trial `pruned` and exclude it from best -- what's out of scope is
aborting a producer mid-run.) `study.optimize()` runs in physarum's own process and calls a
resolved `Producer` directly and synchronously — no queue, no distributed
study, no mid-training callback. Widening any of that is a real design
decision, not a default to reach for; see the three gaps below before
touching any of it.

## Package shape

`src/physarum/`. Depends on [`packages/reishi`](../reishi/AGENTS.md) as a
required workspace member (for `Task`/`Dataset`/`Recipe`/`Trial` and
`reishi.store`) and optionally [`packages/oyster`](../oyster/AGENTS.md),
installed via this package's own `mlx` extra, only needed to actually run a
sweep whose template targets `runtime: mlx`. A `cpu` extra (trafilatura)
covers `runtime: cpu` sweeps instead, physarum's own in-process producer.

| Module | What it is |
|---|---|
| `primitives/sweep.py` | `Sweep`: a recipe template + search space + goal + sampler — physarum's own primitive, layered on reishi's `Recipe`. |
| `objective.py` | Turns one `Sweep` + one already-resolved `Producer` into an `optuna.Trial -> float` function. Optuna never sees a `Recipe`; this module is the only place that translates suggested scalars into one. |
| `mcm_plugin.py` | The `sweep` domain: `mcm sweep optimize sweep.yaml`, `mcm sweep watch <name>`. Resolves which `Producer` to call from `template.runtime`. |
| `producers/trafilatura_extract.py` | The `cpu` runtime's `Producer`: scores an Optuna-suggested trafilatura extraction config against a recipe's dataset — no gradient step, no model. |
| `watch.py` | `mcm sweep watch`: a read-only localhost HTTP server that polls `reishi.store` and graphs a sweep's trials live. |

## Three deliberate gaps (do not close without a real reason)

These were each the subject of an explicit design review before v1 shipped
without them — read the reasoning before reopening any of them:

1. **No mid-producer pruning.** Needs a `Producer` contract change (`mlx_lm.lora.run()`
   currently swallows its own `training_callback` argument — supporting
   this means calling `train_model()` directly, not a small addition) and,
   for distributed mode, a report/abort channel split across two separate
   store records (never the same manifest field two writers touch — that's
   a lost-update race on Postgres, only accidentally safe on git).
2. **No claim/coordination semantics.** enoki's `claim_next` (Postgres row
   locking) and oyster's `queue.claim()` (optimistic git-push-race) solve
   genuinely different problems, not one problem twice — do not generalize
   from two data points. If a shared primitive is ever needed, it's
   compare-and-swap on `save()`, not a hoisted `claim_next`. Also: check
   whether Optuna's own distributed storage (`RDBStorage`/journal,
   `study.ask()`) already solves this before assuming physarum needs
   anything from reishi at all.
3. **No sweep-manifest persistence.** A `Sweep`'s progress lives in
   Optuna's own study object for now (in-memory unless you pass
   `storage=` yourself) — only the `Trial`s it produces land in
   `reishi.store`. `mcm sweep list`/`describe` aren't implemented because
   there's no persisted `Sweep` state to read yet.

## Conventions

Same as reishi's (this package inherits its grammar and its rules):
1. **Grammar is closed and disjoint.** This package adds two verbs, `optimize`
   and `watch`, both homed under `sweep` — not `run` (already home to
   `recipe`).
2. **`-o json` everywhere.** Canonical-form echo goes to stderr.
3. **No emojis in tool output.** ASCII status indicators only (`[OK]`,
   `[FAIL]`, `[WARN]`, `[INFO]`, `->`).
4. **Comments: only when the *why* is non-obvious.**

## Working in this package

```
uv sync --all-extras           # from the repo root
uv run pytest packages/physarum -q
```

`uv sync --all-extras` at the repo root installs oyster's `mlx` stack and
trafilatura for `cpu` too — no sibling checkout needed for either.

## Sibling packages

- [`packages/reishi`](../reishi/AGENTS.md) — the contract layer this depends on.
- [`packages/oyster`](../oyster/AGENTS.md) — self-hosted mesh execution (currently `mlx` on Apple Silicon).
- [`packages/enoki`](../enoki/AGENTS.md) — KubeRay (cloud) execution: `l4`, `h100`, `v5e`
  (not yet wired into `_resolve_producer` — `mlx` (via oyster) and `cpu`
  (physarum's own in-process trafilatura producer) are the only runtimes
  physarum can actually dispatch to today).

See `CONTRIBUTING.md` for commit conventions and the PR/CI gate.

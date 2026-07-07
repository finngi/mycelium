# AGENTS.md — mcm-physarum

Guidance for AI coding agents (Claude Code, GitHub Copilot, Cursor, Codex,
etc.) working in this repo. `CLAUDE.md` in this repo is just `@AGENTS.md` —
this file is the canonical source, tool-specific files import it.

## What this repo is

physarum is the Optuna hyperparameter-search adaptor for
[reishi (mcm)](https://github.com/finngi/mcm-reishi):
one Optuna-suggested point in a search space becomes one ordinary `Recipe`,
planned and run through the exact same `Trainer` contract oyster and enoki
already implement. physarum defines no execution of its own — it is a
scheduler over recipes that already work.

v1 scope, deliberately: **in-process execution only, no pruning, no claim
semantics.** `study.optimize()` runs in physarum's own process and calls a
resolved `Trainer` directly and synchronously — no queue, no distributed
study, no mid-training callback. Widening any of that is a real design
decision, not a default to reach for; see the three gaps below before
touching any of it.

## Repo shape

Single package, `src/physarum/`, installed with `uv`. Depends on
`../mcm-reishi` (required path dependency, for `Task`/`Dataset`/`Recipe`/
`Trial` and `reishi.store`) and optionally `../mcm-oyster` (only needed to
actually run a sweep whose template targets `accelerator: mlx` — install
via the `mlx` extra).

| Module | What it is |
|---|---|
| `primitives/sweep.py` | `Sweep`: a recipe template + search space + objective + sampler — physarum's own primitive, layered on reishi's `Recipe`. |
| `objective.py` | Turns one `Sweep` + one already-resolved `Trainer` into an `optuna.Trial -> float` function. Optuna never sees a `Recipe`; this module is the only place that translates suggested scalars into one. |
| `mcm_plugin.py` | The `sweep` domain: `mcm sweep optimize sweep.yaml`, `mcm sweep watch <name>`. Resolves which `Trainer` to call from `template.accelerator`. |
| `trainers/trafilatura_extract.py` | The `local` accelerator's `Trainer`: scores an Optuna-suggested trafilatura extraction config against a recipe's dataset — no gradient step, no model. |
| `watch.py` | `mcm sweep watch`: a read-only localhost HTTP server that polls `reishi.store` and graphs a sweep's trials live. |

## Three deliberate gaps (do not close without a real reason)

These were each the subject of an explicit design review before v1 shipped
without them — read the reasoning before reopening any of them:

1. **No pruning.** Needs a `Trainer` contract change (`mlx_lm.lora.run()`
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

Same as reishi's (this repo inherits its grammar and its rules):
1. **Grammar is closed and disjoint.** This repo adds two verbs, `optimize`
   and `watch`, both homed under `sweep` — not `run` (already home to
   `recipe`).
2. **`-o json` everywhere.** Canonical-form echo goes to stderr.
3. **No emojis in tool output.** ASCII status indicators only (`[OK]`,
   `[FAIL]`, `[WARN]`, `[INFO]`, `->`).
4. **Comments: only when the *why* is non-obvious.**

## Working in this repo

```
uv venv && uv pip install -e . --group dev
uv run pytest -q
```

Requires `../mcm-reishi` checked out as a sibling directory (`uv.sources`
path dependency); `../mcm-oyster` too if you're actually running an `mlx`
sweep rather than just testing the scheduling logic.

## Sibling repos

- [mcm-reishi](../mcm-reishi) — the contract layer this depends on.
- [mcm-oyster](../mcm-oyster) — self-hosted mesh execution (currently `mlx` on Apple Silicon).
- [mcm-enoki](../mcm-enoki) — KubeRay (cloud) execution: `l4`, `h100`, `v5e`
  (not yet wired into `_resolve_trainer` — `mlx` (via oyster) and `local`
  (physarum's own in-process trafilatura trainer) are the only accelerators
  physarum can actually dispatch to today).

See `CONTRIBUTING.md` for commit conventions and the PR/CI gate.

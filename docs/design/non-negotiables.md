# Implementation rules — the gate

Every proposed change or PR is checked against this document before it is
opened. **Hard boundaries** are non-negotiable: a change that crosses one is
wrong even if it works. **Practices** are strong defaults: deviating is
allowed, but the PR must say why. The gate at the bottom is the checklist to
run last.

Tier vocabulary (core layer / stable protocols / standard library) and the
stability ladder (`stable`/`provisional`/`experimental`/`internal`) are
defined in `architecture.md` (a local design doc in this directory, also
published as the "reishi architecture" artifact) — this document assumes
them.

## Hard boundaries

1. **Dependency direction.** reishi's dependency closure is the contract
   layer only: it never imports a plugin or any of a plugin's third-party
   stack. Plugins import reishi, never the reverse — otherwise one broken
   optional stack takes down the contract layer for everyone. Entry-point
   discovery (`importlib.metadata`) is runtime name resolution, not a static
   import, and is the sanctioned way for core to reach plugin code; a failing
   entry point degrades to a `[WARN]`.

2. **Protocol changes are never silent.** Anything in the stable-protocols
   tier — manifest TypedDicts, `StorageBackend`/`Scorable`/`Producer`
   signatures, the `(kind, name) -> opaque JSON` store rule — defaults to
   additive: new optional fields are free. A breaking change (rename,
   removal, retype, changed meaning) is permitted before 1.0 — the alpha
   exists to get these shapes right — but only through the gate: validated
   against every module that reads the shape (round-trip tests included),
   declared as breaking in the PR, and carrying the right version
   consequence (0.x: it rides the prerelease bump; post-1.0: a major). The
   litmus test for "is this breaking": could it break a manifest already on
   disk, or a sibling module's implementation you cannot see? The
   non-negotiable is the process — an undeclared or unvalidated breaking
   change is wrong even if it works.

3. **The additivity round-trip.** Every load -> mutate -> save path must
   preserve manifest keys it does not recognise (`Trial.from_manifest` routes
   unknown keys to `extra`; every store backend persists the whole JSON
   document unshredded, no fixed columns). One lossy reader is enough to
   destroy a newer field forever — oyster's heartbeat re-saves every 30s, so
   the field is gone on the first save, not slowly eroded. A new reader or
   backend ships with a round-trip test proving unknown keys survive.

4. **The standard library never becomes a requirement.** `run_eval`,
   `field_aggregate`, the codecs, the producer registry, the local executor —
   all bypassable. A Producer only has to satisfy the protocol (return a
   metrics dict). If a change makes a stdlib helper the only way to satisfy
   a protocol, it has silently promoted convenience into contract.

5. **Grammar is closed and disjoint.** CLI domains and verbs are separate
   closed vocabularies; every verb has one home domain; an omitted action
   defaults to something read-only. Plugins extend via `grammar.extend()`,
   which raises on collision — never resolve a collision by letting one side
   silently win, because "which command runs" would then depend on plugin
   import order.

6. **Provenance separation: `metrics` vs `observables`.** The scorer judges
   answers; only the executor can observe what the run cost. Quality numbers
   go in `metrics` (written by the scoring side, with `ScoringInfo`
   provenance); run-resource facts go in `observables` (written by the executor,
   unit-suffixed: `wall_time_s`, `cost_usd`). Mixing them corrupts K-pinning:
   a Board cannot tell a scored quantity from a measured one.

7. **Trials are comparable only within one K.** K = (task, codec, scorer
   closure, aggregator, dataset version, split, n_eval). No feature may
   compare or aggregate trials across different K as if they were the same
   measurement — a different split or scorer version is a different
   instrument, not more noise.

8. **The Board is computed, never stored as truth.** Trial manifests are the
   samples; every Board is derived from them on demand. Persisting a Board
   creates a second source of truth that drifts from the manifests it
   summarises.

## Practices

- **Conventional commits drive versioning.** release-please reads commit
  types (`feat:`/`fix:` bump, `chore:`/`docs:` don't) — a mislabelled type
  mis-versions the next release. Version markers in source use
  release-please's semver spelling (`0.0.0-a.N`), which normalises to the
  same PEP 440 version; the compact spelling breaks the updater regex.
- **Entry points live in both places.** The dev workspace reads each member's
  `pyproject.toml`; the published wheel is built from the root
  `pyproject.toml`. An entry point declared in only one of them works in one
  environment and silently vanishes in the other.
- **New manifest fields are `NotRequired`,** unit-suffixed where they carry a
  quantity, and — when protocol-tier — reflected in `architecture.md` in the
  same PR, so the doc and the contract cannot drift.
- **Tool output is ASCII.** `[OK]` `[FAIL]` `[WARN]` `[INFO]` `->` — no
  emoji. Canonical-form echo goes to stderr so `-o json` stdout stays
  parseable.
- **Comments carry the why, not the what.** A comment restating the line
  drifts into a lie on the next edit; reasoning outlives implementation.
- **No rhetorical redundancy.** Stating a rule and then restating its
  negation ("degrades to a `[WARN]`, never a dead CLI") says one thing
  twice; write the rule once and stop. Applies to code comments and docs
  alike — flair spends review attention without adding information.
- **Docs have one canonical home.** `AGENTS.md` is canonical per package;
  `CLAUDE.md` is `@AGENTS.md`. Design rationale lives in `docs/design/`;
  don't fork it into READMEs.

## The gate

Run before opening any PR:

```text
[ ] Named the tier(s) touched (core / stable protocols / standard library)
    in the PR description.
[ ] Any protocol-tier change: additive? litmus test passed? round-trip
    test added? architecture.md updated?
[ ] No new static import of plugin code (or its third-party stack) inside
    reishi.
[ ] Grammar untouched, or extended via grammar.extend() with no collision.
[ ] metrics / observables provenance separation respected.
[ ] Commit types match the intended version bump.
[ ] Output ASCII; comments why-only.
[ ] Names match the glossary below; no rejected term reintroduced.
```

## Glossary — ratified vocabulary

Ratified 2026-07-10 from the antagonistic nomenclature review (all 17
findings dispositioned). Canonical names below are binding for new code and
docs; where the codebase still carries a pre-ratification name, the rename
batch (tracked in beads) brings it in line. One-sentence definitions are the
contract — argue with the definition, not the word.

### Primitives and records

| Term | Definition |
|---|---|
| `Task` | The scoring instrument: output schema + codec + scorer + aggregator. A component of the measurement key, held constant across an experiment. |
| `Dataset` | Versioned data identity + card + leak contract. `advisory_task` (was `task`) hints but never binds. |
| `Recipe` | One frozen configuration theta: model x dataset x prompt x hparams. |
| `Trial` | One Recipe x seed execution manifest — one sample e_K(theta, omega). Kept over `Run` (rejected: collides with the CLI's `run` verb); Optuna's trial object is only ever `Suggester`/`ot` in code and never called a "trial" in prose. |
| `Board` | The estimator over trial manifests — computed on demand, never stored. |
| `Comparison` | Pairwise/relative measurement record beside Trial (anchored comparisons only). |
| experiment | Not a primitive: the whole frame (Theta, e, K, preceq). |

### Execution

| Term | Definition |
|---|---|
| `Producer` | The callable contract `(TrialManifest) -> ProducerResult` — the formalism's T:(theta, omega) -> a; may or may not learn (identity producers are legal). Replaces "Trainer" as the contract name. |
| trainer | Reserved for real gradient-descent Producer implementations (`mlx_lora`, `train_l4`) — never the contract, never the config dict. |
| `hparams` | `Recipe.hparams` — the free-form hyperparameter dict (was `Recipe.trainer`). |
| `runtime` | `Recipe.runtime` (was `accelerator`) — the named execution environment a trial requires: determines who claims it, which Producer implementation runs it, and what infrastructure is provisioned. Values: `cpu` (was `local` — dissolves the collision with the local-executor placement sense), `mlx`, `l4`, `h100`, `v5e`. |
| accelerator | Reserved for actual silicon (`l4`/`h100`/`v5e`); no longer a field name. |
| plugin | An opt-in module built on the reishi core: a construction of the core integrated with alternate packages for specific functionality or features. Contributes CLI domains via `mcm.plugins` and Producers via `mcm.producers`. enoki and oyster are execution plugins; physarum is a search plugin. |
| executor | The execution-flavoured plugin layer: drives Trials through `planned -> running -> done/failed` and writes `execution`, `observables`, artifacts. |
| `runner` | The machine/process identity in `ExecutionInfo.runner`. |
| artifact | The produced thing that induces f_a (weights, config, prompt). `TrialArtifacts.outputs` (was `predictions`) holds persisted raw model outputs — outputs are not artifacts. |

### Measurement

| Term | Definition |
|---|---|
| score | Per-example output of `Task.score(pred, ref)` — sufficient statistics, not a metric. (`ref`, ratified earlier over `gold` and `y`: whatever the row carries — reference, source, or context.) |
| `metrics` | Aggregated quality numbers only, written by the scoring side. |
| `observables` | Executor-measured run-resource facts, unit-suffixed (`wall_time_s`, `cost_usd`), disjoint from metrics. |
| scoring | The act (module `scoring.py`, was `eval.py`); `ScoringInfo` (was `EvalInfo`) is its provenance record; manifest keys `scoring`/`scorings` (was `eval`/`evals`). |
| eval set | The held-out data — a split of a versioned Dataset; `n_eval_rows` (was `eval_n`) is its row count. |
| `scored_on` | Where scoring ran (was `Placement`/`placement`); values `cpu`/`gpu`/`tpu`. |
| measurement key (K) | (task, codec, scorer closure, aggregator, dataset revision, split, n_eval_rows). Trials comparable iff K matches. |
| `recipe_name` | `Trial.recipe_name` (was `Trial.recipe`) — the Recipe's name string. `Trial.spec` is unchanged: the embedded RecipeManifest. |
| `n_seeds` | `Recipe.n_seeds` (was `seeds`) — how many seeds to plan. `Trial.seed` is the root seed from which named sub-streams derive. |
| `train_dataset` / `eval_dataset` | Recipe's data fields (was single `dataset`); each optional, at least one required — gives the leak contract something to check. |

### Search (physarum)

| Term | Definition |
|---|---|
| `goal` | The sweep config block `{metric, direction, constraints}` (was `objective`). |
| preference | The order (preceq) a goal induces — the math word, docs only. |
| `trial_fn` | The internal callable handed to the Suggester loop (was the inner `objective`). |
| sampler | The search algorithm string (`tpe`, ...). `Suggester` is the interface it satisfies. "Search backend" is banned from prose. |

### Rejected (do not reintroduce)

- `Run` for reishi's Trial — collides with the closed grammar's `run` verb.
- `y` for the scorer's second argument — `ref` ratified (uex.13).
- `comparison_key` — "measurement key / K" ratified (uex.9).
- "two rings" / "frozen contract" — the tier vocabulary ratified (uex.3).

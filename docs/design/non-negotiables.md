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

1. **Dependency direction.** reishi imports no accelerator (no Ray, no MLX,
   no torch) and never imports an executor module. Executors import reishi,
   never the reverse — otherwise one broken accelerator stack takes down the
   contract layer for everyone. Entry-point discovery (`importlib.metadata`)
   is runtime name resolution, not a static import, and is the sanctioned way
   for core to reach plugin code; a failing entry point degrades to a
   `[WARN]`, never a dead CLI.

2. **Protocol changes are additive.** Anything in the stable-protocols tier —
   manifest TypedDicts, `StorageBackend`/`Scorable`/`Trainer` signatures, the
   `(kind, name) -> opaque JSON` store rule — may gain optional fields but may
   not rename, remove, retype, or change the meaning of existing ones. The
   litmus test: could this change break a manifest already on disk, or a
   sibling module's implementation you cannot see? If yes, it is a breaking
   protocol change and must be called out as such in the PR (under 0.x it
   rides a minor bump; post-1.0 it costs a major).

3. **The additivity round-trip.** Every load -> mutate -> save path must
   preserve manifest keys it does not recognise (`Trial.from_manifest` routes
   unknown keys to `extra`; every store backend persists the whole JSON
   document unshredded, no fixed columns). One lossy reader is enough to
   destroy a newer field forever — oyster's heartbeat re-saves every 30s, so
   the field is gone on the first save, not slowly eroded. A new reader or
   backend ships with a round-trip test proving unknown keys survive.

4. **The standard library never becomes a requirement.** `run_eval`,
   `field_aggregate`, the codecs, the trainer registry, the local executor —
   all bypassable. A producer only has to satisfy the protocol (a Trainer
   only has to return a metrics dict). If a change makes a stdlib helper the
   only way to satisfy a protocol, it has silently promoted convenience into
   contract.

5. **Grammar is closed and disjoint.** CLI domains and verbs are separate
   closed vocabularies; every verb has one home domain; an omitted action
   defaults to something read-only. Plugins extend via `grammar.extend()`,
   which raises on collision — never resolve a collision by letting one side
   silently win, because "which command runs" would then depend on plugin
   import order.

6. **Provenance separation: `metrics` vs `observables`.** The scorer judges
   answers; only the executor can observe what the run cost. Quality numbers
   go in `metrics` (written by the scoring side, with `EvalInfo` provenance);
   run-resource facts go in `observables` (written by the executor,
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
- **Docs have one canonical home.** `AGENTS.md` is canonical per package;
  `CLAUDE.md` is `@AGENTS.md`. Design rationale lives in `docs/design/`;
  don't fork it into READMEs.

## The gate

Run before opening any PR:

```
[ ] Named the tier(s) touched (core / stable protocols / standard library)
    in the PR description.
[ ] Any protocol-tier change: additive? litmus test passed? round-trip
    test added? architecture.md updated?
[ ] No new static import of accelerator or executor code inside reishi.
[ ] Grammar untouched, or extended via grammar.extend() with no collision.
[ ] metrics / observables provenance separation respected.
[ ] Commit types match the intended version bump.
[ ] Output ASCII; comments why-only.
```

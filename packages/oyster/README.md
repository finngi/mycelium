# oyster

No-cloud execution layer for [reishi (mcm)](https://github.com/Digital-Insight-Technologies-Ltd/mcm-reishi): a mesh of
self-hosted Macs behind GitHub Actions runners, with tag-based
availability. Like [enoki](https://github.com/Digital-Insight-Technologies-Ltd/mcm-enoki) (the KubeRay layer), oyster consumes
recipe manifests and writes trial manifests to the mcm store — it defines
no shapes of its own. Recipes target the mesh with `accelerator: mlx`.

## Scheduling: pull, not push

Pushing work at runners fails in a specific way: GitHub assigns a queued
job to any online runner — including one mid-training — so the job enters,
fails a guard, and burns a retry, while the work itself is welded to
whichever machine caught it. oyster inverts this: planned trials sit in the
store as a priority queue, and a machine **claims** the next trial that
fits only when it is actually idle, one at a time. A claim is a commit
(the store lives in git), so it is atomic: the loser of a push race
rebases, sees the trial taken, and picks the next one. Dispatched Actions
jobs are generic "be a worker" jobs — safe to over-dispatch, exit in
seconds when nothing is claimable.

| Need | Mechanism |
|---|---|
| Spread load | Trials are the unit, not batches: an 8-trial experiment lands on up to 8 machines. The herder wakes `min(planned, ready-idle)` workers on a schedule. |
| Retract load | `mcm drain`: removes the runner's `ready` label (assignment level — no job lands at all) and sets the busy file (execution level — an already-assigned job fails fast). The in-flight trial finishes; nothing more is claimed. `mcm mesh requeue <id>` yanks a specific trial back. |
| Priority | `priority:` on the recipe flows into `trial.spec`; the queue orders by (priority desc, created asc). |
| Distribute correctly | Assignment happens at claim time, by the machine itself, against its measured memory budget and installed trainers — never by GitHub's blind runner pick. |
| Recover lost runners | The reaper requeues trials whose heartbeat went stale (closed lid, killed service), bounded at 3 attempts, then fails them. |

## Availability is two layers, label first

The `ready` runner label gates *assignment*: remove it and GitHub never
hands the machine a job — no pickup/refuse/retry-burn cycle. The
`~/.mycelium-busy` file gates *execution*: it only catches jobs assigned
before the label came off. `mcm drain`/`mcm undrain` toggle both in that
order. The `~/.mycelium-*` paths are a fleet contract — onboarded machines
already have them; renaming the files orphans the fleet.

## Layout

| Path | What it is |
|---|---|
| `src/oyster/queue.py` | Scheduler: eligibility (fit + trainer + attempts), priority order, atomic claim, requeue |
| `src/oyster/worker.py` | Claim -> train -> record loop; drains between trials |
| `src/oyster/machine.py` | This machine: budget, identity, `ready`-label toggle, busy file |
| `src/oyster/footprint.py` | Conservative unified-memory estimate per trial |
| `src/oyster/gitstore.py` | Store-over-git: pull-rebase before deciding, push-with-retry after; lost race = clean abort |
| `src/oyster/trainers/` | MLX trainers, keyed by accelerator ("adapter" is reserved for the LoRA artifact) |
| `src/oyster/mcm_plugin.py` | Registers the `mesh` domain into the mcm CLI (entry point `mcm.plugins`) |
| `store/` | The mcm manifest store, committed — the queue IS the repo |
| `.github/workflows/worker.yml` | Generic worker on `[self-hosted, macOS, mlx, ready]` |
| `.github/workflows/herder.yml` | Scheduled: dispatches `min(planned, ready-idle)` workers |
| `.github/workflows/reaper.yml` | Scheduled: requeues stale-heartbeat trials |
| `scripts/join_network.sh` | Mac onboarding: capacity detection, two runners (mlx+cpu), keep-awake |

## CLI

oyster has no CLI of its own — installing it teaches `mcm` the `mesh` domain
(one grammar, one canonical echo, one `-o json`):

```
mcm mesh                  # queue counts + this machine's fit
mcm next                  # > mcm mesh next     what this machine would claim
mcm work                  # > mcm mesh work     claim-and-train until drained/empty
mcm drain / mcm undrain   # retract/rejoin this machine (label + busy file)
mcm requeue <trial-id>    # pull a trial back to planned
mcm reap                  # requeue stale-heartbeat trials (what the reaper runs)
```

## Setup

```
uv venv && uv pip install -e . --group dev
uv run pytest -q
```

`MCM_STORE` defaults to `./store` inside this checkout. Submit work from
reishi: `mcm run recipe.yaml --plan` against this store, then let the
herder tick (or `gh workflow run herder.yml`).

## Not here yet

The MLX LoRA trainer (port source: `proto/mycelium/bench.py` — until it
lands, workers find nothing claimable and exit clean), heartbeats *during*
a trial (trainers must call `queue.heartbeat`; the reaper timeout assumes
they do), and the CPU-job lane for dataset builds.

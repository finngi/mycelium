# hyphae

Execution layer for [mycelium (mcm)](../mycelium) on KubeRay (GKE `train`
namespace). mcm defines the shapes — recipes in, trial manifests out — hyphae
is the thing that actually trains.

The boundary, in both directions:

- hyphae **consumes** recipe manifests and **writes** trial manifests to the
  mcm store. It never defines shapes of its own.
- mcm **never imports Ray**. Everything Ray-shaped (RayJob templating, the
  driver, trainers) lives here.

## Layout

| Path | What it is |
|---|---|
| `jobs/rayjob.yaml` | RayJob template, filled in by `mcm experiment submit` — worker group per accelerator (`l4`, `h100`, `v5e`) |
| `jobs/Dockerfile` | Training image, pushed to the `training-images` Artifact Registry repo |
| `src/hyphae/driver.py` | In-cluster entrypoint: recipe manifest -> one trial per seed -> trainer -> trial manifests + artifacts |
| `src/hyphae/trainers/` | Trainers, selected by the recipe's `accelerator` field ("adapter" is reserved for the LoRA artifact) |

## Setup

```
uv venv && uv pip install -e . --group dev
uv run python -m hyphae.driver <recipe.yaml>   # plans trials, then fails: no trainer yet
```

Ray itself is in the `cluster` dependency group — it only needs to exist
inside the training image, not on your laptop.

## Not here yet

The TRL/PEFT trainer (CUDA: `l4`, `h100`), the XLA/JAX trainer (`v5e`),
image build/push, and the `mcm experiment submit` wiring that templates
`jobs/rayjob.yaml`.

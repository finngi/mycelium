# reishi (mcm)

Experiment contract layer for small-model training on KubeRay (GKE `train`
namespace). mcm makes a training run mean something — reproducible, scored
identically, comparable on one board — regardless of what executes it.

## Primitives

| Primitive | What it is |
|---|---|
| **Task** | Output schema + codec + constrained decoder + scorer. The scorer is the cross-accelerator invariant. |
| **Dataset** | Versioned `gs://` prefix + card + leak contract. |
| **Recipe** | Declarative model x dataset x prompt x trainer spec; `accelerator` selects the trainer. |
| **Trial** | One recipe x seed execution (Ray Tune's term) — a manifest, not a log line. |
| **Board** | Aggregation over trial manifests; computed, never stored as truth. |

Execution lives in sibling repos — [enoki](../mcm-enoki) (KubeRay: `l4`,
`h100`, `v5e`) and [oyster](../mcm-oyster) (the self-hosted Mac mesh: `mlx`);
mcm never imports Ray or MLX. Executors consume recipe manifests and write
trial manifests — the store is the only interface.

## CLI

`mcm <domain> <action> <objects> --<parameters>` — every segment optional;
tokens classify by vocabulary, so shorthand canonicalizes deterministically
(echoed on stderr):

```
mcm                      # status
mcm trials               # > mcm trial list
mcm trial 7f3a           # > mcm trial describe 7f3a
mcm logs 7f3a            # > mcm trial logs 7f3a
mcm run parse-org.yaml   # > mcm recipe run parse-org.yaml   (local, in-process)
mcm submit nameparse-x   # > mcm experiment submit nameparse-x  (RayJob)
```

Installed executors extend this one CLI via `mcm.plugins` entry points
(oyster adds a `mesh` domain: `mcm work`, `mcm drain`, ...) — same grammar,
same disjointness law, one help.

Grammar anchors (enforced by `tests/test_grammar.py`):
- domains and verbs are disjoint closed vocabularies; `run` is a VERB, the noun is `trial`
- an omitted action defaults to something read-only
- every verb has one home domain; object names must contain `-./_` or a digit
- `-o json` everywhere; canonical echo goes to stderr so stdout stays parseable

## Setup

```
uv venv && uv pip install -e . --group dev
uv run pytest -q
uv run mcm tasks
```

`MCM_STORE` overrides the manifest store root (default `~/.mcm/store`;
becomes `gs://example-bucket` when the GCS store lands).

## Not here yet

The nameparse scorer port and the GCS store. `experiment submit` now
templates enoki's `jobs/rayjob.yaml` (convention: `experiments/<name>/recipe.yaml`,
image via `--image`/`MCM_TRAIN_IMAGE`) and applies it with `kubectl` --
only the `l4` accelerator has a verified node selector/toleration, others
fail cleanly. Everything Ray-shaped (driver, trainers, RayJob template,
Dockerfile) stays in enoki.

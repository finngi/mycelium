# mycelium (mcm)

Experiment contract layer for small-model training on KubeRay (GKE `train`
namespace). mcm makes a training run mean something — reproducible, scored
identically, comparable on one board — regardless of what executes it.

Successor to `proto/mycelium` (now the frozen research archive); learnings
migrated, code mostly didn't.

## Primitives

| Primitive | What it is |
|---|---|
| **Task** | Output schema + codec + constrained decoder + scorer. The scorer is the cross-accelerator invariant. |
| **Dataset** | Versioned `gs://` prefix + card + leak contract. |
| **Recipe** | Declarative model x dataset x prompt x trainer spec; `accelerator` selects the trainer adapter. |
| **Trial** | One recipe x seed execution (Ray Tune's term) — a manifest, not a log line. |
| **Board** | Aggregation over trial manifests; computed, never stored as truth. |

Only `driver.py` and the CLI ever know Ray exists.

## CLI

`mcm <domain> <action> <objects> --<parameters>` — every segment optional;
tokens classify by vocabulary, so shorthand canonicalizes deterministically
(echoed on stderr):

```
mcm                      # status
mcm trials               # > mcm trial list
mcm trial 7f3a           # > mcm trial describe 7f3a
mcm logs 7f3a            # > mcm trial logs 7f3a
mcm run example.yaml   # > mcm recipe run example.yaml   (local, in-process)
mcm submit extract-x   # > mcm experiment submit extract-x  (RayJob)
```

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

Trainer adapters (TRL/PEFT for CUDA, XLA/JAX for v5e), the Ray driver,
RayJob templating in `jobs/`, the extract scorer port, GCS store.

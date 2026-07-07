# physarum (mcm)

Optuna hyperparameter-search adaptor for
[reishi (mcm)](https://github.com/finngi/mcm-reishi):
one Optuna-suggested point in a search space becomes one ordinary `Recipe`,
planned and run through the exact same `Trainer` contract
[oyster](https://github.com/finngi/mcm-oyster) and
[enoki](https://github.com/finngi/mcm-enoki)
already implement. physarum defines no execution of its own.

## What a sweep is

A `Sweep` is a recipe template plus a search space over its `trainer`
hyperparameters, an objective metric, and a sampler:

```yaml
name: nameparse-lora-sweep-1
template:
  name: nameparse-lora-sweep-1   # overridden per-trial; still required
  task: nameparse
  dataset: nameparse-v3
  base_model: mlx-community/Qwen2.5-7B-Instruct-4bit
  accelerator: mlx
  prompt: prompts/nameparse_v2.txt
search_space:
  trainer.lr:    { type: loguniform, low: 1e-6, high: 1e-4 }
  trainer.rank:  { type: categorical, choices: [4, 8, 16, 32] }
  trainer.iters: { type: int, low: 200, high: 2000, step: 200 }
objective: { metric: f1, direction: maximize }
sampler: tpe
n_trials: 40
```

```
mcm sweep optimize sweep.yaml
```

Each of the `n_trials` suggestions becomes a real `Trial` in the mcm
store — same manifests, same `mcm trial describe <id>`, same board — with
no separate physarum-only bookkeeping.

`accelerator: local` skips oyster/mlx entirely: physarum's own trafilatura
extraction trainer runs in-process, no model or gradient step involved. See
`experiments/htmlmd-trafilatura/sweep.yaml` for a full grid-search example.

## v1 scope

In-process only: `study.optimize()` calls the resolved `Trainer` directly
and synchronously, one suggestion at a time, in physarum's own process.
No pruning, no distributed study, no claim semantics — see `AGENTS.md` for
why each of those was deliberately left out rather than overlooked.

## Setup

```
uv venv && uv pip install -e . --group dev
uv run pytest -q
```

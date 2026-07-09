# mycelium

Experiment management for statistical optimisation, with integrations for
portable executors and architectures — built for small-model fine-tuning.

> **Alpha.** Versioned as `0.0.0aN` prereleases; every release may break the
> API. Published on PyPI as **`mcm-mycelium`** (a [PEP 541 request](https://github.com/pypi/support/issues/11421)
> for the bare `mycelium` name is pending).

## Install

```bash
uv add mcm-mycelium               # contract layer (reishi) only
uv add 'mcm-mycelium[physarum]'   # + Optuna sweep tooling
uv add 'mcm-mycelium[oyster]'     # + MLX mesh-executor deps (Apple Silicon)
uv add 'mcm-mycelium[enoki]'      # KubeRay executor (deps are image-managed)
uv add 'mcm-mycelium[all]'
```

One wheel ships all four modules; extras only add each module's third-party
dependencies. Importing a module whose extra is missing fails with a plain
`ModuleNotFoundError` naming the dependency.

## Modules

| Module | What it is |
|---|---|
| `reishi` | The experiment contract layer: Task, Dataset, Recipe, Trial, and Board records, storage backends, and the plugin CLI. |
| `physarum` | Statistical optimisation: Optuna-backed sweeps over recipe search spaces, with epsilon-constraint objectives and pruning. |
| `enoki` | KubeRay (cloud) execution — renders recipes into RayJobs. Its torch/ray stack is installed by the training image, not from PyPI. |
| `oyster` | Self-hosted mesh execution on Apple Silicon (MLX LoRA training, GitHub Actions transport). |

`enoki` and `physarum` register `mcm.plugins` entry points; the `reishi`
plugin CLI discovers them automatically and warns (rather than fails) when an
extra's dependencies are absent.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

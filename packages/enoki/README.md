```
           ____
       _.-'78o `"`--._
   ,o888o.  .o888o,   ''-.
 ,88888P  `78888P..______.]
/_..__..----""        __.'
`-._       /""| _..-''
    "`-----\  `\              .-'~~~-.
            |   ;.-""--..   .'o  oOOOo`.
            | ,8o.  o88. `.:~~~-.oOo   o`.
            `;888P  `788P  :`. \ ~-.  oOOo.
      .o""-.|`-._         ./  `.; / ~.  OO:
     J88 _.-/    ";"-P----'   .'  ;-- `.o.'
     `--'\`|     /  /        ,'  ; ~~--'~
         | /     |  |        ;  ;
_\|/_____\|      |  |akn___\\;_\\//___\|/________
     __\|/___---`---'           _
888888 88b 88  dP"Yb  88  dP 88
88__   88Yb88 dP   Yb 88odP  88
88""   88 Y88 Yb   dP 88"Yb  88
888888 88  Y8  YbodP  88  Yb 88
```
Execution layer for [reishi (mcm)](https://github.com/finngi/mycelium/tree/main/packages/reishi) on KubeRay (GKE `train`
namespace). mcm defines the shapes — recipes in, trial manifests out — enoki
is the thing that actually trains.

The boundary, in both directions:

- enoki **consumes** recipe manifests and **writes** trial manifests to the
  mcm store. It never defines shapes of its own.
- mcm **never imports Ray**. Everything Ray-shaped (RayJob templating, the
  driver, trainers) lives here.

## Layout

| Path | What it is |
|---|---|
| `jobs/rayjob.yaml` | RayJob template, filled in by `mcm experiment submit` — worker group per accelerator (`l4`, `h100`, `v5e`) |
| `jobs/Dockerfile` | Training image, pushed to the `training-images` Artifact Registry repo |
| `src/enoki/driver.py` | In-cluster entrypoint: recipe manifest -> one trial per seed -> trainer -> trial manifests + artifacts |
| `src/enoki/trainers/` | Trainers, selected by the recipe's `accelerator` field ("adapter" is reserved for the LoRA artifact) |

## Setup

```
uv venv && uv pip install -e . --group dev
uv run python -m enoki.driver <recipe.yaml>   # plans trials; running an l4 trial needs --group cluster too
```

Ray and the LoRA trainer stack (torch/transformers/peft) are in the
`cluster` dependency group — they only need to exist inside the training
image, not on your laptop, so `--group dev` alone plans trials but a
laptop run of an `l4` recipe fails inside `train_l4` on the missing heavy
deps (or falls back to running in-process, without Ray, if `ray` itself
isn't installed either).

## Not here yet

The `h100` trainer (the `l4` LoRA trainer is real: plain transformers +
peft, see `src/enoki/trainers/`), the XLA/JAX trainer (`v5e`), and image
build/push. `mcm experiment submit` (in mcm-reishi) now templates and
applies `jobs/rayjob.yaml` for `l4`.

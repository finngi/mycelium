"""Contract tests for the primitives: what a manifest must accept and reject.

These pin the training-method-agnostic core: a recipe without base_model is a
from-scratch run, and a dataset is not bound to any task (the recipe is).
"""

import pytest

import mcm.tasks  # noqa: F401  (populate the task registry)
from mcm.primitives import trial
from mcm.primitives.dataset import Dataset
from mcm.primitives.recipe import Recipe


def _write_recipe(tmp_path, body: str):
    p = tmp_path / "recipe.yaml"
    p.write_text(body)
    return p


FINETUNE = """
name: extract-ft
task: extract
base_model: LiquidAI/LFM2.5-1.2B-Base
dataset: sample-040726
seeds: 2
"""

FROM_SCRATCH = """
name: extract-scratch
task: extract
dataset: sample-040726
accelerator: v5e
trainer:
  arch: {layers: 12, d_model: 768, tokenizer: bpe-32k}
  iters: 100000
"""


def test_finetune_recipe_roundtrip(tmp_path):
    r = Recipe.from_yaml(_write_recipe(tmp_path, FINETUNE))
    r.validate()
    assert r.base_model == "LiquidAI/LFM2.5-1.2B-Base"
    assert r.to_manifest()["base_model"] == r.base_model


def test_from_scratch_recipe_validates(tmp_path):
    r = Recipe.from_yaml(_write_recipe(tmp_path, FROM_SCRATCH))
    r.validate()
    assert r.base_model is None
    assert r.trainer["arch"]["layers"] == 12


def test_from_scratch_recipe_plans_trials(tmp_path):
    r = Recipe.from_yaml(_write_recipe(tmp_path, FROM_SCRATCH))
    trials = trial.plan(r)
    assert len(trials) == 1
    assert trials[0].spec["base_model"] is None


@pytest.mark.parametrize("field", ["name", "task", "dataset"])
def test_recipe_required_fields(tmp_path, field):
    body = "\n".join(
        line for line in FINETUNE.strip().splitlines() if not line.startswith(field)
    )
    with pytest.raises(ValueError, match=field):
        Recipe.from_yaml(_write_recipe(tmp_path, body))


def test_recipe_rejects_unknown_fields(tmp_path):
    with pytest.raises(ValueError, match="unknown recipe fields"):
        Recipe.from_yaml(_write_recipe(tmp_path, FINETUNE + "epochs: 3\n"))


def test_recipe_priority_defaults_and_flows_to_spec(tmp_path):
    r = Recipe.from_yaml(_write_recipe(tmp_path, FINETUNE))
    assert r.priority == 0
    hot = Recipe.from_yaml(_write_recipe(tmp_path, FINETUNE + "priority: 5\n"))
    assert trial.plan(hot)[0].spec["priority"] == 5


def test_trial_manifest_tolerates_unknown_keys():
    t = trial.Trial(id="t-1", recipe="r", seed=0)
    m = t.to_manifest() | {"from_the_future": True}
    assert trial.Trial.from_manifest(m).id == "t-1"


def test_dataset_task_is_optional():
    ds = Dataset(name="pile-mini-040726", uri="gs://example-bucket/pile-mini")
    m = ds.to_manifest()
    assert m["task"] == ""
    assert Dataset.from_manifest(m) == ds


def test_dataset_manifest_without_task_key_loads():
    ds = Dataset.from_manifest({"name": "d-1", "uri": "gs://x/d-1"})
    assert ds.task == ""

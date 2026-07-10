"""Contract tests for the primitives: what a manifest must accept and reject --
a recipe without base_model is a from-scratch run, a dataset need not name a
task, and unknown manifest keys are tolerated.
"""

import pytest

from reishi.primitives import trial
from reishi.primitives.dataset import Dataset
from reishi.primitives.recipe import Recipe


def _write_recipe(tmp_path, body: str):
    p = tmp_path / "recipe.yaml"
    p.write_text(body)
    return p


FINETUNE = """
name: fixture-ft
task: fixture
base_model: LiquidAI/LFM2.5-1.2B-Base
train_dataset: sample-040726
n_seeds: 2
"""

FROM_SCRATCH = """
name: fixture-scratch
task: fixture
train_dataset: sample-040726
runtime: v5e
hparams:
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
    assert r.hparams["arch"]["layers"] == 12


def test_from_scratch_recipe_plans_trials(tmp_path):
    r = Recipe.from_yaml(_write_recipe(tmp_path, FROM_SCRATCH))
    trials = trial.plan(r)
    assert len(trials) == 1
    assert trials[0].spec["base_model"] is None


@pytest.mark.parametrize("field", ["name", "task"])
def test_recipe_required_fields(tmp_path, field):
    body = "\n".join(
        line for line in FINETUNE.strip().splitlines() if not line.startswith(field)
    )
    with pytest.raises(ValueError, match=field):
        Recipe.from_yaml(_write_recipe(tmp_path, body))


def test_recipe_needs_at_least_one_dataset_field(tmp_path):
    body = "\n".join(
        line
        for line in FINETUNE.strip().splitlines()
        if not line.startswith("train_dataset")
    )
    with pytest.raises(ValueError, match="train_dataset.*eval_dataset"):
        Recipe.from_yaml(_write_recipe(tmp_path, body))


def test_recipe_rejects_unknown_fields(tmp_path):
    with pytest.raises(ValueError, match="unknown recipe fields"):
        Recipe.from_yaml(_write_recipe(tmp_path, FINETUNE + "epochs: 3\n"))


def test_recipe_rejects_all_unknown_fields_at_once(tmp_path):
    with pytest.raises(ValueError) as exc_info:
        Recipe.from_yaml(
            _write_recipe(tmp_path, FINETUNE + "epochs: 3\nhparams_kwarg: {}\n")
        )
    assert "epochs" in str(exc_info.value)
    assert "hparams_kwarg" in str(exc_info.value)


def test_recipe_unknown_field_suggests_close_match(tmp_path):
    with pytest.raises(ValueError, match="did you mean 'hparams'"):
        Recipe.from_yaml(_write_recipe(tmp_path, FINETUNE + "hparam: {}\n"))


def test_recipe_priority_defaults_and_flows_to_spec(tmp_path):
    r = Recipe.from_yaml(_write_recipe(tmp_path, FINETUNE))
    assert r.priority == 0
    hot = Recipe.from_yaml(_write_recipe(tmp_path, FINETUNE + "priority: 5\n"))
    assert trial.plan(hot)[0].spec["priority"] == 5


def test_trial_manifest_tolerates_unknown_keys():
    t = trial.Trial(id="t-1", recipe_name="r", seed=0)
    m = t.to_manifest() | {"from_the_future": True}
    loaded = trial.Trial.from_manifest(m)
    assert loaded.id == "t-1"
    # Tolerated means preserved, not silently dropped, so the round trip is lossless.
    assert loaded.to_manifest()["from_the_future"] is True


def test_dataset_advisory_task_is_optional():
    ds = Dataset(name="pile-mini-040726", uri="gs://example-bucket/pile-mini")
    m = ds.to_manifest()
    assert m["advisory_task"] == ""
    assert Dataset.from_manifest(m) == ds


def test_dataset_manifest_without_advisory_task_key_loads():
    ds = Dataset.from_manifest({"name": "d-1", "uri": "gs://x/d-1"})
    assert ds.advisory_task == ""


def test_recipe_validate_rejects_registered_eval_only_train_dataset(
    tmp_path, monkeypatch
):
    from reishi import store
    from reishi.primitives import dataset as dataset_registry
    from reishi.primitives.dataset import Dataset

    monkeypatch.setenv("MCM_STORE", str(tmp_path))
    store.use_backend(store.LocalFilesystemBackend())
    try:
        dataset_registry.save(
            Dataset(name="holdout", uri="x.jsonl", advisory_task="fixture", eval_only=True)
        )
        r = Recipe(name="r", task="fixture", train_dataset="holdout")
        with pytest.raises(ValueError, match="eval_only"):
            r.validate()
        # Names resolve at run time: an unregistered train_dataset passes here
        # (the training producers re-check on load).
        Recipe(name="r2", task="fixture", train_dataset="not-registered").validate()
    finally:
        store.use_backend(store.LocalFilesystemBackend())

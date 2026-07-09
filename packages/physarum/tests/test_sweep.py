"""Contract tests for Sweep: what a sweep manifest must accept and reject."""

import pytest

from physarum.primitives.sweep import Sweep

SWEEP_YAML = """
name: extract-lora-sweep-1
template:
  name: extract-lora-sweep-1
  task: extract-fixture
  dataset: extract-v3
  base_model: mlx-community/Qwen2.5-7B-Instruct-4bit
  accelerator: mlx
  prompt: prompts/extract_v2.txt
search_space:
  trainer.lr:    {{ type: loguniform, low: 1e-6, high: 1e-4 }}
  trainer.rank:  {{ type: categorical, choices: [4, 8, 16, 32] }}
  trainer.iters: {{ type: int, low: 200, high: 2000, step: 200 }}
objective: {{ metric: f1, direction: maximize }}
sampler: {sampler}
n_trials: {n_trials}
"""


def _write(tmp_path, body: str):
    p = tmp_path / "sweep.yaml"
    p.write_text(body)
    return p


def test_sweep_roundtrip(tmp_path):
    sw = Sweep.from_yaml(
        _write(tmp_path, SWEEP_YAML.format(sampler="tpe", n_trials=40))
    )
    sw.validate()
    assert sw.n_trials == 40
    assert sw.template["accelerator"] == "mlx"
    assert sw.to_manifest()["search_space"]["trainer.rank"]["choices"] == [4, 8, 16, 32]


def test_unknown_accelerator_rejected(tmp_path):
    body = SWEEP_YAML.format(sampler="tpe", n_trials=1).replace(
        "accelerator: mlx", "accelerator: quantum"
    )
    with pytest.raises(ValueError, match="accelerator"):
        Sweep.from_yaml(_write(tmp_path, body)).validate()


def test_n_trials_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="n_trials"):
        Sweep.from_yaml(
            _write(tmp_path, SWEEP_YAML.format(sampler="tpe", n_trials=0))
        ).validate()


def test_search_space_keys_must_target_trainer(tmp_path):
    body = SWEEP_YAML.format(sampler="tpe", n_trials=1).replace(
        "trainer.lr:", "top_level_lr:"
    )
    with pytest.raises(ValueError, match="trainer\\."):
        Sweep.from_yaml(_write(tmp_path, body)).validate()


def test_unknown_top_level_field_rejected(tmp_path):
    body = SWEEP_YAML.format(sampler="tpe", n_trials=1) + "extra_field: nope\n"
    with pytest.raises(ValueError, match="unknown sweep fields"):
        Sweep.from_yaml(_write(tmp_path, body))


def test_constraints_default_to_empty(tmp_path):
    sw = Sweep.from_yaml(_write(tmp_path, SWEEP_YAML.format(sampler="tpe", n_trials=1)))
    assert sw.constraints == []
    assert sw.to_manifest()["constraints"] == []


def test_constraint_roundtrip(tmp_path):
    body = SWEEP_YAML.format(sampler="tpe", n_trials=1) + (
        "constraints:\n  - {metric: cost, max: 5.0}\n"
    )
    sw = Sweep.from_yaml(_write(tmp_path, body))
    sw.validate()
    assert sw.constraints == [{"metric": "cost", "max": 5.0}]


def test_constraint_must_set_exactly_one_of_max_or_min(tmp_path):
    body = SWEEP_YAML.format(sampler="tpe", n_trials=1) + (
        "constraints:\n  - {metric: cost}\n"
    )
    with pytest.raises(ValueError, match="exactly one of 'max' or 'min'"):
        Sweep.from_yaml(_write(tmp_path, body)).validate()


def test_constraint_rejects_both_max_and_min(tmp_path):
    body = SWEEP_YAML.format(sampler="tpe", n_trials=1) + (
        "constraints:\n  - {metric: cost, max: 5.0, min: 1.0}\n"
    )
    with pytest.raises(ValueError, match="exactly one of 'max' or 'min'"):
        Sweep.from_yaml(_write(tmp_path, body)).validate()

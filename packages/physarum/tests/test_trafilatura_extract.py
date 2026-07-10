"""Contract tests for the 'cpu' runtime's trafilatura producer: it must
satisfy the same Producer shape as oyster's mlx_lora without a model, a
dataset URI, or a search backend in the loop."""

import json

import pytest

from reishi.primitives import dataset as dataset_registry
from reishi.primitives.dataset import Dataset

from physarum import mcm_plugin
from physarum.producers.trafilatura_extract import train

HTML_ROWS = [
    {
        "html": "<html><body><article><h1>Widget Report</h1>"
        "<p>Sales reached 1,250 units in 2024.</p></article></body></html>",
        "markdown": "# Widget Report\n\nSales reached 1,250 units in 2024.",
        "converter": "gemini",
        "html_sha1": "a" * 40,
    },
    {
        "html": "<html><body><article><h1>Second Page</h1>"
        "<p>Another paragraph of real content here.</p></article></body></html>",
        "markdown": "# Second Page\n\nAnother paragraph of real content here.",
        "converter": "gemini",
        "html_sha1": "b" * 40,
    },
]


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path))


def _register_dataset(tmp_path) -> Dataset:
    data_path = tmp_path / "rows.jsonl"
    with open(data_path, "w") as f:
        for row in HTML_ROWS:
            f.write(json.dumps(row) + "\n")
    ds = Dataset(
        name="htmlmd-fixture",
        uri=str(data_path),
        advisory_task="extract-fixture",
        eval_only=True,
    )
    dataset_registry.save(ds)
    return ds


def _manifest(dataset_name: str, hparams_cfg: dict) -> dict:
    return {
        "id": "t1",
        "recipe_name": "r1",
        "seed": 0,
        "status": "running",
        "created": "",
        "metrics": {},
        "artifacts": {},
        "spec": {
            "name": "r1",
            "task": "extract-fixture",
            "base_model": None,
            "eval_dataset": dataset_name,
            "runtime": "cpu",
            "prompt": None,
            "n_seeds": 1,
            "priority": 0,
            "hparams": hparams_cfg,
        },
        "execution": {},
    }


def test_train_returns_metrics_and_no_artifacts(tmp_path):
    ds = _register_dataset(tmp_path)
    result = train(_manifest(ds.name, {"include_tables": True}))
    assert result["artifacts"] == {}
    assert result["metrics"]["n"] == 2
    assert result["metrics"]["backend"] == "cpu"
    assert result["metrics"]["extractor"] == "trafilatura"
    assert result["metrics"]["params"] == {"include_tables": True}
    assert 0.0 <= result["metrics"]["field_f1"] <= 1.0


def test_n_eval_rows_bounds_rows_scored(tmp_path):
    ds = _register_dataset(tmp_path)
    result = train(_manifest(ds.name, {"n_eval_rows": 1}))
    assert result["metrics"]["n_rows"] == 1
    assert result["metrics"]["n"] == 1


def test_unknown_hparams_key_rejected(tmp_path):
    ds = _register_dataset(tmp_path)
    with pytest.raises(ValueError, match="unknown hparams keys"):
        train(_manifest(ds.name, {"not_a_real_param": True}))


def test_resolve_producer_wires_cpu_runtime():
    producer_fn = mcm_plugin._resolve_producer("cpu")
    assert producer_fn is train


def test_empty_eval_set_raises_clearly_instead_of_a_bare_keyerror(tmp_path):
    ds = _register_dataset(tmp_path)
    with pytest.raises(ValueError, match="eval set is empty"):
        train(_manifest(ds.name, {"n_eval_rows": 0}))

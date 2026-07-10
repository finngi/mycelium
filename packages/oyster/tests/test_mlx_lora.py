"""Contract test for oyster's mlx_lora trainer: run-resource facts (wall time,
iteration count) must land in the executor-written `observables` namespace, not
in `metrics` -- see math-foundations.md 3(iii). Model/tokenizer/lora_run/generate
are faked; only mlx-lm-independent code (mx.random.seed, make_sampler) runs for
real, so this needs mlx installed (darwin+arm64) but no model download.
"""

import json

import pytest

pytest.importorskip("mlx_lm", reason="mlx_lora trainer requires darwin+arm64 (mlx-lm)")

from reishi.primitives import dataset as dataset_registry  # noqa: E402
from reishi.primitives.dataset import Dataset  # noqa: E402

from oyster.trainers import mlx_lora  # noqa: E402

ROWS = [
    {"input": "a", "target": json.dumps({"x": "a"})},
    {"input": "b", "target": json.dumps({"x": "b"})},
]


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MCM_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("OYSTER_ARTIFACT_ROOT", str(tmp_path / "artifacts"))


def _register_dataset(tmp_path) -> Dataset:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for split in ("train", "val"):
        with open(data_dir / f"{split}.jsonl", "w") as f:
            for row in ROWS:
                f.write(json.dumps(row) + "\n")
    ds = Dataset(name="mlx-fixture", uri=str(data_dir), task="fixture")
    dataset_registry.save(ds)
    return ds


class _FakeTokenizer:
    has_chat_template = False


def _fake_load(model, adapter_path=None):
    return object(), _FakeTokenizer()


def _fake_lora_run(args):
    pass


def _fake_generate(model, tokenizer, prompt, max_tokens, sampler):
    # Echo the input back as the "prediction" -- ROWS is built so decoding this
    # equals gold, giving a deterministic, non-zero score.
    return json.dumps({"x": prompt})


def _manifest(dataset_name: str, seed: int, hparams_cfg: dict) -> dict:
    return {
        "id": "t1",
        "recipe_name": "r1",
        "seed": seed,
        "status": "running",
        "created": "",
        "metrics": {},
        "artifacts": {},
        "spec": {
            "name": "r1",
            "task": "fixture",
            "base_model": "x/tiny-model",
            "train_dataset": dataset_name,
            "runtime": "mlx",
            "prompt": None,
            "n_seeds": 1,
            "priority": 0,
            "hparams": hparams_cfg,
        },
        "execution": {},
    }


def test_train_writes_observables_not_metrics(tmp_path, monkeypatch):
    ds = _register_dataset(tmp_path)
    monkeypatch.setattr(mlx_lora, "load", _fake_load)
    monkeypatch.setattr(mlx_lora, "lora_run", _fake_lora_run)
    monkeypatch.setattr(mlx_lora, "generate", _fake_generate)

    result = mlx_lora.train(
        _manifest(ds.name, seed=7, hparams_cfg={"iters": 5, "n_eval_rows": 2})
    )

    assert result["observables"]["iters"] == 5
    assert isinstance(result["observables"]["wall_time_s"], float)

    assert "wall_s" not in result["metrics"]
    assert "iters" not in result["metrics"]
    assert "seed" not in result["metrics"]
    assert result["metrics"]["n"] == 2

"""LoRA fine-tune on Apple Silicon with mlx-lm, ported from mycelium's mlx_backend.

mycelium's checkpoint_sync (mid-run HF-checkpoint resume) doesn't port: oyster's
reaper already retries a whole trial on a dead runner (queue.requeue_stale), and
adding a second, finer-grained resume path on top would duplicate that recovery
mechanism for a benefit (resume mid-iters instead of from iters=0) that hasn't
been needed yet. _Heartbeat below replaces it -- same "poll during a blocking
call" shape, wired to the mesh's own liveness signal instead.
"""

import json
import os
import re
import sys
import threading
import time
import types
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate
from mlx_lm.lora import load
from mlx_lm.lora import run as lora_run
from mlx_lm.sample_utils import make_sampler

from reishi.primitives import codec as codec_registry
from reishi.primitives import dataset as dataset_registry
from reishi.primitives import task as task_registry
from reishi.primitives.trial import Trial, TrialManifest

from oyster import queue
from oyster.trainers.contract import TrainerResult

_HEARTBEAT_INTERVAL_S = 30.0


class _Heartbeat:
    """Keeps the trial's heartbeat fresh while lora_run() blocks this thread."""

    def __init__(self, trial_manifest: TrialManifest):
        self._trial = Trial.from_manifest(trial_manifest)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(_HEARTBEAT_INTERVAL_S):
            try:
                queue.heartbeat(self._trial)
            except Exception as e:
                print(f"[WARN] heartbeat failed: {e}", file=sys.stderr)

    def start(self):
        self._thread.start()

    def stop(self):
        # No timeout: an in-flight queue.heartbeat() write (gitstore.publish can be slow)
        # must fully land before the caller moves on to queue.finish() -- otherwise this
        # stale "running" snapshot can land after and clobber the done result.
        self._stop.set()
        self._thread.join()


def _prepare_data(train_path: Path, val_path: Path, out_dir: Path, codec, prompt: str | None, use_chat_template: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def convert(src: Path, dst: Path):
        with open(src) as f, open(dst, "w") as out:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                encoded_target = codec.encode(json.loads(obj["target"]))
                prompt_text = prompt.replace("{name}", obj["input"]) if prompt else obj["input"]
                if use_chat_template:
                    row = {"prompt": prompt_text, "completion": encoded_target}
                else:
                    row = {"text": prompt_text + encoded_target}
                out.write(json.dumps(row) + "\n")

    convert(train_path, out_dir / "train.jsonl")
    convert(val_path, out_dir / "valid.jsonl")


def _maybe_push_to_hf(adapter_dir: Path, trial_id: str) -> str | None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    from huggingface_hub import HfApi

    repo_id = f"{os.environ.get('OYSTER_HF_REPO_PREFIX', 'finngi/mcm-oyster')}-{trial_id}"
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id, private=True, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=str(adapter_dir))
        return f"hf://{repo_id}"
    except Exception as e:
        # publishing is a convenience, not the trial's success criterion -- a transient
        # upload failure shouldn't discard an otherwise-complete, expensive training run
        print(f"[WARN] HF publish failed: {e} -> falling back to local adapter path", file=sys.stderr)
        return None


def train(trial_manifest: TrialManifest) -> TrainerResult:
    spec = trial_manifest["spec"]
    base_model = spec.get("base_model")
    if base_model is None:
        raise ValueError("mlx_lora trainer needs base_model (from-scratch MLX training isn't implemented)")

    task_obj = task_registry.get(spec["task"])
    if task_obj.score is None:
        raise ValueError(f"task '{task_obj.name}' has no scorer registered; mlx_lora can't eval it")
    ds = dataset_registry.load(spec["dataset"])
    codec = codec_registry.get_codec(task_obj.codec)
    train_path, val_path = Path(ds.uri) / "train.jsonl", Path(ds.uri) / "val.jsonl"

    trainer_cfg = spec.get("trainer", {})
    method = trainer_cfg.get("method", "lora")
    iters = trainer_cfg.get("iters", 1000)
    seed = trial_manifest.get("seed", 0)
    prompt = spec.get("prompt")
    eval_n = trainer_cfg.get("eval_n", 50)

    out_dir = Path(os.environ.get("OYSTER_ARTIFACT_ROOT", "/tmp/oyster-artifacts")) / trial_manifest["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # Covers the whole run, not just lora_run(): cold model download, data prep,
    # eval, and HF upload can each individually exceed the reaper's stale-heartbeat
    # window, and a requeue mid-run means a second runner duplicates live work.
    heartbeat = _Heartbeat(trial_manifest)
    heartbeat.start()
    try:
        print(f"[INFO] loading {base_model} (cold start may take a while to download weights)", file=sys.stderr)
        # load()'s stub declares a Union[2-tuple, 3-tuple] keyed on return_config, which defaults
        # to False and is never passed True here -- always a 2-tuple at runtime (verified live).
        model, tokenizer = load(base_model, adapter_path=None)  # type: ignore[misc]

        use_chat_template = tokenizer.has_chat_template
        if not use_chat_template:
            print(f"[WARN] {base_model} has no chat_template -> training/eval will concatenate raw prompt+completion text", file=sys.stderr)

        mask_prompt = trainer_cfg.get("mask_prompt", False)
        if mask_prompt and not use_chat_template:
            # mlx-lm only supports prompt masking on chat/completion-format data, not the
            # raw-text rows this path produces when a model has no chat template
            print("[WARN] mask_prompt is only supported for chat/completion datasets -> ignoring for this text-format run", file=sys.stderr)
            mask_prompt = False

        data_dir = out_dir / "mlx_data"
        _prepare_data(train_path, val_path, data_dir, codec, prompt, use_chat_template)

        mx.random.seed(seed)  # lora_run() only seeds np.random (batch order), not mx.random (LoRA init/dropout)

        lora_params = None
        if method != "full":
            lora_params = {
                "rank": trainer_cfg.get("rank", 8),
                "dropout": trainer_cfg.get("dropout", 0.0),
                "scale": trainer_cfg.get("scale", 20.0),
            }

        args = types.SimpleNamespace(
            data=str(data_dir),
            train=True,
            model=base_model,
            iters=iters,
            batch_size=trainer_cfg.get("batch_size", 4),
            learning_rate=trainer_cfg.get("lr", 1e-5),
            max_seq_length=trainer_cfg.get("max_seq_length", 512),
            adapter_path=str(out_dir / "adapters"),
            steps_per_report=max(10, iters // 10),
            steps_per_eval=max(20, iters // 5),
            val_batches=trainer_cfg.get("val_batches", 25),
            mask_prompt=mask_prompt,
            grad_checkpoint=trainer_cfg.get("grad_checkpoint", False),
            grad_accumulation_steps=1,
            clear_cache_threshold=trainer_cfg.get("clear_cache_threshold", 100),
            num_layers=trainer_cfg.get("layers", 16),
            fine_tune_type=method,
            optimizer="adamw",
            optimizer_config={"adam": {}, "adamw": {}, "muon": {}, "sgd": {}, "adafactor": {}},
            save_every=trainer_cfg.get("save_every", max(iters // 6, 50) if iters >= 50 else max(iters // 2, 1)),
            seed=seed,
            test=False,
            report_to=None,
            project_name=None,
            resume_adapter_file=None,
            config=None,
            lr_schedule=None,
            test_batches=500,
            lora_parameters=lora_params,
        )

        print(f"[INFO] training: method={method} iters={iters} seed={seed}", file=sys.stderr)
        lora_run(args)

        print("[INFO] reloading model with trained adapter", file=sys.stderr)
        model, tokenizer = load(base_model, adapter_path=str(out_dir / "adapters"))  # type: ignore[misc]

        with open(val_path) as f:
            val_rows = [json.loads(line) for line in f if line.strip()][:eval_n]
        if not val_rows:
            raise ValueError("mlx_lora eval requires at least one validation row")

        sampler = make_sampler(temp=0.0)
        scores = []
        gen_start = time.time()
        for row in val_rows:
            prompt_text = prompt.replace("{name}", row["input"]) if prompt else row["input"]
            if use_chat_template:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}], add_generation_prompt=True, return_dict=False
                )
            else:
                formatted = prompt_text
            pred_text = generate(model, tokenizer, formatted, max_tokens=768, sampler=sampler)
            # reasoning models emit chain-of-thought before the answer; the codec must see only the structured output
            pred_text = re.sub(r"<think>.*?</think>", "", pred_text, flags=re.DOTALL).strip()
            gold = json.loads(row["target"])
            scores.append(task_obj.score(codec.decode(pred_text), gold))
        gen_seconds = time.time() - gen_start

        # A plain dict, not the stricter AggregateMetrics: run provenance (model, iters, seed...)
        # isn't part of the task's scoring contract, just this trainer's report alongside it.
        metrics: dict = {
            **task_registry.aggregate(scores),
            "model": base_model,
            "backend": "mlx",
            "method": method,
            "iters": iters,
            "seed": seed,
            "eval_names_per_s": round(len(scores) / gen_seconds, 2) if gen_seconds else None,
            "wall_s": round(time.time() - t0, 0),
        }

        adapter_dir = out_dir / "adapters"
        weights_uri = _maybe_push_to_hf(adapter_dir, trial_manifest["id"]) or str(adapter_dir)
    finally:
        heartbeat.stop()

    return {"metrics": metrics, "artifacts": {"weights": weights_uri}}

"""Producers, keyed by the recipe's runtime.

A producer is a callable: (trial_manifest: TrialManifest) -> ProducerResult
(see trainers/contract.py). Named 'trainers' not 'adapters' -- in this
domain an adapter is the LoRA artifact a trial produces.

Heavy ML deps (torch/transformers/peft/google-cloud-storage) are imported
lazily inside train_l4, not at module scope: the driver imports this module
unconditionally, including on a laptop with only the base dependency group,
so only calling a trainer should pull in the cluster group.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from urllib.parse import urlparse

from reishi import store
from reishi.primitives.trial import TrialManifest

from enoki.trainers.contract import Producer, ProducerResult

if TYPE_CHECKING:
    # Annotation-only: keeps torch out of the runtime import path (see module
    # docstring) while still typing the tensor fields below.
    import torch


class _ChatMessage(TypedDict):
    role: str
    content: str


class _TokenizedExample(TypedDict):
    input_ids: list[int]
    labels: list[int]


class _CollatedBatch(TypedDict):
    input_ids: "torch.Tensor"
    attention_mask: "torch.Tensor"
    labels: "torch.Tensor"


# ~1.5B-param model for HTML -> Markdown extraction, small enough to LoRA
# fine-tune on a single L4's 24GB.
BASE_MODEL = "jinaai/ReaderLM-v2"

# Smoke-scale defaults, all overridable per-recipe via `hparams:` (surfaces as
# trial_manifest["spec"]["hparams"]) or the matching ENOKI_L4_* env var.
# max_length 4096, not 2048: median htmlmd corpus length is ~2251 tokens, so
# 2048 would truncate over half the corpus. Truncation no longer corrupts
# supervision (see _build_example), but a larger default still drops less HTML.
DEFAULT_MAX_STEPS = 30
DEFAULT_SUBSET_SIZE = 64
DEFAULT_MAX_LENGTH = 4096
DEFAULT_RANK = 16
DEFAULT_LR = 2e-4
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 4

# Local staging copy (data/<task>/{train,valid}.jsonl) used as the fallback
# when a dataset's gs:// bucket isn't reachable (see _load_split).
_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


def _load_local_jsonl(task: str, split: str) -> list[dict]:
    path = _DATA_ROOT / task / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"no local dataset staged at {path} (and GCS was unreachable)"
        )
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _try_load_gcs_jsonl(uri: str, split: str) -> list[dict] | None:
    try:
        from google.cloud import storage
    except ImportError:
        return None
    try:
        client = storage.Client()
        parsed = urlparse(uri)
        blob = client.bucket(parsed.netloc).blob(
            f"{parsed.path.lstrip('/')}{split}.jsonl"
        )
        if not blob.exists(client):
            return None
        text = blob.download_as_text()
    except Exception:
        return None
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _load_split(dataset, split: str) -> list[dict]:
    """Try the dataset's gs:// uri first; fall back to the local staged copy
    when the bucket/blob isn't reachable (placeholder bucket, no creds, etc.)."""
    if dataset.uri.startswith("gs://"):
        rows = _try_load_gcs_jsonl(dataset.uri, split)
        if rows is not None:
            return rows
    return _load_local_jsonl(dataset.task or "htmlmd", split)


def _upload_dir_to_gcs(local_dir: Path, dataset_uri: str, trial_id: str) -> str | None:
    """Best-effort upload of the saved local checkpoint to the dataset's
    bucket, under artifacts/<trial_id>/. Returns the gs:// URI only on a
    completed upload; any failure returns None so the caller keeps the local
    path as the artifact of record."""
    if not dataset_uri.startswith("gs://"):
        return None
    try:
        from google.cloud import storage
    except ImportError:
        return None
    try:
        client = storage.Client()
        bucket_name = urlparse(dataset_uri).netloc
        bucket = client.bucket(bucket_name)
        if not bucket.exists(client):
            return None
        prefix = f"artifacts/{trial_id}/"
        for file in sorted(local_dir.iterdir()):
            if file.is_file():
                bucket.blob(prefix + file.name).upload_from_filename(str(file))
        return f"gs://{bucket_name}/{prefix}"
    except Exception:
        return None


def _build_example(
    tokenizer, messages: list[_ChatMessage], max_length: int
) -> _TokenizedExample | None:
    """Mask the loss to the assistant/completion tokens only. This is
    instruction-tuned chat data (user = HTML + instruction, assistant =
    markdown or INVALID_PAGE): training on the HTML-input tokens too would
    waste capacity learning to predict HTML back instead of the extraction.

    The chat template renders messages by strict concatenation, so the
    full-conversation text always starts with the prompt-only text -- token
    counts on each side of that boundary give the label mask directly,
    without hand-rolling ReaderLM-v2's ChatML template.

    Truncation is applied to the *prompt* (dropping its earliest tokens),
    never to the completion: tokenizing full_text directly with
    truncation=True right-truncates the concatenation, and since
    prompt-then-completion means the completion sits at the end, that ate
    into (or entirely removed) the assistant's completion instead of the
    HTML input -- silently masking every label to -100 or cutting the
    completion off before its EOS token. Returns None if the completion
    alone doesn't fit in max_length: such a row has nothing to supervise
    and must be dropped rather than handed to the Trainer with an
    all -100-label row (an all-masked batch's loss is NaN and poisons the
    whole accumulated-gradient step once it lands).
    """
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    completion_ids = full_ids[len(prompt_ids) :]
    if len(completion_ids) >= max_length:
        return None

    if len(full_ids) > max_length:
        keep_prompt = max_length - len(completion_ids)
        prompt_ids = prompt_ids[-keep_prompt:]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + list(completion_ids)
    return {"input_ids": input_ids, "labels": labels}


class _ChatDataset:
    def __init__(self, examples: list[_TokenizedExample]):
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> _TokenizedExample:
        return self._examples[idx]


def _collate(batch: list[_TokenizedExample], pad_token_id: int) -> _CollatedBatch:
    import torch

    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, attention_mask, labels = [], [], []
    for ex in batch:
        pad = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * pad)
        attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad)
        labels.append(ex["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def train_l4(trial_manifest: TrialManifest) -> ProducerResult:
    """Real, minimal LoRA fine-tune of ReaderLM-v2 -- plain transformers +
    peft, no TRL/Axolotl (simplest option, per instruction). Returns
    {"metrics": {...}, "artifacts": {"weights": <uri>}}, the same contract
    mcm-oyster's fake_trainer uses."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from transformers import Trainer as HFTrainer

    from reishi.primitives import dataset as dataset_registry

    spec = trial_manifest["spec"]
    hparams_cfg = spec.get("hparams") or {}
    base_model = spec.get("base_model") or BASE_MODEL

    def _cfg(key: str, env: str, default, cast):
        return (
            cast(hparams_cfg[key])
            if key in hparams_cfg
            else cast(os.environ.get(env, default))
        )

    max_steps = _cfg("iters", "ENOKI_L4_MAX_STEPS", DEFAULT_MAX_STEPS, int)
    subset_size = _cfg("subset_size", "ENOKI_L4_SUBSET_SIZE", DEFAULT_SUBSET_SIZE, int)
    max_length = _cfg("max_length", "ENOKI_L4_MAX_LENGTH", DEFAULT_MAX_LENGTH, int)
    rank = _cfg("rank", "ENOKI_L4_RANK", DEFAULT_RANK, int)
    lr = _cfg("lr", "ENOKI_L4_LR", DEFAULT_LR, float)
    batch_size = _cfg("batch_size", "ENOKI_L4_BATCH_SIZE", DEFAULT_BATCH_SIZE, int)
    grad_accum = _cfg("grad_accum", "ENOKI_L4_GRAD_ACCUM", DEFAULT_GRAD_ACCUM, int)

    train_dataset = spec.get("train_dataset")
    if train_dataset is None:
        raise ValueError("train_l4 needs a recipe with 'train_dataset' set")
    ds = dataset_registry.load(train_dataset)
    rows = _load_split(ds, "train")[:subset_size]
    if not rows:
        raise RuntimeError(f"dataset '{ds.name}' train split is empty")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    built = [_build_example(tokenizer, row["messages"], max_length) for row in rows]
    examples = [ex for ex in built if ex is not None]
    skipped = len(built) - len(examples)
    if skipped:
        print(
            f"[WARN] skipped {skipped}/{len(built)} example(s): assistant completion alone "
            f">= max_length={max_length}, nothing to supervise",
            file=sys.stderr,
        )
    if not examples:
        raise RuntimeError(
            f"all {len(built)} example(s) in this subset had a completion >= "
            f"max_length={max_length} -- increase trainer.max_length"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and os.environ.get("ENOKI_ALLOW_CPU_TRAINING") != "1":
        raise RuntimeError(
            "train_l4 requires a CUDA GPU (runtime='l4') but torch.cuda.is_available() "
            "is False -- refusing to silently fall back to CPU training (this usually means "
            "the job landed on a non-GPU node). Set ENOKI_ALLOW_CPU_TRAINING=1 to override "
            "for a deliberate CPU smoke test."
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    artifact_root = os.environ.get("ENOKI_ARTIFACT_ROOT") or str(store.artifact_root())
    output_dir = hparams_cfg.get("output_dir") or os.path.join(
        artifact_root, trial_manifest["id"]
    )
    os.makedirs(output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=max(1, max_steps // 10),
        save_strategy="no",
        report_to=[],
        bf16=device == "cuda",
        remove_unused_columns=False,
    )

    hf_trainer = HFTrainer(
        model=model,
        args=args,
        train_dataset=_ChatDataset(examples),
        data_collator=lambda batch: _collate(batch, tokenizer.pad_token_id),
    )
    result = hf_trainer.train()

    train_loss = float(result.training_loss)
    if math.isnan(train_loss) or math.isinf(train_loss):
        raise RuntimeError(
            f"training loss is {train_loss} after {max_steps} step(s) -- refusing to save/report "
            "a NaN/Inf adapter (check subset_size/max_length/batch composition upstream)"
        )

    # HF PEFT directory layout: adapter_config.json + adapter_model.safetensors.
    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)

    weights_uri = adapter_dir
    gcs_uri = _upload_dir_to_gcs(Path(adapter_dir), ds.uri, trial_manifest["id"])
    if gcs_uri is not None:
        weights_uri = gcs_uri

    metrics = {
        "train_loss": train_loss,
        "steps": max_steps,
        "examples": len(examples),
        "skipped_examples": skipped,
    }
    return {"metrics": metrics, "artifacts": {"weights": weights_uri}}


TRAINERS: dict[str, Producer] = {"l4": train_l4}

# GPUs each trainer's actual work needs, keyed the same as TRAINERS -- read
# by enoki.driver to size the ray.remote() call that routes training onto a
# GPU-holding worker rather than running in-process on the (CPU-only) head.
TRAINER_GPUS: dict[str, int] = {"l4": 1}


def get(runtime: str) -> Producer:
    if runtime not in TRAINERS:
        known = ", ".join(sorted(TRAINERS)) or "none yet"
        raise KeyError(f"no trainer for '{runtime}' (installed: {known})")
    return TRAINERS[runtime]

"""
Modal entrypoint for Whisper ASR LoRA fine-tuning using Unsloth.

Expected input data on `sna-data-vol`:
  - /data/{data_dir_name}/metadata_normalized.csv (default; normalized transcripts)
  - /data/{data_dir_name}/{file_name}.wav (as referenced by the CSV column `file_name`)

Preprocess on CPU first (cheaper than building this cache on a GPU worker):

  modal run -d preprocess_whisper_dataset.py --run-name <same-as-training>

CSV columns (minimum required):
  - file_name
  - transcription

Usage (Modal):
  modal run run_unsloth_whisper_finetune.py --run-name <your-run-name> --data-dir-name <wav_normalised>

Dataset cache helpers are defined in this file (not a separate module) so Modal’s single-file mount includes them.

Defaults favor throughput: train batch 2 × grad_accum 4 (effective batch 8), eval batch 4, eval/save every 1000
steps. If CUDA OOM, use e.g. --batch-size 1 --gradient-accumulation-steps 8 --eval-batch-size 2.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import modal


def run_dir_for(run_name: str, model_root: Path = Path("/model")) -> Path:
    return model_root / "sna-whisper-asr" / "unsloth" / run_name


def cache_paths(
    run_name: str,
    eval_max_samples: Optional[int],
    model_root: Path = Path("/model"),
) -> tuple[Path, Path, Path]:
    run_dir = run_dir_for(run_name, model_root)
    processed_root = run_dir / "processed"
    train_cache = processed_root / "train"
    eval_suffix = f"n{eval_max_samples}" if eval_max_samples is not None else "all"
    eval_cache = processed_root / f"eval_{eval_suffix}"
    return train_cache, eval_cache, processed_root


def _default_num_proc(requested: Optional[int]) -> int:
    if requested is not None:
        return max(1, requested)
    return 1


def _wav_file_is_readable(path: str) -> bool:
    try:
        import soundfile as sf

        sf.info(path)
        return True
    except Exception:
        return False


def _filter_row_readable(example: dict[str, Any]) -> bool:
    return _wav_file_is_readable(str(example["audio"]))


def _format_batch(examples: dict[str, list], processor: Any) -> dict[str, list]:
    audios = examples["audio"]
    texts = examples["text"]
    input_features_out: list[Any] = []
    for audio in audios:
        features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        )
        input_features_out.append(features.input_features[0])
    tok = processor.tokenizer(texts, padding=False, truncation=False)
    labels_out = tok.input_ids
    if hasattr(labels_out, "tolist"):
        labels_out = labels_out.tolist()
    if labels_out and isinstance(labels_out[0], int):
        labels_out = [labels_out]
    return {"input_features": input_features_out, "labels": labels_out}


def build_and_save_processed_datasets(
    *,
    metadata_path: Path,
    data_root: Path,
    run_name: str,
    file_extension: str = ".wav",
    text_column: str = "transcription",
    file_name_column: str = "file_name",
    eval_fraction: float = 0.06,
    eval_max_samples: Optional[int] = 300,
    seed: int = 3407,
    model_id: str = "unsloth/whisper-large-v3-turbo",
    model_root: Path = Path("/model"),
    num_proc: Optional[int] = None,
    map_batch_size: int = 64,
    writer_batch_size: int = 500,
    force_reprocess: bool = False,
) -> dict[str, Any]:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")

    from datasets import Audio, load_dataset
    from transformers import AutoProcessor

    train_cache, eval_cache, processed_root = cache_paths(run_name, eval_max_samples, model_root)

    if (
        not force_reprocess
        and train_cache.exists()
        and eval_cache.exists()
    ):
        return {
            "skipped": True,
            "train_cache": str(train_cache),
            "eval_cache": str(eval_cache),
        }

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")

    processor = AutoProcessor.from_pretrained(model_id)
    n_proc = _default_num_proc(num_proc)

    raw_ds = load_dataset("csv", data_files=str(metadata_path), split="train")
    if file_name_column not in raw_ds.column_names:
        raise KeyError(
            f"CSV column '{file_name_column}' not found. Columns: {raw_ds.column_names}"
        )
    if text_column not in raw_ds.column_names:
        raise KeyError(
            f"CSV column '{text_column}' not found. Columns: {raw_ds.column_names}"
        )

    def map_row(x: dict[str, Any]) -> dict[str, Any]:
        file_name = str(x[file_name_column])
        if file_name.endswith(file_extension):
            wav_name = file_name
        else:
            wav_name = file_name + file_extension
        wav_path = data_root / wav_name
        return {
            "audio": str(wav_path),
            "text": str(x[text_column]).strip(),
        }

    raw_ds = raw_ds.map(map_row, remove_columns=raw_ds.column_names)
    raw_ds = raw_ds.filter(
        lambda x: len(x["text"]) > 0 and os.path.exists(x["audio"])
    )
    n_before_decode = len(raw_ds)
    wf_proc = min(8, max(1, os.cpu_count() or 8))
    raw_ds = raw_ds.filter(_filter_row_readable, num_proc=wf_proc)
    dropped_unreadable = n_before_decode - len(raw_ds)
    if len(raw_ds) == 0:
        raise RuntimeError("No training samples found after filtering.")

    target_sr = int(processor.feature_extractor.sampling_rate)
    raw_ds = raw_ds.cast_column("audio", Audio(sampling_rate=target_sr))

    split_ds = raw_ds.train_test_split(test_size=eval_fraction, seed=seed)
    train_base = split_ds["train"]
    eval_base = split_ds["test"]
    if eval_max_samples is not None and len(eval_base) > eval_max_samples:
        eval_base = eval_base.shuffle(seed=seed).select(range(eval_max_samples))

    fmt = partial(_format_batch, processor=processor)
    map_kw: dict[str, Any] = {
        "batched": True,
        "batch_size": map_batch_size,
        "remove_columns": train_base.column_names,
        "num_proc": n_proc,
        "writer_batch_size": writer_batch_size,
    }
    train_dataset = train_base.map(fmt, **map_kw)
    map_kw["remove_columns"] = eval_base.column_names
    eval_dataset = eval_base.map(fmt, **map_kw)

    processed_root.mkdir(parents=True, exist_ok=True)
    train_dataset.save_to_disk(str(train_cache))
    eval_dataset.save_to_disk(str(eval_cache))

    return {
        "skipped": False,
        "train_cache": str(train_cache),
        "eval_cache": str(eval_cache),
        "num_train": len(train_dataset),
        "num_eval": len(eval_dataset),
        "num_proc": n_proc,
        "map_batch_size": map_batch_size,
        "dropped_unreadable_wav": dropped_unreadable,
        "wav_filter_num_proc": wf_proc,
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUN_NAME_DEFAULT = "unsloth-whisper-v3-turbo-v1"
DATA_VOLUME_NAME = "sna-data-vol"
MODEL_VOLUME_NAME = "sna-model-vol"
GPU_TYPE = "L40S"

DATA_DIR_NAME_DEFAULT = "wav_normalised"
METADATA_FILENAME_DEFAULT = "metadata_normalized.csv"


app = modal.App("sna-whisper-asr-unsloth")
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .uv_pip_install(
        "unsloth",
        "transformers==4.56.2",
        "datasets>=3.4.1,<4.0.0",
        "accelerate",
        "peft",
        "bitsandbytes",
        "xformers",
        "triton",
        "librosa",
        "soundfile",
        "evaluate",
        "jiwer",
        "wandb",
    )
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_volume, "/model": model_volume},
    secrets=[modal.Secret.from_dotenv()],
    timeout=60 * 60 * 6,
)
def run_train_unsloth_whisper(
    run_name: str = RUN_NAME_DEFAULT,
    force_reprocess: bool = False,
    data_dir_name: str = DATA_DIR_NAME_DEFAULT,
    metadata_filename: str = METADATA_FILENAME_DEFAULT,
    file_extension: str = ".wav",
    text_column: str = "transcription",
    file_name_column: str = "file_name",
    learning_rate: float = 1e-5,
    num_train_epochs: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_steps: int = 5000,
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    load_in_4bit: bool = False,
    warmup_steps: int = 300,
    weight_decay: float = 0.001,
    eval_fraction: float = 0.06,
    eval_max_samples: int | None = 300,
    eval_steps: int = 1000,
    logging_steps: int = 50,
    eval_accumulation_steps: int = 1,
    save_steps: int = 1000,
    eval_batch_size: int = 4,
    dataloader_num_workers: int = 2,
    save_total_limit: int = 3,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "wer",
    seed: int = 3407,
    max_text_tokens: int | None = 256,
    allow_gpu_preprocess: bool = False,
):
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    import unsloth  # noqa: F401 — before transformers

    from datasets import load_from_disk

    data_root = Path("/data") / data_dir_name
    metadata_path = data_root / metadata_filename
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")

    run_dir = Path("/model") / "sna-whisper-asr" / "unsloth" / run_name
    output_dir = run_dir / "outputs"
    lora_dir = run_dir / "lora"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_dir.mkdir(parents=True, exist_ok=True)

    train_cache, eval_cache, _ = cache_paths(run_name, eval_max_samples)
    have_cache = train_cache.exists() and eval_cache.exists()

    if have_cache and not force_reprocess:
        train_dataset = load_from_disk(str(train_cache))
        eval_dataset = load_from_disk(str(eval_cache))
    elif allow_gpu_preprocess:
        build_and_save_processed_datasets(
            metadata_path=metadata_path,
            data_root=data_root,
            run_name=run_name,
            file_extension=file_extension,
            text_column=text_column,
            file_name_column=file_name_column,
            eval_fraction=eval_fraction,
            eval_max_samples=eval_max_samples,
            seed=seed,
            force_reprocess=force_reprocess,
            num_proc=1,
            map_batch_size=64,
            writer_batch_size=500,
        )
        train_dataset = load_from_disk(str(train_cache))
        eval_dataset = load_from_disk(str(eval_cache))
    elif have_cache:
        raise RuntimeError(
            "force_reprocess=True but allow_gpu_preprocess=False. "
            "Run: modal run preprocess_whisper_dataset.py --run-name "
            f"{run_name!r} --force-reprocess"
        )
    else:
        raise RuntimeError(
            "Processed dataset cache not found. Build it on CPU first:\n"
            "  modal run -d preprocess_whisper_dataset.py "
            f"--run-name {run_name!r}\n"
            f"Expected under model volume: {train_cache} and {eval_cache}\n"
            "Or pass allow_gpu_preprocess=True to build on this GPU worker."
        )

    import numpy as np
    import torch
    import evaluate
    from transformers import WhisperForConditionalGeneration
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
    from unsloth import FastModel, is_bfloat16_supported

    # Enable train-time generation config: avoid forcing decoder language.
    # Unsloth/Whisper may set forced decoder ids depending on init params; we clear them.
    # (Code switching works better when we don't hard-bias to English only.)

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/whisper-large-v3-turbo",
        dtype=None,
        load_in_4bit=load_in_4bit,
        auto_model=WhisperForConditionalGeneration,
        whisper_language="English",
        whisper_task="transcribe",
    )

    # LoRA adapters: update a subset of projection matrices.
    model = FastModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        # Disable checkpointing for Whisper to avoid autograd double-backward errors
        # observed with both Unsloth + torch checkpointing paths.
        use_gradient_checkpointing=False,
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
        task_type=None,
    )

    # Clear forced-decoder constraints for more flexible language/code switching.
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
        model.generation_config.task = "transcribe"
        # Some setups also populate `language`. We unset it if present.
        if hasattr(model.generation_config, "language"):
            model.generation_config.language = None
    if hasattr(model, "config") and hasattr(model.config, "suppress_tokens"):
        model.config.suppress_tokens = []
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    metric = evaluate.load("wer")

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            # Remove initial BOS token if present (matches notebook behavior).
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    def compute_metrics(pred):
        preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        label_ids = pred.label_ids
        label_ids = label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.tokenizer.pad_token_id

        # If `predict_with_generate=True`, `preds` are token IDs.
        # Otherwise, they can be logits (float) and we take argmax.
        if hasattr(preds, "ndim") and preds.ndim >= 2 and np.issubdtype(preds.dtype, np.floating):
            pred_ids = np.argmax(preds, axis=-1)
        else:
            pred_ids = preds
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    if report_to == "wandb":
        try:
            import wandb
        except ModuleNotFoundError:
            print("[wandb] WANDB_API_KEY found but wandb is unavailable. Falling back to report_to='none'.")
            report_to = "none"

        if report_to == "wandb":
            wandb.init(
                project="sna-whisper-asr",
                name=run_name,
                config={
                    "data_dir_name": data_dir_name,
                    "learning_rate": learning_rate,
                    "num_train_epochs": num_train_epochs,
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "load_in_4bit": load_in_4bit,
                    "eval_fraction": eval_fraction,
                    "eval_max_samples": eval_max_samples,
                    "eval_batch_size": eval_batch_size,
                    "dataloader_num_workers": dataloader_num_workers,
                    "max_steps": max_steps,
                    "seed": seed,
                },
                dir=str(run_dir / "wandb"),
                reinit=True,
            )

    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=tokenizer),
        tokenizer=tokenizer.feature_extractor,
        compute_metrics=compute_metrics,
        args=Seq2SeqTrainingArguments(
            # Avoid OOM during eval by generating token IDs (small) instead of storing full logits (huge).
            predict_with_generate=True,
            generation_max_length=int(max_text_tokens) if max_text_tokens else 256,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=True,
            warmup_steps=warmup_steps,
            gradient_checkpointing=False,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            optim="adamw_8bit",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            weight_decay=weight_decay,
            remove_unused_columns=False,
            lr_scheduler_type="linear",
            label_names=["labels"],
            eval_steps=eval_steps,
            eval_strategy="steps",
            eval_accumulation_steps=eval_accumulation_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=False,
            seed=seed,
            output_dir=str(output_dir),
            report_to=report_to,
        ),
    )

    trainer_stats = trainer.train()
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    summary = {
        "run_name": run_name,
        "data_dir_name": data_dir_name,
        "num_train_samples": len(train_dataset),
        "num_eval_samples": len(eval_dataset),
        "max_steps": max_steps,
        "train_runtime": float(trainer_stats.metrics.get("train_runtime", -1.0)),
        "train_loss": float(trainer_stats.metrics.get("train_loss", -1.0))
        if "train_loss" in trainer_stats.metrics
        else None,
        "best_metric": None,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if report_to == "wandb":
        wandb.finish()

    # Persist outputs to the model volume.
    model_volume.commit()

    return {"lora_dir": str(lora_dir), "summary_path": str(run_dir / "summary.json")}


@app.local_entrypoint()
def main(
    run_name: str = RUN_NAME_DEFAULT,
    force_reprocess: bool = False,
    data_dir_name: str = DATA_DIR_NAME_DEFAULT,
    metadata_filename: str = METADATA_FILENAME_DEFAULT,
    file_extension: str = ".wav",
    text_column: str = "transcription",
    file_name_column: str = "file_name",
    learning_rate: float = 1e-5,
    num_train_epochs: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_steps: int = 5000,
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    load_in_4bit: bool = False,
    warmup_steps: int = 300,
    weight_decay: float = 0.001,
    eval_fraction: float = 0.06,
    eval_max_samples: int | None = 300,
    eval_steps: int = 1000,
    logging_steps: int = 50,
    eval_accumulation_steps: int = 1,
    save_steps: int = 1000,
    eval_batch_size: int = 4,
    dataloader_num_workers: int = 2,
    save_total_limit: int = 3,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "wer",
    seed: int = 3407,
    max_text_tokens: int | None = 256,
    allow_gpu_preprocess: bool = False,
):
    run_train_unsloth_whisper.remote(
        run_name=run_name,
        force_reprocess=force_reprocess,
        data_dir_name=data_dir_name,
        metadata_filename=metadata_filename,
        file_extension=file_extension,
        text_column=text_column,
        file_name_column=file_name_column,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        load_in_4bit=load_in_4bit,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        eval_fraction=eval_fraction,
        eval_max_samples=eval_max_samples,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        eval_accumulation_steps=eval_accumulation_steps,
        save_steps=save_steps,
        eval_batch_size=eval_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        seed=seed,
        max_text_tokens=max_text_tokens,
        allow_gpu_preprocess=allow_gpu_preprocess,
    )


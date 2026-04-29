"""
Modal entrypoint for fine-tuning `facebook/w2v-bert-2.0` on Shona ASR data.

This script ports the Hugging Face Mongolian W2V2-BERT notebook into the
workspace conventions used in this repository:

- Data volume:  `sna-data-vol` mounted at `/data`
- Model volume: `sna-model-vol` mounted at `/model`
- Default data root: `/data/wav_normalised`
- Default metadata CSV: `metadata_normalized.csv`
- Required CSV columns: `file_name`, `transcription`

Outputs are written under:
  /model/sna-w2v2-bert-asr/hf/<run_name>/

Example usage:
  modal run run_w2v2_bert_finetune.py --preprocess-only --run-name sna-w2v2-v1
  modal run run_w2v2_bert_finetune.py --run-name sna-w2v2-v1
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import modal


RUN_NAME_DEFAULT = "sna-w2v2-bert-v1"
DATA_VOLUME_NAME = "sna-data-vol"
MODEL_VOLUME_NAME = "sna-model-vol"
GPU_TYPE = "L40S"
BASE_MODEL_ID = "facebook/w2v-bert-2.0"

DATA_DIR_NAME_DEFAULT = "wav_normalised"
METADATA_FILENAME_DEFAULT = "metadata_normalized.csv"
FILE_EXTENSION_DEFAULT = ".wav"
TEXT_COLUMN_DEFAULT = "transcription"
FILE_NAME_COLUMN_DEFAULT = "file_name"

WORD_DELIMITER_TOKEN = "|"
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"


def run_dir_for(run_name: str, model_root: Path = Path("/model")) -> Path:
    return model_root / "sna-w2v2-bert-asr" / "hf" / run_name


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


def processor_dir_for(run_name: str, model_root: Path = Path("/model")) -> Path:
    return run_dir_for(run_name, model_root) / "processor"


def vocab_path_for(run_name: str, model_root: Path = Path("/model")) -> Path:
    return run_dir_for(run_name, model_root) / "vocab.json"


def preprocess_summary_path_for(
    run_name: str, model_root: Path = Path("/model")
) -> Path:
    return run_dir_for(run_name, model_root) / "preprocess_summary.json"


def summary_path_for(run_name: str, model_root: Path = Path("/model")) -> Path:
    return run_dir_for(run_name, model_root) / "summary.json"


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


def _prepare_dataset_row(
    batch: dict[str, Any],
    *,
    processor: Any,
) -> dict[str, Any]:
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["input_length"] = len(batch["input_features"])
    batch["labels"] = processor(text=batch["text"]).input_ids
    return batch


def _extract_vocab_from_dataset(dataset: Any) -> dict[str, int]:
    joined_text = " ".join(str(text) for text in dataset["text"] if str(text))
    vocab_list = sorted(set(joined_text))
    vocab_dict = {char: idx for idx, char in enumerate(vocab_list)}

    if " " in vocab_dict:
        vocab_dict[WORD_DELIMITER_TOKEN] = vocab_dict[" "]
        del vocab_dict[" "]
    elif WORD_DELIMITER_TOKEN not in vocab_dict:
        vocab_dict[WORD_DELIMITER_TOKEN] = len(vocab_dict)

    if UNK_TOKEN not in vocab_dict:
        vocab_dict[UNK_TOKEN] = len(vocab_dict)
    if PAD_TOKEN not in vocab_dict:
        vocab_dict[PAD_TOKEN] = len(vocab_dict)
    return vocab_dict


def build_and_save_w2v2_bert_assets(
    *,
    metadata_path: Path,
    data_root: Path,
    run_name: str,
    file_extension: str = FILE_EXTENSION_DEFAULT,
    text_column: str = TEXT_COLUMN_DEFAULT,
    file_name_column: str = FILE_NAME_COLUMN_DEFAULT,
    eval_fraction: float = 0.06,
    eval_max_samples: Optional[int] = 300,
    seed: int = 3407,
    model_root: Path = Path("/model"),
    num_proc: Optional[int] = None,
    writer_batch_size: int = 200,
    force_reprocess: bool = False,
) -> dict[str, Any]:
    for env_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(env_name, "1")

    if not 0.0 < float(eval_fraction) < 1.0:
        raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}")

    from datasets import Audio, load_dataset
    from transformers import (
        SeamlessM4TFeatureExtractor,
        Wav2Vec2BertProcessor,
        Wav2Vec2CTCTokenizer,
    )

    run_dir = run_dir_for(run_name, model_root)
    train_cache, eval_cache, processed_root = cache_paths(
        run_name, eval_max_samples, model_root
    )
    processor_dir = processor_dir_for(run_name, model_root)
    vocab_path = vocab_path_for(run_name, model_root)
    preprocess_summary_path = preprocess_summary_path_for(run_name, model_root)

    if (
        not force_reprocess
        and train_cache.exists()
        and eval_cache.exists()
        and processor_dir.exists()
        and vocab_path.exists()
        and preprocess_summary_path.exists()
    ):
        existing_summary = json.loads(preprocess_summary_path.read_text(encoding="utf-8"))
        existing_summary["skipped"] = True
        return existing_summary

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")

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
        file_name = str(x[file_name_column]).strip()
        text = str(x[text_column]).strip()
        if file_name.endswith(file_extension):
            audio_name = file_name
        else:
            audio_name = file_name + file_extension
        return {
            "audio": str(data_root / audio_name),
            "text": text,
            "source_file_name": file_name,
        }

    mapped_ds = raw_ds.map(map_row, remove_columns=raw_ds.column_names)
    raw_rows = len(mapped_ds)

    empty_text_rows = len(mapped_ds.filter(lambda x: len(x["text"]) == 0))
    non_empty_ds = mapped_ds.filter(lambda x: len(x["text"]) > 0)

    missing_audio_rows = len(non_empty_ds.filter(lambda x: not os.path.exists(x["audio"])))
    existing_audio_ds = non_empty_ds.filter(lambda x: os.path.exists(x["audio"]))

    n_before_decode = len(existing_audio_ds)
    readable_num_proc = min(8, max(1, os.cpu_count() or 8))
    filtered_ds = existing_audio_ds.filter(_filter_row_readable, num_proc=readable_num_proc)
    unreadable_audio_rows = n_before_decode - len(filtered_ds)

    if len(filtered_ds) == 0:
        raise RuntimeError("No training samples found after filtering invalid rows.")
    if len(filtered_ds) < 2:
        raise RuntimeError(
            "Need at least 2 valid samples after filtering to create train/eval splits."
        )

    vocab_dict = _extract_vocab_from_dataset(filtered_ds)
    run_dir.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(
        json.dumps(vocab_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_path),
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        word_delimiter_token=WORD_DELIMITER_TOKEN,
    )
    feature_extractor = SeamlessM4TFeatureExtractor(
        feature_size=80,
        num_mel_bins=80,
        sampling_rate=16000,
        padding_value=0.0,
    )
    processor = Wav2Vec2BertProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    processor.save_pretrained(str(processor_dir))

    filtered_ds = filtered_ds.cast_column(
        "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
    )
    split_ds = filtered_ds.train_test_split(test_size=eval_fraction, seed=seed)
    train_base = split_ds["train"]
    eval_base = split_ds["test"]
    if eval_max_samples is not None and len(eval_base) > eval_max_samples:
        eval_base = eval_base.shuffle(seed=seed).select(range(eval_max_samples))

    if len(train_base) == 0 or len(eval_base) == 0:
        raise RuntimeError(
            "Train/eval split produced an empty dataset. Adjust eval_fraction or data."
        )

    prepare_fn = partial(_prepare_dataset_row, processor=processor)
    map_kwargs: dict[str, Any] = {
        "num_proc": _default_num_proc(num_proc),
        "writer_batch_size": writer_batch_size,
        "remove_columns": train_base.column_names,
    }
    train_dataset = train_base.map(prepare_fn, **map_kwargs)
    map_kwargs["remove_columns"] = eval_base.column_names
    eval_dataset = eval_base.map(prepare_fn, **map_kwargs)

    train_dataset.save_to_disk(str(train_cache))
    eval_dataset.save_to_disk(str(eval_cache))

    summary = {
        "skipped": False,
        "run_name": run_name,
        "metadata_path": str(metadata_path),
        "data_root": str(data_root),
        "train_cache": str(train_cache),
        "eval_cache": str(eval_cache),
        "processor_dir": str(processor_dir),
        "vocab_path": str(vocab_path),
        "raw_rows": raw_rows,
        "empty_text_rows": empty_text_rows,
        "missing_audio_rows": missing_audio_rows,
        "unreadable_audio_rows": unreadable_audio_rows,
        "valid_rows": len(filtered_ds),
        "num_train": len(train_dataset),
        "num_eval": len(eval_dataset),
        "eval_fraction": eval_fraction,
        "eval_max_samples": eval_max_samples,
        "seed": seed,
        "file_extension": file_extension,
        "text_column": text_column,
        "file_name_column": file_name_column,
        "vocab_size": len(vocab_dict),
        "num_proc": _default_num_proc(num_proc),
        "readable_filter_num_proc": readable_num_proc,
    }
    preprocess_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: Union[bool, str] = True

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], Any]]],
    ) -> Dict[str, Any]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )
        batch["labels"] = labels
        return batch


app = modal.App("sna-w2v2-bert-asr")
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .uv_pip_install(
        "torch",
        "transformers==4.56.2",
        "datasets>=3.4.1,<4.0.0",
        "accelerate",
        "evaluate",
        "jiwer",
        "librosa",
        "soundfile",
        "wandb",
    )
)


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    volumes={"/data": data_volume, "/model": model_volume},
    timeout=60 * 60 * 4,
)
def prepare_w2v2_bert_assets(
    run_name: str = RUN_NAME_DEFAULT,
    force_reprocess: bool = False,
    data_dir_name: str = DATA_DIR_NAME_DEFAULT,
    metadata_filename: str = METADATA_FILENAME_DEFAULT,
    file_extension: str = FILE_EXTENSION_DEFAULT,
    text_column: str = TEXT_COLUMN_DEFAULT,
    file_name_column: str = FILE_NAME_COLUMN_DEFAULT,
    eval_fraction: float = 0.06,
    eval_max_samples: Optional[int] = 300,
    seed: int = 3407,
    num_proc: Optional[int] = None,
) -> dict[str, Any]:
    data_root = Path("/data") / data_dir_name
    metadata_path = data_root / metadata_filename
    summary = build_and_save_w2v2_bert_assets(
        metadata_path=metadata_path,
        data_root=data_root,
        run_name=run_name,
        file_extension=file_extension,
        text_column=text_column,
        file_name_column=file_name_column,
        eval_fraction=eval_fraction,
        eval_max_samples=eval_max_samples,
        seed=seed,
        num_proc=num_proc,
        force_reprocess=force_reprocess,
    )
    model_volume.commit()
    return summary


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_volume, "/model": model_volume},
    secrets=[modal.Secret.from_dotenv()],
    timeout=60 * 60 * 10,
)
def run_train_w2v2_bert(
    run_name: str = RUN_NAME_DEFAULT,
    force_reprocess: bool = False,
    data_dir_name: str = DATA_DIR_NAME_DEFAULT,
    metadata_filename: str = METADATA_FILENAME_DEFAULT,
    file_extension: str = FILE_EXTENSION_DEFAULT,
    text_column: str = TEXT_COLUMN_DEFAULT,
    file_name_column: str = FILE_NAME_COLUMN_DEFAULT,
    eval_fraction: float = 0.06,
    eval_max_samples: Optional[int] = 300,
    seed: int = 3407,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    eval_batch_size: int = 4,
    max_steps: int = 5000,
    warmup_ratio: float = 0.1,
    eval_steps: int = 1000,
    save_steps: int = 1000,
    logging_steps: int = 50,
    save_total_limit: int = 3,
    gradient_checkpointing: bool = True,
    dataloader_num_workers: int = 2,
    eval_accumulation_steps: int = 1,
    allow_gpu_preprocess: bool = False,
    num_train_epochs: float = 10.0,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "wer",
):
    import numpy as np
    import evaluate
    from datasets import load_from_disk
    from transformers import (
        Trainer,
        TrainingArguments,
        Wav2Vec2BertForCTC,
        Wav2Vec2BertProcessor,
    )

    if load_best_model_at_end and save_steps % eval_steps != 0:
        raise ValueError(
            "save_steps must be a multiple of eval_steps when load_best_model_at_end=True."
        )

    data_root = Path("/data") / data_dir_name
    metadata_path = data_root / metadata_filename
    run_dir = run_dir_for(run_name)
    output_dir = run_dir / "outputs"
    final_dir = run_dir / "final"
    processor_dir = processor_dir_for(run_name)
    train_cache, eval_cache, _ = cache_paths(run_name, eval_max_samples)
    preprocess_summary_path = preprocess_summary_path_for(run_name)

    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    have_assets = (
        train_cache.exists()
        and eval_cache.exists()
        and processor_dir.exists()
        and preprocess_summary_path.exists()
    )

    if force_reprocess and not allow_gpu_preprocess:
        raise RuntimeError(
            "force_reprocess=True requires allow_gpu_preprocess=True on the training worker, "
            "or run the preprocess-only mode first."
        )

    if force_reprocess and allow_gpu_preprocess:
        build_and_save_w2v2_bert_assets(
            metadata_path=metadata_path,
            data_root=data_root,
            run_name=run_name,
            file_extension=file_extension,
            text_column=text_column,
            file_name_column=file_name_column,
            eval_fraction=eval_fraction,
            eval_max_samples=eval_max_samples,
            seed=seed,
            num_proc=1,
            force_reprocess=True,
        )
    elif not have_assets and allow_gpu_preprocess:
        build_and_save_w2v2_bert_assets(
            metadata_path=metadata_path,
            data_root=data_root,
            run_name=run_name,
            file_extension=file_extension,
            text_column=text_column,
            file_name_column=file_name_column,
            eval_fraction=eval_fraction,
            eval_max_samples=eval_max_samples,
            seed=seed,
            num_proc=1,
            force_reprocess=force_reprocess,
        )
    elif not have_assets:
        raise RuntimeError(
            "Processed assets not found. Build them first with:\n"
            "  modal run run_w2v2_bert_finetune.py "
            f"--run-name {run_name!r} --preprocess-only\n"
            "Or pass allow_gpu_preprocess=True."
        )

    train_dataset = load_from_disk(str(train_cache))
    eval_dataset = load_from_disk(str(eval_cache))
    processor = Wav2Vec2BertProcessor.from_pretrained(str(processor_dir))

    metric = evaluate.load("wer")

    def compute_metrics(pred: Any) -> dict[str, float]:
        predictions = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        pred_ids = np.argmax(predictions, axis=-1)

        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        wer = metric.compute(predictions=pred_str, references=label_str)
        return {"wer": float(wer)}

    model = Wav2Vec2BertForCTC.from_pretrained(
        BASE_MODEL_ID,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    wandb_run = None
    if report_to == "wandb":
        try:
            import wandb
        except ModuleNotFoundError:
            print(
                "[wandb] WANDB_API_KEY found but wandb is unavailable. "
                "Falling back to report_to='none'."
            )
            report_to = "none"
        if report_to == "wandb":
            preprocess_summary = json.loads(
                preprocess_summary_path.read_text(encoding="utf-8")
            )
            wandb_run = wandb.init(
                project="sna-w2v2-bert-asr",
                name=run_name,
                dir=str(run_dir / "wandb"),
                reinit=True,
                config={
                    "run_name": run_name,
                    "base_model_id": BASE_MODEL_ID,
                    "data_dir_name": data_dir_name,
                    "metadata_filename": metadata_filename,
                    "eval_fraction": eval_fraction,
                    "eval_max_samples": eval_max_samples,
                    "vocab_size": preprocess_summary["vocab_size"],
                    "empty_text_rows": preprocess_summary["empty_text_rows"],
                    "missing_audio_rows": preprocess_summary["missing_audio_rows"],
                    "unreadable_audio_rows": preprocess_summary["unreadable_audio_rows"],
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "eval_batch_size": eval_batch_size,
                    "max_steps": max_steps,
                    "warmup_ratio": warmup_ratio,
                    "gradient_checkpointing": gradient_checkpointing,
                    "save_steps": save_steps,
                    "eval_steps": eval_steps,
                    "logging_steps": logging_steps,
                    "seed": seed,
                },
            )

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
        args=TrainingArguments(
            output_dir=str(output_dir),
            group_by_length=True,
            length_column_name="input_length",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_strategy="steps",
            save_strategy="steps",
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            gradient_checkpointing=gradient_checkpointing,
            fp16=True,
            bf16=False,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            save_total_limit=save_total_limit,
            remove_unused_columns=False,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=True,
            eval_accumulation_steps=eval_accumulation_steps,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=False,
            seed=seed,
            report_to=report_to,
            label_names=["labels"],
        ),
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )

    trainer_stats = trainer.train()
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    preprocess_summary = json.loads(preprocess_summary_path.read_text(encoding="utf-8"))
    summary = {
        "run_name": run_name,
        "base_model_id": BASE_MODEL_ID,
        "data_dir_name": data_dir_name,
        "metadata_filename": metadata_filename,
        "train_cache": str(train_cache),
        "eval_cache": str(eval_cache),
        "processor_dir": str(processor_dir),
        "output_dir": str(output_dir),
        "final_dir": str(final_dir),
        "summary_path": str(summary_path_for(run_name)),
        "num_train_samples": len(train_dataset),
        "num_eval_samples": len(eval_dataset),
        "vocab_size": preprocess_summary["vocab_size"],
        "empty_text_rows": preprocess_summary["empty_text_rows"],
        "missing_audio_rows": preprocess_summary["missing_audio_rows"],
        "unreadable_audio_rows": preprocess_summary["unreadable_audio_rows"],
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "eval_batch_size": eval_batch_size,
        "max_steps": max_steps,
        "warmup_ratio": warmup_ratio,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "save_total_limit": save_total_limit,
        "gradient_checkpointing": gradient_checkpointing,
        "seed": seed,
        "train_runtime": float(trainer_stats.metrics.get("train_runtime", -1.0)),
        "train_loss": float(trainer_stats.metrics.get("train_loss", -1.0))
        if "train_loss" in trainer_stats.metrics
        else None,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "report_to": report_to,
    }
    summary_path_for(run_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if wandb_run is not None:
        wandb_run.finish()

    model_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    run_name: str = RUN_NAME_DEFAULT,
    force_reprocess: bool = False,
    preprocess_only: bool = False,
    allow_gpu_preprocess: bool = False,
    data_dir_name: str = DATA_DIR_NAME_DEFAULT,
    metadata_filename: str = METADATA_FILENAME_DEFAULT,
    file_extension: str = FILE_EXTENSION_DEFAULT,
    text_column: str = TEXT_COLUMN_DEFAULT,
    file_name_column: str = FILE_NAME_COLUMN_DEFAULT,
    eval_fraction: float = 0.06,
    eval_max_samples: Optional[int] = 300,
    seed: int = 3407,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    eval_batch_size: int = 4,
    max_steps: int = 5000,
    warmup_ratio: float = 0.1,
    eval_steps: int = 1000,
    save_steps: int = 1000,
    logging_steps: int = 50,
    save_total_limit: int = 3,
    gradient_checkpointing: bool = True,
    dataloader_num_workers: int = 2,
    eval_accumulation_steps: int = 1,
):
    if preprocess_only:
        payload = prepare_w2v2_bert_assets.remote(
            run_name=run_name,
            force_reprocess=force_reprocess,
            data_dir_name=data_dir_name,
            metadata_filename=metadata_filename,
            file_extension=file_extension,
            text_column=text_column,
            file_name_column=file_name_column,
            eval_fraction=eval_fraction,
            eval_max_samples=eval_max_samples,
            seed=seed,
        )
    else:
        payload = run_train_w2v2_bert.remote(
            run_name=run_name,
            force_reprocess=force_reprocess,
            data_dir_name=data_dir_name,
            metadata_filename=metadata_filename,
            file_extension=file_extension,
            text_column=text_column,
            file_name_column=file_name_column,
            eval_fraction=eval_fraction,
            eval_max_samples=eval_max_samples,
            seed=seed,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_batch_size=eval_batch_size,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            eval_steps=eval_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            save_total_limit=save_total_limit,
            gradient_checkpointing=gradient_checkpointing,
            dataloader_num_workers=dataloader_num_workers,
            eval_accumulation_steps=eval_accumulation_steps,
            allow_gpu_preprocess=allow_gpu_preprocess,
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))

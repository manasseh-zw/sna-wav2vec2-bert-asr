"""
Modal script to run inference with a fine-tuned Whisper LoRA adapter.

Usage examples:
  modal run run_unsloth_whisper_inference.py --run-name unsloth-whisper-v3-turbo-v1 --audio-path wav_normalised/sample.wav
  modal run run_unsloth_whisper_inference.py --run-name unsloth-whisper-v3-turbo-v1 --audio-dir wav_normalised --limit 3
  modal run run_unsloth_whisper_inference.py --run-name unsloth-whisper-v3-turbo-v1 --my-samples --sample-files "wav_normalised/sample_1.wav,wav_normalised/sample_2.wav"

  Load a specific training checkpoint (PEFT folder) instead of lora/:
  modal run run_unsloth_whisper_inference.py --run-name sna-whisper-norm-v1 \\
    --adapter-subpath outputs/checkpoint-5000 --audio-dir wav_normalised --limit 10 \\
    --metadata-filename metadata_normalized.csv --shuffle-samples
"""

from __future__ import annotations

import json
import csv
import time
from pathlib import Path

import modal


DATA_VOLUME_NAME = "sna-data-vol"
MODEL_VOLUME_NAME = "sna-model-vol"
GPU_TYPE = "L40S"
BASE_MODEL_ID = "unsloth/whisper-large-v3-turbo"

app = modal.App("sna-whisper-asr-inference")
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .uv_pip_install(
        "transformers==4.56.2",
        "peft",
        "torch",
        "accelerate",
    )
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_volume, "/model": model_volume},
    timeout=60 * 30,
)
def run_whisper_inference(
    run_name: str,
    audio_path: str = "",
    audio_dir: str = "wav_normalised",
    limit: int = 3,
    language: str = "",
    task: str = "transcribe",
    return_timestamps: bool = False,
    metadata_filename: str = "metadata.csv",
    transcript_column: str = "transcription",
    file_name_column: str = "file_name",
    my_samples: bool = False,
    sample_files: str = "",
    adapter_subpath: str = "lora",
    shuffle_samples: bool = False,
    shuffle_seed: int = 42,
):
    import random
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    run_root = Path("/model") / "sna-whisper-asr" / "unsloth" / run_name
    adapter_dir = run_root / adapter_subpath
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter dir not found: {adapter_dir} (set adapter_subpath, e.g. lora or outputs/checkpoint-5000)"
        )

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=0 if torch.cuda.is_available() else -1,
    )

    if my_samples and sample_files.strip():
        files = [Path("/data") / s.strip() for s in sample_files.split(",") if s.strip()]
    elif audio_path:
        files = [Path("/data") / audio_path]
    else:
        base = Path("/data") / audio_dir
        if not base.exists():
            raise FileNotFoundError(f"Audio directory not found: {base}")
        all_wav = list(base.glob("*.wav"))
        if shuffle_samples:
            rng = random.Random(shuffle_seed)
            rng.shuffle(all_wav)
        else:
            all_wav.sort()
        files = all_wav[: max(1, limit)]

    # Build filename->reference transcript mapping from metadata.csv when available.
    if audio_path:
        metadata_dir = (Path("/data") / audio_path).parent
    else:
        metadata_dir = Path("/data") / audio_dir
    metadata_path = metadata_dir / metadata_filename

    transcript_by_file = {}
    if (not my_samples) and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = str(row.get(file_name_column, "")).strip()
                if not key:
                    continue
                transcript_by_file[key] = str(row.get(transcript_column, "")).strip()

    results = []
    for p in files:
        if not p.exists():
            continue
        generate_kwargs = {"task": task}
        if language:
            generate_kwargs["language"] = language

        t0 = time.perf_counter()
        out = asr(
            str(p),
            return_timestamps=return_timestamps,
            generate_kwargs=generate_kwargs,
        )
        dt = time.perf_counter() - t0

        # Match on either "name.wav" or "name".
        base_name = p.name
        stem = p.stem
        reference_text = transcript_by_file.get(base_name, transcript_by_file.get(stem, ""))
        results.append(
            {
                "file": str(p),
                "file_name": p.name,
                "reference_text": reference_text,
                "text": out.get("text", ""),
                "chunks": out.get("chunks", []),
                "transcription_time_seconds": round(dt, 4),
            }
        )

    out_dir = Path("/model") / "sna-whisper-asr" / "unsloth" / run_name / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "last_inference.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    model_volume.commit()
    return {"results": results, "saved_to": str(out_path)}


@app.local_entrypoint()
def main(
    run_name: str,
    audio_path: str = "",
    audio_dir: str = "wav_normalised",
    limit: int = 3,
    language: str = "",
    task: str = "transcribe",
    return_timestamps: bool = False,
    metadata_filename: str = "metadata.csv",
    transcript_column: str = "transcription",
    file_name_column: str = "file_name",
    my_samples: bool = False,
    sample_files: str = "",
    adapter_subpath: str = "lora",
    shuffle_samples: bool = False,
    shuffle_seed: int = 42,
):
    payload = run_whisper_inference.remote(
        run_name=run_name,
        audio_path=audio_path,
        audio_dir=audio_dir,
        limit=limit,
        language=language,
        task=task,
        return_timestamps=return_timestamps,
        metadata_filename=metadata_filename,
        transcript_column=transcript_column,
        file_name_column=file_name_column,
        my_samples=my_samples,
        sample_files=sample_files,
        adapter_subpath=adapter_subpath,
        shuffle_samples=shuffle_samples,
        shuffle_seed=shuffle_seed,
    )

    print(f"Saved: {payload['saved_to']}")
    for item in payload["results"]:
        print("-" * 80)
        print(item["file"])
        print(f"reference: {item['reference_text']}")
        print(item["text"])
        print(f"time_s: {item['transcription_time_seconds']}")

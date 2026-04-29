"""
Modal script to run inference with a fine-tuned W2V2-BERT ASR model.

The script can load either:
- a local exported artifact from the model volume, or
- a published Hugging Face model repo

Examples:
  uv run modal run -d run_w2v2_bert_inference.py --run-name sna-w2v2-v1 --audio-dir wav_normalised --limit 3

  uv run modal run -d run_w2v2_bert_inference.py --run-name sna-w2v2-v1 \
    --my-samples --sample-root model --sample-files "samples/mixed_1.wav,samples/mixed_2.wav"

  uv run modal run -d run_w2v2_bert_inference.py --hf-repo-id manassehzw/sna-w2v2-bert-shona \
    --my-samples --sample-root model --sample-files "samples/demo.wav"
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import modal


DATA_VOLUME_NAME = "sna-data-vol"
MODEL_VOLUME_NAME = "sna-model-vol"
GPU_TYPE = "L40S"
RUN_NAME_DEFAULT = "sna-w2v2-v1"
BASE_MODEL_ID = "facebook/w2v-bert-2.0"
MODEL_SUBDIR_DEFAULT = "final"


app = modal.App("sna-w2v2-bert-asr-inference")
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .uv_pip_install(
        "transformers==4.56.2",
        "torch",
        "accelerate",
        "librosa",
        "soundfile",
    )
)


def run_dir_for(run_name: str, model_root: Path = Path("/model")) -> Path:
    return model_root / "sna-w2v2-bert-asr" / "hf" / run_name


def _root_path(root_name: str) -> Path:
    root = root_name.strip().lower()
    if root == "data":
        return Path("/data")
    if root == "model":
        return Path("/model")
    raise ValueError(f"Unsupported root {root_name!r}. Use 'data' or 'model'.")


def _resolve_input_path(root_name: str, value: str) -> Path:
    candidate = Path(value.strip())
    if str(candidate).startswith("/data/") or str(candidate).startswith("/model/"):
        return candidate
    return _root_path(root_name) / candidate


def _reference_lookup(
    *,
    metadata_root: str,
    audio_path: str,
    audio_dir: str,
    metadata_filename: str,
    transcript_column: str,
    file_name_column: str,
    my_samples: bool,
) -> dict[str, str]:
    if my_samples:
        return {}

    if audio_path:
        metadata_dir = _resolve_input_path(metadata_root, audio_path).parent
    else:
        metadata_dir = _resolve_input_path(metadata_root, audio_dir)
    metadata_path = metadata_dir / metadata_filename

    transcript_by_file: dict[str, str] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = str(row.get(file_name_column, "")).strip()
                if not key:
                    continue
                transcript_by_file[key] = str(row.get(transcript_column, "")).strip()
    return transcript_by_file


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_volume, "/model": model_volume},
    timeout=60 * 30,
)
def run_w2v2_bert_inference(
    run_name: str = RUN_NAME_DEFAULT,
    hf_repo_id: str = "",
    model_subdir: str = MODEL_SUBDIR_DEFAULT,
    audio_path: str = "",
    audio_dir: str = "wav_normalised",
    audio_root: str = "data",
    limit: int = 3,
    metadata_filename: str = "metadata_normalized.csv",
    metadata_root: str = "data",
    transcript_column: str = "transcription",
    file_name_column: str = "file_name",
    my_samples: bool = False,
    sample_files: str = "",
    sample_root: str = "data",
    shuffle_samples: bool = False,
    shuffle_seed: int = 42,
):
    import random
    import torch
    from transformers import AutoModelForCTC, AutoProcessor, pipeline

    hf_repo_id = hf_repo_id.strip()
    if hf_repo_id:
        if "/" not in hf_repo_id:
            raise ValueError(
                "hf_repo_id must be a fully qualified Hugging Face repo id like "
                "'owner/repo'. Omit --hf-repo-id to use the local model volume artifact."
            )
        model_ref = hf_repo_id
        save_run_name = run_name or "hf-repo-inference"
    else:
        run_root = run_dir_for(run_name)
        model_dir = run_root / model_subdir
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model dir not found: {model_dir} "
                "(set run_name/model_subdir or use --hf-repo-id)"
            )
        model_ref = str(model_dir)
        save_run_name = run_name

    processor = AutoProcessor.from_pretrained(model_ref)
    model = AutoModelForCTC.from_pretrained(model_ref)
    model.eval()

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=0 if torch.cuda.is_available() else -1,
    )

    if my_samples and sample_files.strip():
        files = [
            _resolve_input_path(sample_root, s.strip())
            for s in sample_files.split(",")
            if s.strip()
        ]
    elif audio_path:
        files = [_resolve_input_path(audio_root, audio_path)]
    else:
        base = _resolve_input_path(audio_root, audio_dir)
        if not base.exists():
            raise FileNotFoundError(f"Audio directory not found: {base}")
        all_wav = list(base.glob("*.wav"))
        if shuffle_samples:
            rng = random.Random(shuffle_seed)
            rng.shuffle(all_wav)
        else:
            all_wav.sort()
        files = all_wav[: max(1, limit)]

    transcript_by_file = _reference_lookup(
        metadata_root=metadata_root,
        audio_path=audio_path,
        audio_dir=audio_dir,
        metadata_filename=metadata_filename,
        transcript_column=transcript_column,
        file_name_column=file_name_column,
        my_samples=my_samples,
    )

    results = []
    for p in files:
        if not p.exists():
            results.append(
                {
                    "file": str(p),
                    "file_name": p.name,
                    "reference_text": "",
                    "text": "",
                    "transcription_time_seconds": None,
                    "error": "file_not_found",
                }
            )
            continue

        t0 = time.perf_counter()
        out = asr(str(p))
        dt = time.perf_counter() - t0

        base_name = p.name
        stem = p.stem
        reference_text = transcript_by_file.get(base_name, transcript_by_file.get(stem, ""))
        results.append(
            {
                "file": str(p),
                "file_name": p.name,
                "reference_text": reference_text,
                "text": out.get("text", ""),
                "transcription_time_seconds": round(dt, 4),
            }
        )

    out_dir = run_dir_for(save_run_name) / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "last_inference.json"
    out_path.write_text(
        json.dumps(
            {
                "model_ref": model_ref,
                "run_name": save_run_name,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    model_volume.commit()
    return {"results": results, "saved_to": str(out_path), "model_ref": model_ref}


@app.local_entrypoint()
def main(
    run_name: str = RUN_NAME_DEFAULT,
    hf_repo_id: str = "",
    model_subdir: str = MODEL_SUBDIR_DEFAULT,
    audio_path: str = "",
    audio_dir: str = "wav_normalised",
    audio_root: str = "data",
    limit: int = 3,
    metadata_filename: str = "metadata_normalized.csv",
    metadata_root: str = "data",
    transcript_column: str = "transcription",
    file_name_column: str = "file_name",
    my_samples: bool = False,
    sample_files: str = "",
    sample_root: str = "data",
    shuffle_samples: bool = False,
    shuffle_seed: int = 42,
):
    if not hf_repo_id and not run_name:
        raise ValueError("run_name is required when not using --hf-repo-id.")

    payload = run_w2v2_bert_inference.remote(
        run_name=run_name,
        hf_repo_id=hf_repo_id,
        model_subdir=model_subdir,
        audio_path=audio_path,
        audio_dir=audio_dir,
        audio_root=audio_root,
        limit=limit,
        metadata_filename=metadata_filename,
        metadata_root=metadata_root,
        transcript_column=transcript_column,
        file_name_column=file_name_column,
        my_samples=my_samples,
        sample_files=sample_files,
        sample_root=sample_root,
        shuffle_samples=shuffle_samples,
        shuffle_seed=shuffle_seed,
    )

    print(f"Model: {payload['model_ref']}")
    print(f"Saved: {payload['saved_to']}")
    for item in payload["results"]:
        print("-" * 80)
        print(item["file"])
        print(f"reference: {item['reference_text']}")
        print(item["text"])
        print(f"time_s: {item['transcription_time_seconds']}")
        if item.get("error"):
            print(f"error: {item['error']}")

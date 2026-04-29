"""
Modal entrypoint to publish a trained W2V2-BERT ASR model from the model volume
to the Hugging Face Hub.

The script uploads the saved model artifacts from:
  /model/sna-w2v2-bert-asr/hf/<run_name>/<model_subdir>

Defaults target the Shona ASR fine-tune created by this repository.

Example:
  uv run modal run -d upload_w2v2_bert_to_hf.py \
    --run-name sna-w2v2-v1 \
    --hf-username manassehzw \
    --repo-name sna-w2v2-bert-shona
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


MODEL_VOLUME_NAME = "sna-model-vol"
RUN_NAME_DEFAULT = "sna-w2v2-v1"
MODEL_SUBDIR_DEFAULT = "final"
DATASET_REPO_ID_DEFAULT = "manassehzw/sna-dataset-annotated"
BASE_MODEL_ID_DEFAULT = "facebook/w2v-bert-2.0"
HF_TOKEN_ENV_VAR_DEFAULT = "HF_TOKEN"
DEFAULT_LICENSE = "mit"
DEFAULT_LANGUAGE = "sna"
DEFAULT_PRETTY_NAME = "Shona W2V2-BERT ASR"
DEFAULT_AUTHOR = "Manasseh Changachirere"
DEFAULT_AUTHOR_AFFILIATION = "Harare Institute of Technology"
DEFAULT_AUTHOR_URL = "https://www.manasseh.dev/"


def run_dir_for(run_name: str, model_root: Path = Path("/model")) -> Path:
    return model_root / "sna-w2v2-bert-asr" / "hf" / run_name


def _coalesce_repo_id(hf_username: str, repo_name: str) -> str:
    repo_name = repo_name.strip()
    if "/" in repo_name:
        return repo_name
    if not hf_username.strip():
        raise ValueError("hf_username is required when repo_name does not include an owner.")
    return f"{hf_username.strip()}/{repo_name}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _format_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _repo_url(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id}"


def _build_model_card(
    *,
    repo_id: str,
    pretty_name: str,
    base_model_id: str,
    dataset_repo_id: str,
    language: str,
    license_id: str,
    author_name: str,
    author_affiliation: str,
    author_url: str,
    run_name: str,
    model_subdir: str,
    summary: dict[str, Any],
    preprocess_summary: dict[str, Any],
) -> str:
    best_metric = summary.get("best_metric")
    best_metric_pct = _format_percent(best_metric)
    best_checkpoint = summary.get("best_model_checkpoint") or "N/A"
    train_runtime = summary.get("train_runtime")
    train_runtime_h = (
        round(float(train_runtime) / 3600.0, 3)
        if isinstance(train_runtime, (int, float))
        else "N/A"
    )

    train_loss = summary.get("train_loss")
    train_loss_text = f"{float(train_loss):.5f}" if train_loss is not None else "N/A"

    num_train = preprocess_summary.get("num_train", summary.get("num_train_samples", "N/A"))
    num_eval = preprocess_summary.get("num_eval", summary.get("num_eval_samples", "N/A"))
    vocab_size = preprocess_summary.get("vocab_size", "N/A")
    valid_rows = preprocess_summary.get("valid_rows", "N/A")

    yaml_block = f"""---
language:
- {language}
license: {license_id}
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- automatic-speech-recognition
- audio
- speech
- shona
- wav2vec2-bert
- ctc
base_model:
- {base_model_id}
datasets:
- {dataset_repo_id}
metrics:
- wer
model-index:
- name: {pretty_name}
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: {dataset_repo_id}
      type: {dataset_repo_id}
    metrics:
    - name: Word Error Rate
      type: wer
      value: {best_metric if best_metric is not None else '"N/A"'}
---
"""

    body = f"""
# {repo_id}

{pretty_name} is a Shona (`{language}`) automatic speech recognition model fine-tuned from
[`{base_model_id}`](https://huggingface.co/{base_model_id}) using the annotated dataset
[`{dataset_repo_id}`](https://huggingface.co/datasets/{dataset_repo_id}).

## Model Details

- **Curated by:** [{author_name} ({author_affiliation})]({author_url})
- **Base model:** [`{base_model_id}`](https://huggingface.co/{base_model_id})
- **Training dataset:** [`{dataset_repo_id}`](https://huggingface.co/datasets/{dataset_repo_id})
- **Run name:** `{run_name}`
- **Published artifact subdirectory:** `{model_subdir}`
- **Best eval WER:** {best_metric_pct}
- **Best checkpoint:** `{best_checkpoint}`
- **Train runtime:** {train_runtime_h} hours

## Training Summary

- **Training examples:** {num_train}
- **Evaluation examples:** {num_eval}
- **Valid rows after filtering:** {valid_rows}
- **Vocabulary size:** {vocab_size}
- **Final logged train loss:** {train_loss_text}
- **Gradient checkpointing:** {summary.get("gradient_checkpointing", "N/A")}
- **Learning rate:** {summary.get("learning_rate", "N/A")}
- **Batch size:** {summary.get("batch_size", "N/A")}
- **Gradient accumulation steps:** {summary.get("gradient_accumulation_steps", "N/A")}
- **Max steps:** {summary.get("max_steps", "N/A")}
- **Eval cadence:** every {summary.get("eval_steps", "N/A")} steps
- **Save cadence:** every {summary.get("save_steps", "N/A")} steps

## Intended Use

This model is intended for Shona ASR research and speech pipeline development,
including transcription experiments and downstream speech-to-speech systems.

## Limitations

- Evaluation was performed on a held-out subset derived from the training corpus
  split used in this project, not on an external benchmark.
- Real-world performance may differ on domain-shifted audio, unseen accents,
  noisy recordings, or code-switched speech.
- The model card metric is the best evaluation WER recorded during fine-tuning.

## References

- Base model: Meta AI, *W2v-BERT 2.0 speech encoder*,
  [`{base_model_id}`](https://huggingface.co/{base_model_id})
- Dataset used for fine-tuning:
  [`{dataset_repo_id}`](https://huggingface.co/datasets/{dataset_repo_id})

## Files

This repository contains:

- the exported fine-tuned model weights and config
- tokenizer and processor files
- `training/summary.json`
- `training/preprocess_summary.json`

## Example Usage

```python
from transformers import AutoModelForCTC, AutoProcessor
import torch

repo_id = "{repo_id}"
processor = AutoProcessor.from_pretrained(repo_id)
model = AutoModelForCTC.from_pretrained(repo_id)

# audio_array: 1D float waveform sampled at 16 kHz
inputs = processor(audio_array, sampling_rate=16_000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
pred_ids = torch.argmax(logits, dim=-1)
transcript = processor.batch_decode(pred_ids)[0]
print(transcript)
```
"""
    return yaml_block + body


app = modal.App("sna-w2v2-bert-upload")
model_vol = modal.Volume.from_name(MODEL_VOLUME_NAME)

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "huggingface_hub",
)


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=60 * 60,
    volumes={"/model": model_vol},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def upload_w2v2_bert_model(
    run_name: str = RUN_NAME_DEFAULT,
    hf_username: str = "",
    repo_name: str = "sna-w2v2-bert-shona",
    private_repo: bool = False,
    hf_token_env_var: str = HF_TOKEN_ENV_VAR_DEFAULT,
    model_subdir: str = MODEL_SUBDIR_DEFAULT,
    dataset_repo_id: str = DATASET_REPO_ID_DEFAULT,
    base_model_id: str = BASE_MODEL_ID_DEFAULT,
    pretty_name: str = DEFAULT_PRETTY_NAME,
    language: str = DEFAULT_LANGUAGE,
    license_id: str = DEFAULT_LICENSE,
    author_name: str = DEFAULT_AUTHOR,
    author_affiliation: str = DEFAULT_AUTHOR_AFFILIATION,
    author_url: str = DEFAULT_AUTHOR_URL,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    hf_token = os.environ.get(hf_token_env_var, "").strip()
    if not hf_token:
        raise ValueError(f"Missing {hf_token_env_var} in environment.")

    repo_id = _coalesce_repo_id(hf_username=hf_username, repo_name=repo_name)

    run_dir = run_dir_for(run_name)
    model_dir = run_dir / model_subdir
    summary_path = run_dir / "summary.json"
    preprocess_summary_path = run_dir / "preprocess_summary.json"
    upload_audit_path = run_dir / "upload_audit.json"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model artifact directory not found: {model_dir}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing training summary: {summary_path}")
    if not preprocess_summary_path.exists():
        raise FileNotFoundError(f"Missing preprocess summary: {preprocess_summary_path}")

    summary = _read_json(summary_path)
    preprocess_summary = _read_json(preprocess_summary_path)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private_repo, exist_ok=True)

    print("=" * 72)
    print("SNA ASR PIPELINE - UPLOAD W2V2-BERT MODEL")
    print("=" * 72)
    print(f"Run name:       {run_name}")
    print(f"Repo:           {repo_id}")
    print(f"Artifact dir:   {model_dir}")
    print(f"Base model:     {base_model_id}")
    print(f"Dataset:        {dataset_repo_id}")

    print("Uploading model artifacts...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload fine-tuned model artifacts for {run_name}",
    )

    print("Uploading training metadata...")
    api.upload_file(
        path_or_fileobj=str(summary_path),
        path_in_repo="training/summary.json",
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(preprocess_summary_path),
        path_in_repo="training/preprocess_summary.json",
        repo_id=repo_id,
        repo_type="model",
    )

    readme = _build_model_card(
        repo_id=repo_id,
        pretty_name=pretty_name,
        base_model_id=base_model_id,
        dataset_repo_id=dataset_repo_id,
        language=language,
        license_id=license_id,
        author_name=author_name,
        author_affiliation=author_affiliation,
        author_url=author_url,
        run_name=run_name,
        model_subdir=model_subdir,
        summary=summary,
        preprocess_summary=preprocess_summary,
    )

    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=io.BytesIO(readme.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    upload_audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "repo_id": repo_id,
        "repo_url": _repo_url(repo_id),
        "model_dir": str(model_dir),
        "summary_path": str(summary_path),
        "preprocess_summary_path": str(preprocess_summary_path),
        "dataset_repo_id": dataset_repo_id,
        "base_model_id": base_model_id,
        "best_metric": summary.get("best_metric"),
        "best_model_checkpoint": summary.get("best_model_checkpoint"),
        "private_repo": private_repo,
        "model_subdir": model_subdir,
        "readme_uploaded": True,
        "training_metadata_uploaded": True,
    }
    upload_audit_path.write_text(
        json.dumps(upload_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    model_vol.commit()

    print("=" * 72)
    print("UPLOAD COMPLETE")
    print(f"Model repo: {_repo_url(repo_id)}")
    print(f"Audit:      {upload_audit_path}")
    print("=" * 72)

    return upload_audit


@app.local_entrypoint()
def main(
    run_name: str = RUN_NAME_DEFAULT,
    hf_username: str = "",
    repo_name: str = "sna-w2v2-bert-shona",
    private_repo: bool = False,
    hf_token_env_var: str = HF_TOKEN_ENV_VAR_DEFAULT,
    model_subdir: str = MODEL_SUBDIR_DEFAULT,
    dataset_repo_id: str = DATASET_REPO_ID_DEFAULT,
    base_model_id: str = BASE_MODEL_ID_DEFAULT,
    pretty_name: str = DEFAULT_PRETTY_NAME,
    language: str = DEFAULT_LANGUAGE,
    license_id: str = DEFAULT_LICENSE,
    author_name: str = DEFAULT_AUTHOR,
    author_affiliation: str = DEFAULT_AUTHOR_AFFILIATION,
    author_url: str = DEFAULT_AUTHOR_URL,
):
    payload = upload_w2v2_bert_model.remote(
        run_name=run_name,
        hf_username=hf_username,
        repo_name=repo_name,
        private_repo=private_repo,
        hf_token_env_var=hf_token_env_var,
        model_subdir=model_subdir,
        dataset_repo_id=dataset_repo_id,
        base_model_id=base_model_id,
        pretty_name=pretty_name,
        language=language,
        license_id=license_id,
        author_name=author_name,
        author_affiliation=author_affiliation,
        author_url=author_url,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))

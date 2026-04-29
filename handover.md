# Handover Notes (New Workspace)

This handover is for reusing the same data/volume conventions in a completely new workspace and fine-tuning a different base ASR model.

This file is intentionally standalone and includes full topology + operational context.
Carry only these code references alongside it:

- `run_unsloth_whisper_finetune.py` (training reference implementation)
- `run_unsloth_whisper_inference.py` (inference reference)

---

## 1) Infrastructure and Path Conventions

### Modal volumes

- Data volume: `sna-data-vol`
- Model/output volume: `sna-model-vol`

### Mount points in container

- Data mounted to: `/data`
- Model/output mounted to: `/model`

### Dataset layout

- Directory: `/data/wav_normalised/`
- CSV files:
  - `/data/wav_normalised/metadata.csv` (original text)
- Audio files: `/data/wav_normalised/*.wav`

### CSV column mapping used

- `file_name` -> audio filename/path key
- `transcription` -> label text

Other columns can exist and are ignored by training logic.

---

## 2) Output Convention

Use a predictable run directory structure:

- `/model/sna-whisper-asr/<family_or_framework>/<run_name>/`
  - `outputs/` (HF checkpoints)
  - `lora/` (adapter/model artifacts)
  - `processed/train`, `processed/eval` (cached preprocessed datasets)
  - `summary.json`
  - `inference/last_inference.json` (if inference script writes results)

In previous runs with Unsloth Whisper:

- `/model/sna-whisper-asr/unsloth/<run_name>/...`

---

## 3) What Worked Well

- Base used previously: `unsloth/whisper-large-v3-turbo` with LoRA.
- First stable run (2500 steps) reached ~54.31 raw eval WER.
- Qualitative code-switching transcripts were understandable and useful.
- Caching processed datasets saved significant rerun time.

---

## 4) Bugs/Failures Encountered and Fixes

### A) Missing dependency (`wandb`)

- Symptom: runtime crash when WANDB key exists but `wandb` not installed.
- Fix: include `wandb` in image OR fallback to `report_to="none"` if import fails.

### B) Gradient checkpointing instability

- Symptom: `RuntimeError: Trying to backward through the graph a second time`.
- Triggered with Unsloth/torch checkpointing paths for this Whisper setup.
- Fix used: disable problematic checkpointing mode for stability.

### C) Eval OOM (major)

- Symptom: OOM during evaluation at checkpoint/eval step.
- Cause: Trainer accumulating full logits across eval set and concatenating.
- Fix:
  - `predict_with_generate=True`
  - `eval_accumulation_steps=1`
  - keep eval batch small

### D) Modal CLI boolean flags

- Symptom: `Got unexpected extra argument (True)`
- Cause: passing `--flag True` style.
- Fix: use click-style booleans:
  - `--my-samples` (enable)
  - `--no-my-samples` (disable)

### E) Inference m4a decode

- m4a may fail in pipeline path decode depending on environment/container codecs.
- Simple reliable path: convert m4a -> wav locally with ffmpeg, upload wav.

---

## 5) Whisper Run Configuration Used Historically (Reference Only)

The following were used in the prior Whisper run and are included strictly as historical context:

- `learning_rate`: `1e-5`
- `lora_r`: `32`
- `lora_alpha`: `32`
- `batch_size`: `1`
- `gradient_accumulation_steps`: `8`
- `warmup_steps`: `300`
- `max_steps`: `2500` to `5000` (budget dependent)
- `save_strategy`: steps
- `save_steps`: `250` (or `500` for less overhead)
- `save_total_limit`: `3`
- `load_best_model_at_end`: true
- `metric_for_best_model`: `wer`
- `greater_is_better`: false
- `predict_with_generate`: true
- `eval_accumulation_steps`: `1`

---

## 6) Runtime and Cost Observations (Modal)

- In prior runs, frequent eval materially increased wall-clock time and credits.
- Observed run behavior: eval every 250 steps with ~915 eval samples added roughly ~9 minutes per eval pass.

Pricing reference used during project:

- L40S around $1.95/hour (check live pricing before run).

---

## 7) Inference Conventions

For ad-hoc sample checks (without CSV references):

- use `--my-samples`
- pass explicit files with `--sample-files "path1,path2"`

For dataset-backed checks:

- use `audio_dir`
- optional reference lookup from metadata by `file_name`.

---
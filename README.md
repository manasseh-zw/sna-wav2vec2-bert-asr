# Shona ASR: WAV2VEC2-BERT Fine-tuning Pipeline

A production-ready automatic speech recognition (ASR) pipeline for the Shona language, featuring WAV2VEC2-BERT model fine-tuning with comprehensive evaluation and Hugging Face Hub integration.

## 📊 Overview

This repository contains:

1. **WAV2VEC2-BERT Fine-tuning Pipeline** — State-of-the-art ASR model fine-tuned on Shona speech data
2. **Whisper Fine-tuning Baseline** — Comparison implementation for benchmarking
3. **Inference Scripts** — Ready-to-use inference for both models
4. **HuggingFace Integration** — Automated model publishing to the Hub

## 🏆 Model Selection: Why WAV2VEC2-BERT?

We evaluated two approaches for Shona ASR:

### WAV2VEC2-BERT (Selected) ✅

- **Base Model**: `facebook/w2v-bert-2.0` (1.5B parameters, multilingual pretraining)
- **Key Advantages**:
  - Superior Word Error Rate (WER) on Shona evaluation set
  - More efficient inference compared to Whisper
  - Better performance on accented and conversational speech
  - Lighter weight deployment footprint

### Whisper (Baseline for Comparison)

- **Base Model**: OpenAI Whisper (multilingual, speech translation)
- **Trade-offs**:
  - Higher latency inference due to larger model size
  - Comparable but slightly higher WER on Shona corpus
  - Better for multilingual scenarios (we focused on Shona-only)
  - Supports speech translation as additional capability

**Result**: WAV2VEC2-BERT provided the best balance of accuracy, efficiency, and production viability for Shona-only ASR.

## 📁 Repository Structure

```
.
├── README.md                              # This file
├── pyproject.toml                         # Project dependencies (uv)
├── main.py                                # Entry point / utilities
│
├── run_w2v2_bert_finetune.py             # WAV2VEC2-BERT fine-tuning (Modal)
├── run_w2v2_bert_inference.py            # WAV2VEC2-BERT inference
│
├── run_unsloth_whisper_finetune.py       # Whisper fine-tuning with LoRA (Modal)
├── run_unsloth_whisper_inference.py      # Whisper inference
│
└── upload_w2v2_bert_to_hf.py             # Upload trained model to HuggingFace Hub
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for package management
- [Modal](https://modal.com/) account (for distributed training)
- HuggingFace credentials (for model publishing)

### Installation

```bash
# Install dependencies
uv pip install -r requirements.txt
```

## 📈 Training Pipeline

### Step 1: Prepare Data

Organize your Shona speech data with a metadata CSV:

```csv
file_name,transcription
sample_001.wav,Mwarambe kusvika mangwanani
sample_002.wav,Ndinokuda kuenda kumba
...
```

### Step 2: Fine-tune WAV2VEC2-BERT

```bash
# Preprocess data first (CPU, cheaper)
uv run modal run -d run_w2v2_bert_finetune.py \
  --preprocess-only \
  --run-name sna-w2v2-v1

# Then train on GPU
uv run modal run run_w2v2_bert_finetune.py \
  --run-name sna-w2v2-v1
```

**Training Configuration**:

- Base Model: `facebook/w2v-bert-2.0`
- Optimization: Mixed precision, gradient accumulation, gradient checkpointing
- Evaluation Metric: Word Error Rate (WER)
- Hardware: L40S GPU (Modal)

### Step 3: Evaluate & Compare

Run inference on your test set:

```bash
uv run python run_w2v2_bert_inference.py \
  --model-id manassehzw/sna-w2v-bert-2.0-asr \
  --audio-dir ./test_audio
```

### Step 4: Upload to HuggingFace Hub

```bash
uv run modal run upload_w2v2_bert_to_hf.py \
  --run-name sna-w2v2-v1 \
  --hf-username manassehzw \
  --repo-name sna-w2v2-bert-shona
```

This generates a comprehensive model card including:

- Training metadata and hyperparameters
- Evaluation results (WER, checkpoints)
- Example usage code
- Dataset and base model references

## 🎯 Model Specifications

### WAV2VEC2-BERT (Selected)

- **Base Model**: facebook/w2v-bert-2.0
- **Input**: 16 kHz mono audio (`.wav`)
- **Output**: UTF-8 transcriptions (Shona)
- **Inference Time**: ~0.1x real-time (CPU) / ~50x real-time (GPU)
- **Model Size**: ~1.5GB

**Example Usage**:

```python
from transformers import AutoModelForCTC, AutoProcessor
import torch

processor = AutoProcessor.from_pretrained("manassehzw/sna-w2v-bert-2.0-asr")
model = AutoModelForCTC.from_pretrained("manassehzw/sna-w2v-bert-2.0-asr")

# Load and process audio
inputs = processor(audio_array, sampling_rate=16_000, return_tensors="pt")

# Transcribe
with torch.no_grad():
    logits = model(**inputs).logits

pred_ids = torch.argmax(logits, dim=-1)
transcript = processor.batch_decode(pred_ids)[0]
print(transcript)
```

### Whisper (For Reference)

- **Base Model**: openai/whisper-base (or larger)
- **Advantages**: Multilingual, speech translation support
- **Trade-offs**: Slower inference, marginally higher WER on Shona
- **Use Case**: When multilingual capability or translation is needed

## 📊 Training & Evaluation

Training outputs are structured as:

```
/model/sna-w2v2-bert-asr/hf/{run_name}/
├── final/                          # Best checkpoint artifacts
│   ├── config.json
│   ├── model.safetensors
│   ├── processor_config.json
│   └── ...
├── training/
│   ├── summary.json                # Best metric, runtime, hyperparams
│   └── preprocess_summary.json     # Dataset statistics
└── upload_audit.json               # Upload metadata & timestamps
```

**Key Metrics**:

- `best_metric`: Word Error Rate (WER) on evaluation set
- `train_runtime`: Total training time in seconds
- `num_train_samples`: Training examples after filtering
- `num_eval_samples`: Evaluation examples

## 🔗 HuggingFace Model Cards

### WAV2VEC2-BERT (Recommended)

- **Repo**: [manassehzw/sna-w2v-bert-2.0-asr](https://huggingface.co/manassehzw/sna-w2v-bert-2.0-asr)
- **Training Dataset**: [manassehzw/sna-dataset-annotated](https://huggingface.co/datasets/manassehzw/sna-dataset-annotated)

### Whisper (Comparison Baseline)

- Available in separate HuggingFace repository if published

## 🛠️ Advanced Configuration

### Custom Hyperparameters

Pass arguments to training scripts:

```bash
uv run modal run run_w2v2_bert_finetune.py \
  --run-name sna-w2v2-custom \
  --learning-rate 3e-4 \
  --batch-size 8 \
  --max-steps 10000 \
  --eval-steps 500
```

### Data Filtering

The preprocessing pipeline includes:

- Audio duration filtering (e.g., 1–30 seconds)
- Transcript length validation
- Automatic resampling to 16 kHz
- Invalid row filtering with detailed logging

### Checkpointing & Resumption

Models save periodically during training. To resume:

```bash
uv run modal run run_w2v2_bert_finetune.py \
  --run-name sna-w2v2-v1 \
  --resume-from-checkpoint latest
```

## 📝 Notes & Limitations

- **Evaluation Dataset**: Results reported on a held-out evaluation split from the training corpus, not an external benchmark
- **Domain Adaptation**: Performance may vary on:
  - Noisy or low-quality recordings
  - Unseen speaker accents
  - Code-switched speech (Shona + English)
- **Real-time Performance**: Inference speed depends on hardware; GPU recommended for production

## 🤝 Contributing

To adapt this pipeline for other languages:

1. Prepare language-specific speech data with transcriptions
2. Update base model (try: `facebook/w2v-bert-2.0`, `facebook/wav2vec2-large-xlsr-53`, etc.)
3. Adjust hyperparameters (learning rate, batch size, max steps)
4. Run comparative evaluation with multiple models

## 📜 License

MIT License — see LICENSE file

## 👤 Author

**Manasseh Changachirere**  
Harare Institute of Technology  
[manasseh.dev](https://www.manasseh.dev/)

## 🙏 Acknowledgments

- **Base Models**: Meta AI (WAV2VEC2-BERT), OpenAI (Whisper)
- **Training Infrastructure**: Modal Labs
- **Model Hosting**: Hugging Face
- **Community**: Shona NLP community and contributors

---

**Questions or Issues?**  
Open an issue on GitHub or contact the maintainer.

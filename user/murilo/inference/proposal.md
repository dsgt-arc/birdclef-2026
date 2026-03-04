# Inference Pipeline: Embedding & Prediction Extraction

**Status**: in-progress
**Created**: 2026-03-04
**Author**: Murilo

## Purpose

Experiments 005–009 produced three fine-tuned teacher models (EAT, ConvNeXt, ProtoCLR) that converge to AUROC 0.939–0.945 on BirdCLEF 2025 focal recordings. These models were trained only on labeled `train_audio/` (28,564 focal recordings). The 9,726 unlabeled soundscape files in `train_soundscapes/` remain unused — and this is the data that most closely resembles the Kaggle hidden test set.

This inference pipeline extracts embeddings and per-species probability predictions from both data sources, producing the raw materials for two downstream tasks:

### 1. Pseudo-labeling unlabeled soundscapes (student-teacher)

The core BirdCLEF challenge is the domain shift between clean focal recordings and noisy real-world soundscapes. A student-teacher approach addresses this:

1. **Teacher inference** (this pipeline): Run each teacher model on `train_soundscapes/` to produce per-chunk species probabilities.
2. **Ensemble pseudo-labeling** (downstream): Combine predictions from multiple teachers (e.g., majority vote, probability averaging, SNORKEL-style weak supervision) to generate high-confidence pseudo-labels for each 5-second soundscape chunk.
3. **Student training** (future experiment): Train a student model on the union of labeled focal recordings + pseudo-labeled soundscapes. The student sees in-domain soundscape data during training, directly reducing the domain gap.

Having multiple architecturally diverse teachers (transformer, CNN, CvT) with different pretraining domains (AudioSet, ImageNet, Xeno-Canto) improves pseudo-label quality — their errors are less correlated, so ensemble agreement is a stronger signal of true positives.

### 2. Embedding-based analysis

The 384-dimensional embeddings (for ProtoCLR; 768 for EAT, 1024 for ConvNeXt) enable:

- **Nearest-neighbor retrieval**: Find soundscape chunks most similar to labeled focal recordings, identifying easy targets for pseudo-labeling.
- **Domain gap quantification**: Compare embedding distributions between `train_audio` and `train_soundscapes` to measure how well each encoder bridges the focal-to-soundscape gap.
- **Clustering**: Discover acoustic structure in unlabeled soundscapes (e.g., species co-occurrence patterns, noise profiles).

## Approach

Start with the best ProtoCLR model (exp 009, LoRA attn, AUROC=0.942) as the first teacher. The pipeline is model-agnostic — adding EAT and ConvNeXt teachers requires only a new config YAML and checkpoint path.

### Data flow

```
Audio file (ogg/wav)
  → probe duration with sf.info()
  → split into non-overlapping 5s chunks
  → _load_audio_chunk() at 16kHz
  → ProtoCLRMelSpectrogram (CPU, dataloader workers)
  → LoRA-adapted CvT-13 encoder → 384-dim embedding
  → classification head → 206 species logits → sigmoid probabilities
  → save to parquet
```

### Output schema

Two parquet files per source (`train_audio/`, `train_soundscapes/`):

**`embeddings.parquet`**:
| Column | Type | Description |
|--------|------|-------------|
| filename | str | Relative path (train_audio) or filename (soundscapes) |
| source | str | `"train_audio"` or `"train_soundscapes"` |
| chunk_idx | int | 0-indexed chunk within the file |
| start_sec | float | Chunk start time |
| end_sec | float | Chunk end time |
| embedding | list[float32] | 384-dim feature vector |

**`predictions.parquet`**:
| Column | Type | Description |
|--------|------|-------------|
| filename | str | Same as above |
| source | str | Same as above |
| chunk_idx | int | Same as above |
| start_sec | float | Same as above |
| end_sec | float | Same as above |
| \<species_1\> ... \<species_206\> | float32 | Sigmoid probability per species |

### Directory structure

```
user/murilo/inference/
├── proposal.md                            # This file
├── pyproject.toml
├── src -> ../src                          # Symlink to shared source
├── infer.py                               # Main inference script
├── config/
│   └── protoclr_lora_attn.yaml            # First teacher config
├── sbatch/
│   ├── _job.sh                            # SLURM job script (8h, 1 GPU)
│   └── run.sh                             # Launcher script
└── outputs/                               # Created at runtime
    └── <run_name>/
        ├── train_audio/
        │   ├── embeddings.parquet
        │   └── predictions.parquet
        └── train_soundscapes/
            ├── embeddings.parquet
            └── predictions.parquet
```

### Key implementation details

- **`InferenceChunkDataset`** (`src/birdclef/data/inference.py`): Reusable dataset class that takes a flat list of `{path, filename, source}` dicts, probes each file's duration, and yields `(mel_spectrogram, metadata)` per chunk. Reuses `_load_audio_chunk()` and `ProtoCLRMelSpectrogram` from the training pipeline.
- **Model loading**: Follows `train.py` exactly — `ProtoCLREncoder` → `get_peft_model()` → `LoRAFinetuneModule.load_from_checkpoint()`. The LoRA structure must be applied before loading weights.
- **Inference loop**: Two-step forward pass (`encoder` → `head`) with `torch.amp.autocast("cuda")` for fp16. Accumulates float32 embeddings and sigmoid probabilities.
- **SLURM**: Same venv-in-TMPDIR pattern as exp 009. 8-hour time limit (inference is ~3–4x faster than training).

### CLI

```bash
python infer.py --config config/protoclr_lora_attn.yaml --source train_audio
python infer.py --config config/protoclr_lora_attn.yaml --source train_soundscapes
python infer.py --config config/protoclr_lora_attn.yaml --source all
python infer.py --config config/protoclr_lora_attn.yaml --source train_audio --max-files 10  # debug
```

## Adding new teacher models

To add EAT or ConvNeXt as additional teachers:

1. Create a new config YAML (e.g., `config/eat_lora_attn.yaml`) with the appropriate encoder type, LoRA config, and checkpoint path.
2. Update `load_model()` in `infer.py` to handle the new encoder type (currently ProtoCLR-specific).
3. Run inference: `bash sbatch/run.sh config/eat_lora_attn.yaml --source all`.

Each teacher's outputs land in a separate `outputs/<run_name>/` directory. Downstream ensemble scripts consume all teacher directories together.

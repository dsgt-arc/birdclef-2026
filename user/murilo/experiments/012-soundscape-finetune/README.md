# Experiment 012: Labeled Soundscape Fine-Tuning

Fine-tune exp 011 LoRA models on labeled soundscape data to expand species coverage
from 206 → 234 and adapt to the soundscape domain.

## Models

### Source Checkpoints (exp 011, 80/10/10 split)

| Model | Checkpoint | AUROC | Embed Dim |
|-------|-----------|-------|-----------|
| EAT (ViT) | `best-epochepoch=17.ckpt` | 0.929 | 768 |
| ConvNeXt-Base | `best-epochepoch=09.ckpt` | 0.938 | 1024 |
| ProtoCLR (CvT-13) | `best-epochepoch=21.ckpt` | 0.937 | 384 |

### Head Expansion: 206 → 234 Classes

The classification head is expanded to cover all 234 submission species:
- 206 existing species: weights copied from exp 011 checkpoint (index-remapped)
- 28 new species: randomly initialized (25 insect sonotypes + 3 amphibians)

## How It Works

```
Exp 011 checkpoint (206 classes)
    ↓
Expand head: 206 → 234 (copy weights + init 28 new)
    ↓
Mixed training: focal (35,549 recordings) + soundscape (739 labeled segments)
    ↓
234-class model → ONNX export → Kaggle submission
```

## Quickstart

All commands assume you are in the experiment directory:

```bash
cd user/murilo/experiments/012-soundscape-finetune
```

### Step 1: Fast dev run (pipeline validation)

```bash
bash sbatch/run.sh --all --fast-dev-run
```

### Step 2: Full training

```bash
# Submit all 3 configs
bash sbatch/run.sh --all

# Or submit individually
bash sbatch/run.sh config/eat_soundscape.yaml
bash sbatch/run.sh config/convnext_soundscape.yaml
bash sbatch/run.sh config/protoclr_soundscape.yaml

# Dry run (preview command)
bash sbatch/run.sh --all --dry-run
```

### Step 3: Export + submit to Kaggle

Use the exp 010 pipeline with the 234-class checkpoints:

```bash
cd ../010-submission-pipeline
bash sbatch/run.sh export --model eat-2026-soundscape
python scripts/upload_kaggle.py --model eat-2026-soundscape --push-kernel
```

## Directory Structure

```
012-soundscape-finetune/
├── README.md
├── proposal.md
├── results.md
├── tasks.md
├── pyproject.toml
├── train.py                    # Training script with head expansion
├── config/
│   ├── eat_soundscape.yaml
│   ├── convnext_soundscape.yaml
│   └── protoclr_soundscape.yaml
├── sbatch/
│   ├── run.sh                  # Launcher script
│   └── _job.sh                 # SLURM job script
├── artifacts/
├── outputs/
└── logs/
```

## Data

| Source | Location | Format |
|--------|----------|--------|
| Focal recordings | `train_audio/` | Species subdirs, single-label |
| Labeled soundscapes | `train_soundscapes/` + `train_soundscapes_labels.csv` | Flat dir, multi-label |
| Taxonomy (234 species) | `taxonomy.csv` | All submission species |

## Checkpoint Paths (exp 011)

All under `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/011-lora-baseline-2026/checkpoints/`:

| Model | Path |
|-------|------|
| EAT | `2026-eat-lora-attn-mlphead/best-epochepoch=17.ckpt` |
| ConvNeXt | `2026-convnext-lora-mlp-mlphead/best-epochepoch=09.ckpt` |
| ProtoCLR | `2026-protoclr-lora-attn/best-epochepoch=21.ckpt` |

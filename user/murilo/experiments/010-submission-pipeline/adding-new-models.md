# Adding New Models to the Kaggle Submission Pipeline

How to take a trained checkpoint and get it submitted to the BirdCLEF 2026
competition. This covers the 4 files that need code changes plus the 3
commands to run.

## Pipeline Overview

```
Training (any experiment)
    ↓ checkpoint (.ckpt) + config (.yaml)
ONNX Export (GPU on PACE)
    ↓ model.onnx + mel_config.json + species_list.json
Kaggle Dataset Upload
    ↓ weights dataset on Kaggle
Kernel Push + Submit
    ↓ submission.csv → competition leaderboard
```

## 4 Files to Change

Every new model needs edits in exactly these 4 files. Use an existing model
as a template (e.g., `effnet-b0-sed-v4b` for SED models).

### 1. `sbatch/export.sbatch` — Export function

Add a bash function that calls the right export script with the checkpoint
path and config path. Also add the model name to the `case` dispatch at the
bottom.

```bash
export_effnet_b0_sed_v5a() {
    echo "=== Exporting EfficientNet-B0 SED v5a (BCE loss, 234 classes) ==="
    local CKPT
    CKPT="$(_find_best_ckpt "${CHECKPOINTS_BASE}/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v5a")"
    echo "Using checkpoint: $CKPT"
    python scripts/export_efficientnet_sed.py \
        --checkpoint "$CKPT" \
        --config "${EXP019_DIR}/config/effnet_b0_sed_v5a.yaml" \
        --train-csv "$TAXONOMY_CSV_2026" \
        --output-dir artifacts/weights/effnet-b0-sed-v5a/
}
```

Two places to add the model name in the `case` blocks:
- **Species mapping pattern** (~line 520): determines which mapping to build
- **Dispatch pattern** (~line 600): calls the export function

Use `_find_best_ckpt` when the epoch number isn't known yet (picks the most
recent `best-epoch*.ckpt` in the directory).

### 2. `scripts/upload_kaggle.py` — MODELS dict entry

Add an entry to the `MODELS` dict mapping the model name to its Kaggle slug,
local weights directory, and kernel metadata file.

```python
"effnet-b0-sed-v5a": {
    "slug": "birdclef-2026-effnet-b0-sed-v5a-weights",
    "weights_dir": "artifacts/weights/effnet-b0-sed-v5a",
    "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v5a.json",
},
```

Naming convention: `birdclef-2026-{model-slug}-weights`

### 3. `kaggle_kernel/kernel-metadata-{model}.json` — Kernel metadata

Create a JSON file that tells Kaggle which datasets to attach to the kernel.
The `dataset_sources` must reference the exact slug from step 2.

```json
{
  "id": "murilogustineli/birdclef-2026-effnet-b0-sed-v5a",
  "title": "BirdCLEF 2026 EffNet-B0 SED v5a",
  "code_file": "submission.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [
    "murilogustineli/birdclef-deps",
    "murilogustineli/birdclef-2026-effnet-b0-sed-v5a-weights"
  ],
  "competition_sources": ["birdclef-2026"],
  "kernel_sources": []
}
```

### 4. `kaggle_kernel/submission.py` — MODEL_SLUGS list

Add the weight dataset slug to `MODEL_SLUGS` near the top of the file. The
kernel iterates this list to find which weights are attached. If the slug is
missing, the kernel fails with `FileNotFoundError: No model weights found`.

```python
MODEL_SLUGS = [
    # SED ablations, 234 classes (exp 019)
    "birdclef-2026-effnet-b0-sed-v5a-weights",  # <-- add here
    ...
]
```

Order matters — the kernel uses the first match. Put newer/preferred models
earlier in the list.

## 3 Commands to Run

All commands from `user/murilo/experiments/010-submission-pipeline/`:

```bash
cd user/murilo/experiments/010-submission-pipeline

# Step 1: Export checkpoint → ONNX (GPU job on PACE, ~10 min)
bash sbatch/run.sh export --model effnet-b0-sed-v5a

# Step 2: Upload weights dataset to Kaggle
python scripts/upload_kaggle.py --model effnet-b0-sed-v5a

# Step 3: Push kernel to Kaggle (wait ~2 min if dataset is brand new)
python scripts/upload_kaggle.py --model effnet-b0-sed-v5a --push-kernel
```

Then submit the notebook to the competition from the Kaggle UI.

### New model (first-time upload)

The kernel references the weights dataset by slug. If the dataset doesn't
exist on Kaggle yet, the kernel push fails. For new models:

```bash
# 1. Upload weights (creates the dataset)
python scripts/upload_kaggle.py --model effnet-b0-sed-v5a

# 2. Wait 2-3 minutes for Kaggle to process

# 3. Push kernel
python scripts/upload_kaggle.py --model effnet-b0-sed-v5a --push-kernel
```

### Existing model (weight update)

If the dataset already exists on Kaggle (e.g., re-exporting with a newer
checkpoint), both steps work in one go:

```bash
python scripts/upload_kaggle.py --model effnet-b0-sed-v5a --push-kernel
```

## Export Scripts by Architecture

| Architecture | Export Script | Head Types Supported |
|-------------|-------------|---------------------|
| EfficientNet-B0 SED | `export_efficientnet_sed.py` | AttHead, AttBlockV2, LSEHead |
| HGNetV2-B0 SED | `export_efficientnet_sed.py` | Same (uses generic timm encoder) |
| ProtoCLR (CvT-13) | `export_protoclr.py` | LoRA + TorchScript fallback |
| ConvNeXt-Base | `export_convnext.py` | LoRA |
| EAT (ViT) | `export_eat.py` | LoRA |

The SED export script (`export_efficientnet_sed.py`) handles any timm-based
backbone via `EfficientNetEncoder` with `in_chans=1`. The head type is read
from the config YAML (`sed_head.type`).

## Troubleshooting

**`FileNotFoundError: No model weights found`** — The weight slug is missing
from `MODEL_SLUGS` in `submission.py`. This is the most common mistake when
adding new models.

**`ERROR: No best-epoch*.ckpt found`** — Training hasn't finished yet, or
the checkpoint directory path is wrong. Check the `CHECKPOINTS_BASE` and
run name in the config.

**Kernel push fails with dataset not found** — The weights dataset hasn't
been created on Kaggle yet. Run `upload_kaggle.py` without `--push-kernel`
first, wait a few minutes, then push.

# Kaggle Package Build Command

Build artifacts for Kaggle submission: dependency wheels, ONNX model export, and weight staging.

## Arguments
- `$ARGUMENTS` - Required: "deps", "export", or "all"
  - `deps` - Download dependency wheels for offline Kaggle install
  - `export` - Export fine-tuned model to ONNX (requires checkpoint)
  - `all` - Build deps + export + stage everything

## Paths

```
EXP_DIR=user/anthony/experiments/009-birdclef-pipeline
```

## Instructions

### Building Dependencies (`deps`)

Downloads manylinux wheels for Kaggle's Python 3.12 + x86_64 environment.

1. **Run the deps build script**:
   ```bash
   cd $EXP_DIR && bash scripts/build_deps.sh
   ```

   This downloads to `artifacts/dist/dependencies/`:
   - onnxruntime, numpy, pandas, soundfile, librosa wheels
   - mamba-onnx pre-built wheel from GitHub releases
   - requirements.txt

2. **Verify**:
   ```bash
   ls -lh $EXP_DIR/artifacts/dist/dependencies/
   ```

### Exporting Model (`export`)

Exports the fine-tuned SSAMBAClassifier to ONNX with mamba-onnx custom ops.

1. **Find the best checkpoint**:
   ```bash
   ls -lt $EXP_DIR/logs/finetune/checkpoints/*.ckpt | head -3
   ```

2. **Submit ONNX export job** (needs GPU for loading PyTorch checkpoint):
   ```bash
   cd $EXP_DIR && sbatch sbatch/export-onnx.sbatch [checkpoint_path]
   ```
   If no checkpoint path given, it auto-selects the most recent one.

   Or run directly if on a GPU node:
   ```bash
   cd $EXP_DIR && python scripts/export_onnx.py \
       --checkpoint logs/finetune/checkpoints/best.ckpt \
       --config configs/finetune.yaml \
       --metadata-csv ~/scratch/birdclef-2026/data/birdclef-2025/train.csv \
       --output-dir artifacts/weights/
   ```

3. **Verify outputs**:
   ```bash
   ls -lh $EXP_DIR/artifacts/weights/
   ```
   Should contain: `ssamba_classifier.onnx`, `species_list.json`, `mel_config.json`

### Building All (`all`)

1. Run deps build
2. Run ONNX export (via sbatch or directly)
3. Run upload dry-run to verify:
   ```bash
   cd $EXP_DIR && python scripts/upload_kaggle.py --dry-run
   ```

## Artifact Structure

```
009-birdclef-pipeline/artifacts/
├── dist/
│   └── dependencies/          # Offline wheel files
│       ├── onnxruntime-*.whl
│       ├── mamba_onnx-*.whl
│       ├── librosa-*.whl
│       ├── ...
│       └── requirements.txt
└── weights/                   # Model artifacts
    ├── ssamba_classifier.onnx
    ├── species_list.json
    └── mel_config.json
```

## Report

- ONNX model size (MB)
- Number of dependency wheels and total size
- Species count from species_list.json

## Notes
- PyTorch is NOT needed on Kaggle — mel via librosa, backbone via ONNX Runtime, head included in ONNX
- Targets Linux x86_64 Python 3.12 (manylinux_2_28 + manylinux_2_17)
- mamba-onnx provides AVX2 CPU kernels for Mamba selective scan

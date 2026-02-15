# Kaggle Push Command

Upload datasets and push the inference kernel to Kaggle.

## Arguments
- `$ARGUMENTS` - Optional: "datasets", "kernel", or "all" (default: "all")
  - `datasets` - Upload dependency and weight datasets only
  - `kernel` - Push the inference kernel only (datasets must exist)
  - `all` - Upload datasets then push kernel

## Paths

```
EXP_DIR=user/anthony/experiments/009-birdclef-pipeline
KAGGLE_USER=acmiyaguchi
COMPETITION=birdclef-2025
KERNEL_DIR=$EXP_DIR/kaggle_kernel
```

## Instructions

### Prerequisites

Verify artifacts exist before uploading:
```bash
ls $EXP_DIR/artifacts/dist/dependencies/*.whl
ls $EXP_DIR/artifacts/weights/ssamba_classifier.onnx
```
If missing, run `/kaggle:package all` first.

### Upload Datasets (`datasets`)

1. **Dry-run first**:
   ```bash
   cd $EXP_DIR && python scripts/upload_kaggle.py --dry-run
   ```

2. **Upload for real** (run in background — verbose output):
   ```bash
   cd $EXP_DIR && python scripts/upload_kaggle.py --kaggle-user $KAGGLE_USER
   ```

   This creates/updates two Kaggle datasets:
   - `$KAGGLE_USER/birdclef-2025-deps` — dependency wheels
   - `$KAGGLE_USER/birdclef-2025-weights` — ONNX model + species list + mel config + inference script

### Push Kernel (`kernel`)

1. **Create kernel directory** if it doesn't exist:
   ```bash
   mkdir -p $KERNEL_DIR
   ```

2. **Copy the inference script**:
   ```bash
   cp $EXP_DIR/scripts/kaggle_inference.py $KERNEL_DIR/submission.py
   ```

3. **Create `kernel-metadata.json`**:
   ```json
   {
     "id": "$KAGGLE_USER/birdclef-2025-ssamba",
     "title": "BirdCLEF 2025 SSAMBA",
     "code_file": "submission.py",
     "language": "python",
     "kernel_type": "script",
     "is_private": true,
     "enable_gpu": false,
     "enable_internet": false,
     "dataset_sources": [
       "$KAGGLE_USER/birdclef-2025-deps",
       "$KAGGLE_USER/birdclef-2025-weights"
     ],
     "competition_sources": ["$COMPETITION"],
     "kernel_sources": []
   }
   ```
   Write this to `$KERNEL_DIR/kernel-metadata.json`.

4. **Push the kernel**:
   ```bash
   kaggle kernels push -p $KERNEL_DIR
   ```

5. **Report** the kernel URL:
   ```
   https://www.kaggle.com/code/$KAGGLE_USER/birdclef-2025-ssamba
   ```

### Push All (`all`)

1. Upload datasets
2. Wait for datasets to be available (may take a few minutes)
3. Push kernel

## Report

- Dataset upload status (created vs versioned)
- Kernel push status
- Kernel URL for monitoring

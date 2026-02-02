# 002-kaggle-submission

Scripts for creating offline-compatible Kaggle submissions following the "Three Artifacts" approach:

1. **Dependencies** - Python wheel files for offline `pip install`
2. **Source Code** - Project wheel files (birdclef_adrian, birdclef_anthony)
3. **Model Weights** - Trained checkpoints

---

## Complete Workflow

### Step 1: Build Artifacts

```bash
cd ~/arc/birdclef-2026/user/adrian/experiments/002-kaggle-submission

# Test if compute nodes have network access
sbatch sbatch/test_network.sbatch
cat logs/network-test-*.out

# Build all artifacts (run on login node if network test fails)
sbatch sbatch/build_dependencies.sbatch  # Downloads dependency wheels
sbatch sbatch/build_code.sbatch          # Builds project wheels
sbatch sbatch/build_weights.sbatch       # Copies model checkpoints
```

### Step 2: Upload Datasets to Kaggle

```bash
python scripts/upload_datasets.py
```

### Step 3: Create and Push Kaggle Notebook

```bash
# Create kernel files and push to Kaggle
python scripts/create_kaggle_kernel.py --push

# Or preview first without pushing
python scripts/create_kaggle_kernel.py --dry-run
```

### Step 4: Monitor and Submit

```bash
# Check notebook execution status
kaggle kernels status cheungus/birdclef-2026-submission

# Once complete, submit to competition
kaggle competitions submit birdclef-2025 \
  -f submission.csv \
  -m "Submission from HPC"

# Check results
kaggle competitions submissions birdclef-2025
```

---

## Script Reference

### SLURM Scripts (`sbatch/`)

| Script | Description |
|--------|-------------|
| `common.sh` | Shared functions and path configuration. Sources environment variables and defines `REPO_DIR`, `EXPERIMENT_DIR`, `DIST_DIR`, etc. |
| `test_network.sbatch` | Tests if compute nodes can reach PyPI, Kaggle, and HuggingFace. Determines if you can run builds on compute nodes or need login node. |
| `build_dependencies.sbatch` | Exports `uv.lock` to `requirements.txt`, then downloads all dependency wheels for `manylinux2014_x86_64` / Python 3.10. **Requires network access.** |
| `build_code.sbatch` | Builds wheel files from `user/adrian` and `user/anthony` using `uv build --wheel`. Outputs to `artifacts/dist/code/`. |
| `build_weights.sbatch` | Copies model checkpoints (`.pt`, `.safetensors`, `.pth`) from `checkpoints/` to `artifacts/dist/weights/`. |

### Python Scripts (`scripts/`)

| Script | Description |
|--------|-------------|
| `upload_datasets.py` | Uploads the three artifact directories to Kaggle as datasets using the Kaggle CLI. Creates new datasets or versions existing ones. Use `--dry-run` to preview. |
| `create_kaggle_kernel.py` | Creates `kernel-metadata.json` and copies `submission.py`, then optionally pushes to Kaggle. Use `--push` to upload, `--dry-run` to preview. |

### Templates (`templates/`)

| File | Description |
|------|-------------|
| `kaggle_inference.py` | Template for the Kaggle notebook. Installs dependencies and project wheels offline, loads model, runs inference, outputs `submission.csv`. |

---

## Configuration

Edit `config.yaml` to customize:
- Kaggle username and dataset names
- Source directories to bundle
- Checkpoint paths

---

## Directory Structure

```
002-kaggle-submission/
├── README.md
├── config.yaml
├── checkpoints/           # Model weights to upload
│   └── dummy.pt
├── sbatch/                # SLURM job scripts
│   ├── common.sh
│   ├── test_network.sbatch
│   ├── build_dependencies.sbatch
│   ├── build_code.sbatch
│   ├── build_weights.sbatch
│   └── logs/              # SLURM output logs
├── scripts/               # Python utilities
│   ├── upload_datasets.py
│   └── create_kaggle_kernel.py
├── templates/             # Kaggle notebook templates
│   └── kaggle_inference.py
├── kaggle_kernel/         # Generated kernel (created by create_kaggle_kernel.py)
│   ├── kernel-metadata.json
│   └── submission.py
└── artifacts/             # Build outputs (gitignored)
    └── dist/
        ├── dependencies/  # Wheel files for pip install
        ├── code/          # Project wheel files
        └── weights/       # Model checkpoints
```

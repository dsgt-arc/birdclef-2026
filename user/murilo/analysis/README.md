# Model Explorer — Interactive Attribution Analysis

Interactive tool for understanding WHERE and WHY SED models fail on
specific samples. Visualizes gradient saliency, per-frame logits, and
LSE effective attention.

## Quick Start

### 1. Set up venv and install deps

```bash
# On PACE interactive session
VENV_BASE="${TMPDIR:-/tmp}"
JOB_VENV="$VENV_BASE/birdclef-${SLURM_JOB_ID:-$$}/.venv"
mkdir -p "$(dirname $JOB_VENV)"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

cd /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026/user/murilo
uv pip install -e .
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Pre-compute losses (GPU, ~3 min)

```bash
cd analysis/

# Default: v5j model
bash sbatch/run.sh

# Custom model:
bash sbatch/run.sh --config path/to/config.yaml --checkpoint path/to/ckpt --output data/losses_custom.parquet
```

This creates `data/losses_v5j.parquet` (sample metadata + losses + crop offsets) and
`data/losses_v5j_logits.npy` (raw logits for deeper analysis).

### 3. Open the Jupyter notebook

Open `model_explorer.ipynb` in VS Code / Antigravity (built-in Jupyter support).
Select the venv kernel, then run all cells.

To explore different samples, change `SAMPLE_IDX` in the "Select a sample" cell.

## Files

| File | Purpose |
|------|---------|
| `model_explorer.ipynb` | **Main tool** — Jupyter notebook for interactive exploration |
| `attribution.py` | Gradient saliency + LSE effective attention computation |
| `model_loader.py` | Config-driven model loading (any SED model) |
| `precompute_losses.py` | Batch script: val losses → parquet (GPU) |
| `model_explorer.py` | Marimo app (alternative to Jupyter, requires SSH tunneling) |
| `data/` | Pre-computed parquet + logits files |
| `sbatch/` | SLURM job scripts for precompute |

## What the notebook shows

For each selected sample, a 4-panel visualization:

1. **Mel spectrogram** — the raw input to the model
2. **Gradient saliency** — which time-frequency regions drive the prediction for the target species
3. **Top-10 species** — bar chart of predicted probabilities (red = ground truth, blue = other)
4. **Per-frame analysis** — LSE effective attention + frame probabilities across ~16 temporal frames

Plus: audio playback, GT vs top-1 saliency comparison, and batch view of top-5 highest-loss samples.

## Using with a different model

The pipeline is model-agnostic. To analyze a different SED model:

1. Run precompute with different config/checkpoint:
   ```bash
   bash sbatch/run.sh --config path/to/config.yaml --checkpoint path/to/ckpt --output data/losses_v6l.parquet
   ```
2. Update `LOSSES_PATH` (and optionally `CONFIG_PATH`, `CHECKPOINT_PATH`) in the notebook's config cell

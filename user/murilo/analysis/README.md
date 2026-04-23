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

### 2. Pre-compute losses (GPU recommended, CPU works)

```bash
cd analysis/
python precompute_losses.py \
    --config ../experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml \
    --checkpoint /storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/best-epochepoch=19.ckpt \
    --output losses_v5j.parquet \
    --device cuda
```

This creates `losses_v5j.parquet` (sample metadata + losses) and
`losses_v5j_logits.npy` (raw logits for deeper analysis).

### 3. Launch the Marimo app

```bash
# On the compute node
marimo run model_explorer.py --host 0.0.0.0 --port 8080

# Or in edit mode (to see/modify cells)
marimo edit model_explorer.py --host 0.0.0.0 --port 8080
```

### 4. SSH port forward from local machine

```bash
ssh -L 8080:localhost:8080 user@pace-login-node
```

Then open `http://localhost:8080` in your browser.

## Files

| File | Purpose |
|------|---------|
| `model_explorer.py` | Marimo app — interactive UI |
| `attribution.py` | Gradient saliency + LSE effective attention |
| `model_loader.py` | Config-driven model loading (any SED model) |
| `precompute_losses.py` | Batch script: val losses → parquet |

## Using with a Different Model

The pipeline is model-agnostic. To analyze a different SED model:

1. Point `precompute_losses.py` at a different `--config` and `--checkpoint`
2. Update the `CONFIG_PATH`, `CHECKPOINT_PATH`, and `LOSSES_PATH` in `model_explorer.py`

Any model that uses `EfficientNetEncoder` + `LSEHead/AttHead/AttBlockV2` +
`NNAudioMelSpectrogram/TtaharaMelTransform` will work.

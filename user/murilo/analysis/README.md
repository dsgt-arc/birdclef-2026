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

This creates three files in `data/`:

- `losses_v5j.parquet` — sample metadata + losses + crop offsets
- `losses_v5j_logits.npy` — raw logits for deeper analysis
- `triage_v5j.parquet` — losses enriched with `secondary_labels`,
  `saliency_informative`, and `triage_category` (see `triage.py` below).
  Triage runs automatically after precompute; re-run it manually with
  `python triage.py --losses data/losses_v5j.parquet` if you only need to
  refresh the enrichment without rerunning the GPU step.

### 3. Open the Jupyter notebook

Open `model_explorer.ipynb` in VS Code / Antigravity (built-in Jupyter support).
Select the venv kernel, then run all cells.

To explore different samples, change `SAMPLE_IDX` in the "Select a sample" cell.

## Files

| File | Purpose |
|------|---------|
| `model_explorer.ipynb` | **Interactive tool** — Jupyter notebook for single-sample exploration |
| `export_figures.py` | **Bulk tool** — programmatic export of top-K figures to disk |
| `triage.py` | **Tabular triage** — enrich losses parquet with secondary_labels, classify failure modes |
| `certainty.py` | **Certainty scalars** — add `shannon_norm`, `bernoulli_sum`, `top1_top2_gap`, `n_active` columns to the triage parquet |
| `species_composition.py` | **Species comparison** — per-category and pairwise species breakdowns (e.g. which species over-represent in `mixed_uncertainty`) |
| `run/export_<model>.sh` | Per-model wrapper with hard-coded paths (copy when adding a model) |
| `plotting.py` | Shared plot functions (4-panel, competitor comparison, batch grid) |
| `attribution.py` | Gradient saliency + LSE effective attention computation |
| `model_loader.py` | Config-driven model loading (any SED model) |
| `precompute_losses.py` | Batch script: val losses → parquet (GPU) |
| `model_explorer.py` | Marimo app (alternative to Jupyter, requires SSH tunneling) |
| `notes/` | Captured findings from analysis sessions (date-prefixed `.md`) |
| `data/` | Pre-computed parquet + logits files |
| `sbatch/` | SLURM job scripts for precompute |

## What the notebook shows

For each selected sample, a 4-panel visualization:

1. **Mel spectrogram** — the raw input to the model
2. **Gradient saliency** — which time-frequency regions drive the prediction for the target species
3. **Top-10 species** — bar chart of predicted probabilities (red = ground truth, blue = other)
4. **Per-frame analysis** — LSE effective attention + frame probabilities across ~16 temporal frames

Plus: audio playback, GT vs top-1 saliency comparison, and batch view of top-5 highest-loss samples.

## Tabular triage

`triage.py` runs automatically at the end of `sbatch/run.sh`, so every
precompute produces a matching `triage_<model>.parquet` alongside the losses
parquet. Run it manually when you only need to refresh the enrichment (e.g.,
after editing triage heuristics) without paying the GPU cost again.

It joins the losses parquet with `train.csv` to add `secondary_labels` (which
aren't in the precompute output) and flags:

- `top1_is_secondary` — the "wrong" top-1 is actually a co-occurring species
- `saliency_informative` — gt_prob in [0.05, 0.9], where saliency isn't saturated
- `triage_category` — heuristic bucket (correct_confident, correct_uncertain,
  cooccurrence_label_ambiguity, gt_invisible, confident_wrong, mixed_uncertainty)

```bash
# Summary over all samples + top-20 detail:
python triage.py --losses data/losses_v5j.parquet

# Single-sample detail:
python triage.py --losses data/losses_v5j.parquet --row-id 865
```

See `notes/2026-04-24-triage-v5j-failure-modes.md` for the first run's findings.

## Certainty scalars and species composition

Two small follow-on scripts that enrich the triage parquet with more
information. Both operate on the already-computed triage + logits files — no
GPU, no new inference.

**`certainty.py`** — per-sample scalars derived from clipwise logits:
`shannon_norm` (Shannon entropy of normalized sigmoid probs), `bernoulli_sum`
(sum of per-class Bernoulli entropies, better for multi-label), `top1_top2_gap`
(logit margin), `n_active` (species with `prob > 0.1`). Columns are appended to
`triage_<model>.parquet` in place.

```bash
python certainty.py --triage data/triage_v5j.parquet --summary
```

`shannon_norm` is largely redundant with `top1_prob` (corr ~−0.86 on v5j).
`bernoulli_sum` is the one scalar that distinguishes `confident_wrong` from
`correct_uncertain` — see
`notes/2026-04-24-certainty-and-species-composition.md` for the analysis.

**`species_composition.py`** — per-category or pairwise species stats.

```bash
# Per-category top-20 species by count:
python species_composition.py --triage data/triage_v5j.parquet --top-n 20

# Which species over-represent in one category vs another (log2 ratio):
python species_composition.py --triage data/triage_v5j.parquet \
    --compare mixed_uncertainty correct_uncertain \
    --output data/species_composition_mixed_vs_correct.csv
```

## Bulk figure export

`export_figures.py` is the programmatic counterpart to the notebook. It writes
the same set of figures for the top-K highest- and lowest-loss samples to disk,
so you can scroll through all of them in a file browser without editing the
notebook between samples.

### When to use which

- **Notebook**: single-sample deep dives, exploratory slicing by species, audio
  playback, ad-hoc tweaks.
- **export_figures.py**: first-pass survey of failure modes across many samples,
  or when comparing multiple models side-by-side.

### Run — per-model wrapper (recommended)

Per-model wrappers in `run/` hard-code the paths so you don't have to retype
them. Each file has sensible defaults and passes extra flags through to
`export_figures.py`:

```bash
bash run/export_v5j.sh                       # defaults: top-k 20, source all
bash run/export_v5j.sh --source focal        # override a flag
bash run/export_v5j.sh --top-k 50 --device cuda
```

To add a new model, copy an existing wrapper and update the four paths
(`LOSSES`, `LOGITS`, `CONFIG`, `CHECKPOINT`):

```bash
cp run/export_v5j.sh run/export_v6l.sh
# edit the four paths at the top
bash run/export_v6l.sh
```

### Run — direct invocation

```bash
python export_figures.py \
  --losses data/losses_v5j.parquet \
  --logits data/losses_v5j_logits.npy \
  --config ../experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml \
  --checkpoint /storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/best-epochepoch=19.ckpt \
  --top-k 20 \
  --batch-k 5 \
  --source all \
  --output-dir figures/ \
  --device cpu
```

### Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--top-k` | 20 | Per-sample figures per bucket (highest + lowest) |
| `--batch-k` | 5 | Samples in the summary batch figures |
| `--source` | `all` | `all` \| `focal` \| `soundscape` — filters the loss table before ranking |
| `--output-dir` | `figures/` | Root dir; model/source subdirs created underneath |
| `--model-name` | inferred | Defaults to losses filename stem minus `losses_` |
| `--device` | `cpu` | CPU: ~2–5 s/sample. Use `cuda` for larger runs. |

### Output structure

```
figures/<model_name>/<source>/
  highest_loss/
    rank01_loss<val>_row<row_id>.png             # 4-panel
    rank01_loss<val>_row<row_id>_competitor.png  # GT vs competitor (see rule)
    rank01_loss<val>_row<row_id>.wav             # 5s audio crop
    rank02_...
    ...
  lowest_loss/
    rank01_...
  summary/
    top5_highest_batch.png   # batch-k mel + saliency grid
    top5_lowest_batch.png
  manifest.json              # run config + per-rank metadata
```

The `<source>` subdir (`all` / `focal` / `soundscape`) lets multiple filtered
runs on the same model coexist without overwriting each other.

### Competitor comparison rule

The `_competitor.png` shows GT saliency side-by-side with the saliency of a
"wrong-class competitor" — the species the model was most tempted by.

| Bucket | top-1 == GT | top-1 != GT |
|--------|-------------|-------------|
| `highest_loss/` | save (competitor = **top-2**, "near miss on correct") | save (competitor = **top-1**, "confident wrong") |
| `lowest_loss/` | **skip** (nothing to compare on a clean win) | save (competitor = **top-1**, "confident wrong on easy sample" — notable edge case) |

The asymmetry is intentional: the comparison is saved only when it tells you
something about a failure mode.

### Using with a different model

Pipeline is model-agnostic:

1. Precompute losses for the new model:
   ```bash
   bash sbatch/run.sh \
     --config path/to/config.yaml \
     --checkpoint path/to/ckpt \
     --output data/losses_v6l.parquet
   ```
2. Copy an existing wrapper and edit the four paths:
   ```bash
   cp run/export_v5j.sh run/export_v6l.sh
   # edit LOSSES, LOGITS, CONFIG, CHECKPOINT at the top of the new file
   bash run/export_v6l.sh
   # figures/ gains v6l/all/ alongside v5j/all/
   ```
3. For the notebook: update `LOSSES_PATH` (and optionally `CONFIG_PATH`,
   `CHECKPOINT_PATH`) in the config cell.

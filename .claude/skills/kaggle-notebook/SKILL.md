---
name: kaggle:notebook
description: Pull a Kaggle notebook and analyze its technical details. Downloads the notebook source via kaggle CLI, reads the cells, and extracts architecture, training config, inference pipeline, and key code snippets. Use when the user wants to understand a competitor's notebook or document it as a reference.
argument-hint: "[notebook-url-or-slug]"
disable-model-invocation: true
---

# Kaggle Notebook Analyzer

Pull a Kaggle notebook via the kaggle CLI and analyze its technical content.

## Arguments

- `$ARGUMENTS` - One of:
  - Kaggle notebook URL: `https://www.kaggle.com/code/ttahara/birdclef-2026-hgnetv2-b0-baseline-training`
  - Kaggle slug: `ttahara/birdclef-2026-hgnetv2-b0-baseline-training`

## Steps

### 1. Parse the argument

- If URL: extract the slug (e.g., `ttahara/birdclef-2026-hgnetv2-b0-baseline-training`)
- If slug: use directly

### 2. Pull the notebook

```bash
SLUG="<owner>/<notebook-name>"
DEST="/tmp/kaggle-notebooks/<notebook-name>"
mkdir -p "$DEST"
kaggle kernels pull "$SLUG" -p "$DEST"
```

The notebook will be saved as `$DEST/<notebook-name>.ipynb`.

### 3. Read and analyze the notebook

Read the .ipynb file. Jupyter notebooks are JSON with cells containing source code
and markdown. Extract:

**Architecture:**
- Backbone model (name, source, pretrained weights)
- Head/pooling mechanism (attention type, pooling strategy)
- Input shape and channel configuration
- Model parameter count if available

**Mel/Spectrogram Config:**
- Sample rate, n_fft, hop_length, n_mels, fmin/fmax
- Normalization pipeline (AmplitudeToDB, min-max, AudioSet stats, etc.)
- Window duration and segment handling

**Training:**
- Loss function (exact implementation if custom)
- Optimizer and scheduler (type, lr, weight_decay, warmup)
- Epochs, batch size, gradient accumulation
- Data augmentation (MixUp details, SpecAugment, etc.)
- CV strategy (folds, stratification, grouping)

**Data Pipeline:**
- How train_audio and train_soundscapes are combined
- Label handling (primary, secondary, multi-hot encoding)
- Audio loading strategy (full file vs partial, ogg vs wav)

**Inference:**
- Segment processing (overlapping vs non-overlapping, window size)
- Model format (PyTorch, ONNX, OpenVINO)
- Ensemble strategy (fold averaging, rank normalization)
- Post-processing (TTA, smoothing, TopN, temperature)
- Timing/performance

**Results:**
- Reported scores (CV, public LB, private LB)
- Ablation results if available

### 4. Present the analysis

Output a structured technical summary with:
- Key findings relevant to our experiments
- Exact code snippets for important components
- Comparison to our current approach where relevant

### 5. Optionally save as reference doc

If the user wants to save the analysis, create a reference document at:
`docs/references/kaggle/<competition>/discussions/<slug>.md`
or
`docs/references/kaggle/<competition>/notebooks/<slug>.md`

Use the standard reference frontmatter format.

## Notes

- `kaggle kernels pull` requires kaggle CLI authentication
- Notebooks are downloaded as .ipynb (JSON format) — readable with the Read tool
- Large notebooks may need to be read in chunks (use offset/limit)
- Focus on extracting actionable technical details, not reproducing the entire notebook
- Always compare findings to our current experiment approach

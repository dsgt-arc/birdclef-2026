# Design: Soundscape Pseudo-Labeling Pipeline

## Context

Run L (EfficientNet attention, teacher-filtered, SC auroc 0.870) is our best model. We want to use it to pseudo-label 10,592 unlabeled soundscape files (127K 5s windows) and add them to training. This is the #1 technique from BirdCLEF 2025 winners.

The model produces per-frame (250ms), per-species (206) probabilities. The question is how to turn these into training labels.

## Step 1: Generate predictions (no design decision needed)

Run the teacher model on all unlabeled soundscapes. Save clip-level probabilities for all 206 species per window.

**Output**: `artifacts/soundscape_clip_probs.parquet` (~52 MB)
- filename, chunk_start, + 206 species probability columns
- 127,896 rows (10,592 files × 12 windows, excluding 66 labeled files)

Also optionally save frame-level probs for the top species per window for future temporal labeling.

## Step 2: Turn predictions into training labels (design decision)

### Option 1: Global threshold
Species with prob > T get label=1, others get 0. Confidence weighting via sample weight.
- Exp 101 recommended T=0.15-0.20
- **Pro**: Simple, proven in competition
- **Con**: Ignores per-window density. Quiet windows lose everything. Dense windows get noisy labels.

### Option 2: Top-K per window
Keep top K species per window. K could be fixed (8, median from labeled data) or adaptive.
- **Pro**: Exploits the model's ranking (AUROC 0.88), immune to probability compression
- **Con**: Empty windows get K false positives. Fixed K doesn't adapt to varying density.

### Option 3: Adaptive relative threshold
Keep species where prob > max_species_prob × ratio (e.g., 0.3).
- Empty windows: max is low → threshold is low → nothing passes
- Dense windows: max is high → many species pass
- **Pro**: Adapts naturally to per-window density
- **Con**: Adds a parameter (ratio). If max species is wrong, everything is wrong.

### Option 4: Soft labels (no thresholding)
Use raw probabilities as BCE targets. Species at 0.02 → target=0.02 (near-negative). Species at 0.7 → target=0.7 (strong positive).
- **Pro**: No threshold to tune. Preserves full uncertainty. Model learns calibrated outputs.
- **Con**: Exp 014 showed soft labels hurt when teacher is bad. But our teacher is much better (0.870 auroc vs weak probes in exp 014). Also, background species at 0.01 create tiny positive signal for everything.

### Option 5: Soft labels with floor
Like option 4 but set probs below a floor (e.g., 0.05) to exactly 0. This removes the background noise while keeping the soft positive signal.
- **Pro**: Combines soft labels with noise floor removal
- **Con**: Still needs one parameter (floor threshold)

## Discussion points

1. **Which option?** The BirdCLEF 2025 winners used Option 1 (simple threshold). But we have frame-level resolution and better models now. Soft labels (4 or 5) are more principled but riskier.

2. **Frame-level vs clip-level?** We could pseudo-label at frame level (250ms) for soundscapes, giving 20× more training signal. But the training loop currently uses clip-level BCE. Frame-level requires changing the loss.

3. **Sample weighting?** How to weight pseudo-labeled windows in the loss:
   - Fixed weight (e.g., 0.5× focal weight)
   - max_prob as weight (confident windows count more)
   - Number of species detected as weight

4. **How much data?** 127K pseudo-labeled windows vs 231K focal chunks. Should we subsample or use all?

5. **Exclude labeled files?** The 66 labeled files should be excluded from pseudo-labeling since they're already in train/val with real labels.

## Proposed approach

Start with **Option 5 (soft labels with floor=0.05)** + **max_prob as sample weight** + **clip-level BCE**. This is simple, preserves uncertainty, removes noise, and is compatible with the existing training loop.

If it doesn't help, fall back to Option 1 (hard threshold 0.15-0.20) which is battle-tested.

## Files to create

| File | Purpose |
|------|---------|
| `scripts/build_soundscape_pseudo_labels.py` | Inference on unlabeled soundscapes |
| `sbatch/build-pseudo-labels.sbatch` | GPU job |
| `src/birdclef_102/data.py` | Add `_append_pseudo_labeled_chunks()` |
| `src/birdclef_102/cli.py` | Pass pseudo config to DataModule |
| `configs/run-m-efficientnet-pseudo.yaml` | Training with pseudo-labels |
| `CLAUDE.md` | Update docs |

## Verification

1. Check parquet: 127,896 rows, probability distributions look reasonable
2. Train run M, compare SC val/auroc against run L (0.870)
3. Submit to Kaggle, compare against 0.794 baseline

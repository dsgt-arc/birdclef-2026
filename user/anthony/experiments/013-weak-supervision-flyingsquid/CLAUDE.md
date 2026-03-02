# Experiment 013: Weak Supervision with FlyingSquid

**Status**: Phase 1 IN PROGRESS — probe training implemented

## What This Experiment Does

Fuses multiple noisy label sources into calibrated per-frame soft labels using FlyingSquid's triplet method. Phase 1 uses PyTorch Lightning for probe training (CPU).

**Pipeline**: exp 010 pre-computed outputs → linear probes on embeddings → multi-resolution label matrix → FlyingSquid per species → fused soft labels for downstream training (exp 009/012).

## Data Dependencies (all from exp 010 — COMPLETE)

| Source | Path (under `~/scratch/birdclef-2026/outputs/exp010/`) | Resolution |
|--------|-------------------------------------------------------|-----------|
| BirdNET embeddings | `labels/birdnet/train_audio/parts/embed/` | 3s clips, 1024-dim, 939K rows |
| Perch embeddings | `labels/perch/train_audio/parts/embed/` | 5s clips, 1280-dim, 188K rows |
| BamBird events | `labels/bambird/train_audio/bambird_labels.parquet` | ~10ms event boundaries |
| Acoustic indices | `labels/indices/train_audio/parts/` | 3s/1s step, 939K rows |
| Species metadata | `~/scratch/birdclef-2026/data/birdclef-2025/train.csv` | per recording |
| Taxonomy | `~/scratch/birdclef-2026/data/birdclef-2025/taxonomy.csv` | 206 species → 4 taxa |

## Class Imbalance (why focal loss)

Each species probe is a binary classifier. Extreme positive-class rarity:
- **Median species**: 0.28% positive (2,663 / 939K BirdNET clips)
- **Rarest species** (10 species with 2 recordings): 0.007% positive (~66 clips)
- **Most common** (grekis, 990 recordings): 3.5% positive
- **Taxa**: Aves 96.8%, Amphibia 2.0%, Mammalia 0.6%, Insecta 0.5%

Focal loss `(1-p_t)^gamma` down-weights the dominant easy-negative majority.

## Phase 1: Linear Probes with Focal Loss

### Architecture

**Multi-label species probe** — single `nn.Linear(D, 206)`:
- Input: BirdNET 1024-dim or Perch 1280-dim embeddings
- Output: 206 independent binary logits (one per species)
- Loss: `FocalBCELoss(alpha=balanced, gamma=2.0)` + L2 regularization
- Alpha per-species: `n_neg / n_total` (auto-balanced from class frequencies)

**Multiclass taxa probe** — single `nn.Linear(D, 4)`:
- Output: 4-class logits (Aves/Amphibia/Insecta/Mammalia)
- Loss: `FocalCELoss(alpha=inv_freq, gamma=2.0)` + L2 regularization

### Package Structure

```
src/weaksup/
├── __init__.py
├── focal.py            # FocalBCELoss, FocalCELoss (nn.Module)
├── data.py             # EmbeddingDataModule (loads parquets, builds labels)
├── linear_probe.py     # SpeciesProbeModule, TaxaProbeModule + CLI
├── fuse.py             # CLI: weaksup-fuse (Phase 2, TODO)
└── evaluate.py         # CLI: weaksup-evaluate (Phase 3, TODO)
```

### CLI

```bash
# Train species probe on BirdNET embeddings
weaksup-probe species birdnet --gamma 2.0 --epochs 50

# Train taxa probe on Perch embeddings
weaksup-probe taxa perch --gamma 2.0 --epochs 50

# Train both probes for a source
weaksup-probe all birdnet
```

### Data Loading

- Reads all 50 parquet shards via polars (fast, memory-efficient)
- Joins embed `file` column with `train.csv` `filename` via relative path extraction
- Builds multi-hot species labels `(N, 206)` and integer taxa labels `(N,)`
- Splits by recording (GroupKFold-style) — no clip-level leakage
- Everything loaded into RAM as TensorDatasets (~5 GB total)
- **TODO**: Data loading is too slow (~10 min startup for BirdNET). Refactor to use `polars.scan_parquet` lazy streaming — precompute split/weights in a fast first pass over `file` column only, then stream embedding batches via `IterableDataset` instead of materializing the full matrix.

### Output

```
~/scratch/birdclef-2026/outputs/exp013/probes/
├── birdnet/
│   ├── species_probe.npz    # {coef, intercept, species_list, alpha, gamma}
│   ├── taxa_probe.npz       # {coef, intercept, taxa_list, alpha, gamma}
│   ├── species/              # Lightning logs + checkpoints
│   └── taxa/
└── perch/
    ├── species_probe.npz
    ├── taxa_probe.npz
    ├── species/
    └── taxa/
```

### Quality Gates

- **Pass**: val AUROC (macro) > 0.7 → proceed to Phase 2
- **Warn**: val AUROC 0.5–0.7 → proceed with caution
- **Fail**: val AUROC < 0.6 for most species → embeddings don't separate BirdCLEF species

### Dependencies

```
torch>=2.0, lightning>=2.0, torchmetrics>=1.0
scikit-learn>=1.3, polars>=1.0, numpy>=1.26, pyarrow>=14.0
typer>=0.9, rich>=13.0
```

## Labeling Functions (10 core + 5 optional)

### Tier 1: Neural Probes (trained in Phase 1)
- **LF0**: BirdNET species probe — multi-label linear on 1024-dim embeddings (3s)
- **LF1**: Perch species probe — multi-label linear on 1280-dim embeddings (5s)
- **LF2**: BirdNET taxa probe — 4-class linear (Aves/Amphibia/Insecta/Mammalia)

### Tier 2: Metadata
- **LF3**: Recording label — always +1 for primary species

### Tier 3: Acoustic Events
- **LF4**: BamBird ROI events — wavelet-based, ~10ms precision
- **LF5**: scikit-maad ROI detection — double-threshold dilation

### Tier 4: Spectral Indices (from exp 010)
- **LF6**: NDSI — biophony vs anthropophony
- **LF7**: Spectral contrast — harmonic peak prominence
- **LF8**: Spectral flatness — tonality (Wiener entropy)
- **LF9**: Bioacoustic Index — energy in 2-15 kHz

### Tier 5: Optional Extensions
- HawkEars, RIBBIT, onset detection, harmonic ratio, TFSD

## Probe Training Development Log

### v4: Focal alpha only (baseline)
- Species: alpha per-species (`n_neg/n_total`), shuffle, batch_size=4096
- Taxa: alpha per-class (inv freq), shuffle, batch_size=4096
- **BirdNET species AUROC: 0.761**, Perch species AUROC: 0.770
- Taxa F1 collapsed: Perch 0.042, BirdNET TBD — batches 96.8% Aves, minority taxa starved

### v5: Inverse balanced sampling + alpha (double correction — REJECTED)
- Added `WeightedRandomSampler(1/count)` for species and taxa
- **BirdNET species AUROC: 0.740** (−0.021), Perch: 0.780 (+0.01)
- BirdNET taxa F1: 0.164, Perch taxa F1: 0.009
- **Conclusion**: alpha + sampling double-correct for species. Inverse sampling oversamples
  rarest species ~69x/epoch, and alpha (already 0.999) further upweights — hurts common species
  more than it helps rare ones. Sampling is correct mechanism for taxa (4-class multiclass)
  but wrong for 206 independent binary heads.

### v6: Alpha-only species + balanced taxa + batch 16384 (BEST)
- Species: focal alpha only, no sampling, batch_size=16384 (4x larger)
- Taxa: `WeightedRandomSampler(1/count)`, focal alpha with inv freq weights
- **BirdNET species AUROC: 0.785** (+0.024 vs v4), Perch species AUROC: 0.784 (+0.014 vs v4)
- BirdNET taxa F1: 0.146, Perch taxa F1: 0.017
- Rationale: alpha provides per-head correction (granular). Larger batch raises expected
  positives per batch for rarest species from ~0.3 to ~1.1, letting alpha work as designed.
  Taxa uses sampling because it's a single 4-class multiclass problem where sampling balance
  is straightforward and doesn't interfere with other heads.
- Added per-species AUROC bucket logging: `val/auroc_top20` (common) and `val/auroc_bot20` (rare)
- Added `--balance none/sqrt/inverse` and `--no-alpha` CLI flags
- Fixed all-zeros edge case: unlabeled clips now filtered in `load_embeddings()`

### Results summary

| Source | Metric | v4 (alpha, bs=4096) | v5 (inverse+alpha) | v6 (alpha, bs=16384) |
|--------|--------|--------------------|--------------------|---------------------|
| BirdNET | species AUROC | 0.761 | 0.740 | **0.785** |
| BirdNET | taxa F1 | — | 0.164 | 0.146 |
| Perch | species AUROC | 0.770 | 0.780 | **0.784** |
| Perch | taxa F1 | 0.042 | 0.009 | 0.017 |

**Species probes pass quality gate** (AUROC > 0.7). Both BirdNET and Perch are viable LFs.

**Taxa probes are not useful as standalone LFs.** BirdNET/Perch embeddings organize by acoustic
similarity, not taxonomy — insect/frog species scatter across the space interspersed with birds.
Individual species form compact clusters (hence good species AUROC), but taxa classes are
non-convex regions a linear probe can't separate. For Phase 2, derive taxa signal from species
probe predictions instead of training a separate taxa probe (species deterministically implies taxa).

### Key insight: alpha vs sampling for multi-label
- **Alpha** (loss-level): per-head granularity, each binary classifier gets its own correction.
  Better for multi-label (206 independent binary problems).
- **Sampling** (data-level): one shared knob, can't independently balance 206 heads.
  Better for multiclass (taxa: 4 mutually exclusive classes).
- **Never combine both** for the same problem — they double-correct.

## Phases

| Phase | What | Status | Output |
|-------|------|--------|--------|
| 1 | Train linear probes on embeddings | DONE (v6) | `probes/*.npz` |
| 2 | Build label matrix + FlyingSquid | TODO | `fused/soft_labels.parquet` |
| 3 | Evaluate consistency + calibration | TODO | diagnostic report |

## Relationship to Other Experiments

- **Reads from exp 010**: embeddings, BamBird events, acoustic indices
- **Feeds into exp 009**: fused soft labels for SSL pretrain targets
- **Feeds into exp 012**: fused soft labels for EfficientNet finetune
- **Concept doc**: `docs/concepts/weak-supervision-label-fusion.md`

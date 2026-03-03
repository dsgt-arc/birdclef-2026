# Experiment 013: Weak Supervision with FlyingSquid

**Status**: Phase 2 IMPLEMENTED — indices probe + focal/soundscape fusion ready to run

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
- Input: BirdNET 1024-dim, Perch 1280-dim embeddings, or 4 acoustic index features
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
├── fuse.py             # CLI: weaksup-fuse (Phase 2)
└── evaluate.py         # CLI: weaksup-evaluate (Phase 2 diagnostics)
```

### CLI

```bash
# Train species probe on BirdNET embeddings
weaksup-probe species birdnet --gamma 2.0 --epochs 50

# Train species probe on acoustic indices (4 features, z-score normalized)
weaksup-probe species indices --gamma 2.0 --epochs 50 --batch-size 16384

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
├── perch/
│   ├── species_probe.npz
│   ├── taxa_probe.npz
│   ├── species/
│   └── taxa/
└── indices/
    └── species_probe.npz    # {coef, intercept, species_list, alpha, gamma, feature_mean, feature_std, feature_names}
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

## Phase 2: Label Fusion Architecture

### 3-Tier Fusion (evolved from original 10-LF plan)

The original plan used 7-10 LFs as equal inputs to the triplet method. This failed because
species-agnostic LFs (BamBird, NDSI, spectral contrast, bioacoustic index) are correlated —
they all detect "any biological sound" — and dominated the triplet method, causing inflated
soft labels for absent species. The solution is a 3-tier architecture:

**Tier 1: Triplet method on species-specific LFs**

Focal (4 LFs):
- **LF0**: BirdNET species probe — per-species percentile threshold on raw logits (top 5% → +1, below median → -1)
- **LF1**: Perch species probe — same thresholding, forward-fill from 5s to 1s resolution
- **LF2**: Indices species probe — nn.Linear(4, 206) on z-score normalized acoustic features (NDSI, spectral_contrast, log spectral_flatness, bioacoustic_index)
- **LF3**: Recording label — always +1 for primary species on all tokens of that recording

Soundscape (3 LFs — no recording labels available):
- **LF0**: BirdNET species probe
- **LF1**: Perch species probe
- **LF2**: Indices species probe — provides the 3rd LF needed for the triplet method minimum

**Tier 2: Per-species prevalence prior (in log-odds space)**
- `prevalence = (n_recordings / total) × 0.3`, clipped [0.001, 0.1]
- Without this, the flat 50/50 prior inflates all soft labels

**Tier 3: Activity modulation from acoustic indices (negative-only)**
- NDSI: < -0.2 → -1 (anthropophony), else abstain
- Spectral contrast: < 20 → -1 (flat/noisy), else abstain
- Bioacoustic index: < 100 → -1 (silence in 2-15kHz), else abstain
- **Negative evidence only** — "sound present" doesn't imply "species X present"
- Summed as log-odds penalty in the Bayes rule combination
- BamBird removed — its ROI detection is near-binary at 1s resolution (77% of files have events, ~26 events/file at ~128ms each → nearly every 1s bin overlaps an event), making it redundant with the per-token acoustic indices

### Key Design Decisions

**Why per-species percentile thresholding?**
Probe logits have tiny dynamic range (std ≈ 0.035). Sigmoid(logit) ≈ 0.500 ± 0.01 for all
tokens. A fixed sigmoid > 0.5 threshold gives 55.4% positive across all species (coin flip).
But the probes have real discriminative signal (d' = 2-12) — per-species percentile thresholds
capture it.

**Why not species-agnostic LFs in the triplet?**
BamBird, NDSI, SC, BI all detect "biological sound present" identically for all 206 species.
They're correlated (violating the conditional independence assumption), and the triplet method
estimates them as accuracy ~0.99. This overweights them, producing mean soft label 0.87 for
species not even in the shard.

**Why negative-only agnostic signals?**
"Silence detected" implies "no species present" (useful negative evidence). But "sound detected"
only means "something is vocalizing" — it doesn't discriminate between species. So agnostic
signals provide negative evidence but should abstain (not vote +1) when activity is detected.

### Labeling Functions (evolution from original design)

Original plan had 10 core + 5 optional LFs. Dropped: LF2 (taxa probe — low F1, non-convex),
LF5 (scikit-maad ROI — not implemented in exp 010), LF7 (spectral flatness — high correlation
with spectral contrast), LF8-9 (removed during simplification). LFs 3-6 became activity
modulation signals rather than triplet inputs.

BamBird was also removed from activity modulation — its ROI detection is near-binary at 1s
resolution (77% of files have events, ~26 events per file at ~128ms each, so nearly every
1s bin contains an event). The per-token acoustic indices (NDSI, spectral contrast, BI) already
provide finer-grained activity modulation that subsumes BamBird's contribution.

An **indices probe** (nn.Linear(4, 206) on z-score normalized NDSI, spectral_contrast,
log(spectral_flatness), bioacoustic_index) was added as the 3rd species-specific LF. This
enables soundscape fusion (which lacks recording labels) by providing the minimum 3 LFs
needed for the triplet method. The probe is trained on focal data; domain shift is partially
mitigated by z-score normalization.

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

### Phase 2 CLI

```bash
# Train indices probe (must run before fusion)
weaksup-probe species indices --gamma 2.0 --epochs 50 --batch-size 16384

# Run fusion pipeline — focal (4 species-specific LFs)
weaksup-fuse --split focal \
    --probe-dir ~/scratch/birdclef-2026/outputs/exp013/probes \
    --data-dir ~/scratch/birdclef-2026/data/birdclef-2025 \
    --exp010-dir ~/scratch/birdclef-2026/outputs/exp010/labels \
    --output-dir ~/scratch/birdclef-2026/outputs/exp013/fused \
    --n-workers 4

# Run fusion pipeline — soundscape (3 species-specific LFs)
weaksup-fuse --split soundscape \
    --probe-dir ~/scratch/birdclef-2026/outputs/exp013/probes \
    --data-dir ~/scratch/birdclef-2026/data/birdclef-2025 \
    --exp010-dir ~/scratch/birdclef-2026/outputs/exp010/labels \
    --output-dir ~/scratch/birdclef-2026/outputs/exp013/fused \
    --n-workers 4

# Evaluate fused labels
weaksup-evaluate --split focal \
    --fused-dir ~/scratch/birdclef-2026/outputs/exp013/fused \
    --data-dir ~/scratch/birdclef-2026/data/birdclef-2025 \
    --output-dir ~/scratch/birdclef-2026/outputs/exp013/eval
weaksup-evaluate --split soundscape \
    --fused-dir ~/scratch/birdclef-2026/outputs/exp013/fused \
    --output-dir ~/scratch/birdclef-2026/outputs/exp013/eval
```

### Phase 2 Output

```
~/scratch/birdclef-2026/outputs/exp013/fused/
├── focal/
│   ├── soft_labels.parquet     # (939K tokens × 208 cols: file, start_time, 206 species)
│   └── lf_accuracies.parquet   # (206 species × 4 LF accuracy estimates)
└── soundscape/
    ├── soft_labels.parquet     # (564K tokens × 208 cols)
    └── lf_accuracies.parquet   # (206 species × 3 LF accuracy estimates)
```

### Smoke Test Results

- Present species on recording tokens: mean soft label ~0.575 (correct positive signal)
- Absent species: mean soft label ~0.001 (correctly near zero)
- Species not in shard: all near 0 (0.001-0.003)

## Phases

| Phase | What | Status | Output |
|-------|------|--------|--------|
| 1 | Train linear probes on embeddings | DONE (v6) | `probes/*.npz` |
| 2 | Build label matrix + FlyingSquid fusion | IMPLEMENTED | `fused/soft_labels.parquet` |
| 3 | Evaluate consistency + calibration | IMPLEMENTED | `eval/report.txt` |

## Soundscape Support (IMPLEMENTED)

Soundscape data uses `--split soundscape` flag in `weaksup-fuse` and `weaksup-evaluate`.

**Key differences from focal**:
1. **No recording labels** — soundscapes don't have `primary_label` → focal gets 4 LFs, soundscape gets 3
2. **File key normalization** — `_normalize_file_key(path, split)`: focal extracts `species_id/recording.ogg`, soundscape extracts bare `filename.ogg`
3. **Indices probe** provides the 3rd species-specific LF needed for the triplet method (BirdNET + Perch alone = only 2, which is degenerate)

**Data paths**:
| Source | Focal | Soundscape |
|--------|-------|------------|
| BirdNET | `birdnet/train_audio/parts/embed/` | `birdnet/train_soundscapes/parts/embed/` |
| Perch | `perch/train_audio/parts/embed/` | `perch/train_soundscapes/parts/embed/` |
| Indices | `indices/train_audio/parts/` | `indices/train_soundscapes/parts/` |

**Domain shift concern**: Focal vs soundscape feature distributions differ (NDSI: 0.21 vs 0.62, BI: 526 vs 286). The indices probe is trained on focal data only but z-score normalization partially mitigates this. Even weak signal beats the alternative (degenerate 2-LF triplet).

## Relationship to Other Experiments

- **Reads from exp 010**: embeddings, BamBird events, acoustic indices
- **Feeds into exp 009**: fused soft labels for SSL pretrain targets
- **Feeds into exp 012**: fused soft labels for EfficientNet finetune
- **Concept doc**: `docs/concepts/weak-supervision-label-fusion.md`

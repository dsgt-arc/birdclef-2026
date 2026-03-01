# Experiment 010: Teacher Labels

**Status**: COMPLETE (all stages including acoustic indices)

## What This Experiment Does

Generates teacher labels for the full BirdCLEF 2025 dataset using three independent models (BirdNET, Perch, BamBird) and stores embeddings + predictions as sharded parquets. Experiment 013 consumes these outputs for weak supervision label fusion.

## Outputs (read-only for downstream)

All outputs at `~/scratch/birdclef-2026/outputs/exp010/`:

```
labels/
├── birdnet/
│   └── train_audio/parts/
│       ├── embed/part_00{00..49}.parquet   # 939K clips × 1024-dim (4.3 GB)
│       └── predict/part_00{00..49}.parquet  # 939K clips × 6522 species logits
├── perch/
│   └── train_audio/parts/
│       ├── embed/part_00{00..49}.parquet   # 188K clips × 1280-dim (1.3 GB)
│       └── predict/part_00{00..49}.parquet  # 188K clips × 10932 species logits
├── bambird/
│   ├── features/part_00{00..49}/           # 28,564 per-file parquets (ROIs + wavelet features)
│   └── train_audio/bambird_labels.parquet  # 21,814 rows: cluster output per file
└── indices/
    ├── train_audio/parts/part_00{00..49}.parquet       # 939K rows × 7 cols
    └── train_soundscapes/parts/part_00{00..19}.parquet  # 564K rows × 7 cols
```

**Key facts about existing outputs:**
- BirdNET: 3s windows, 1s step. Logits (not probabilities) — apply sigmoid.
- Perch: 5s windows, 5s step. Logits — apply sigmoid.
- BamBird: temporal event boundaries in `detected_events` column as `[[min_t, max_t], ...]`. Clustering unreliable (34 species 0% signal); use raw ROI events only.
- Species IDs are numeric (6522 BirdNET, 10932 Perch) — NOT mapped to BirdCLEF 206 species. Exp 013 trains linear probes to sidestep this.
- Soundscape outputs also exist under `train_soundscapes/parts/` (20 partitions).
- **Indices**: 3s windows, 1s step (matches BirdNET). Columns: `file`, `window_start`, `ndsi`, `spectral_contrast`, `spectral_flatness`, `bioacoustic_index`, `roi_detected`. Zero NaNs. 939K focal rows align 1:1 with BirdNET clips for direct join.

## Acoustic Indices Stage (COMPLETE)

Computes 5 acoustic indices per 3-second window (1-second step) matching BirdNET resolution.

| Index | Source | What it measures | Focal mean | Soundscape mean |
|-------|--------|-----------------|------------|-----------------|
| NDSI | scikit-maad | Biophony vs anthropophony ratio [-1,1] | 0.20 | 0.64 |
| Spectral contrast | librosa | Harmonic peak prominence | 22.8 | — |
| Spectral flatness | librosa | Tonality / Wiener entropy [0,1] | 3.7e-4 | — |
| Bioacoustic index | scikit-maad | Energy in 2-15 kHz band | 522 | 287 |
| ROI detected | scikit-maad | Any activity in window (6dB > median) | 99.9% | 100% |

**Results:**
- Focal: 939,116 rows across 27,804 files (50 partitions), 0 NaNs
- Soundscape: 564,108 rows across 9,726 files (20 partitions), 0 NaNs
- Runtime: ~4 min per partition with 4 CPU workers (~3.5 min total wall time for all partitions in parallel)
- ROI detection saturated on both sources (6dB threshold too lenient) — threshold downstream in exp 013

**CLI**: `teacher-indices run <source> <partition_id>`
**SBATCH**: `sbatch/indices-focal.sbatch` (50 partitions), `sbatch/indices-soundscape.sbatch` (20 partitions)

## Virtual Environment

Separate venv: `~/scratch/birdclef-2026/.venv-010` (TF + bambird, NOT PyTorch/mamba)

**Setup gotchas:**
- `docopt` 0.6.2 sdist broken → `[tool.uv] override-dependencies = ["docopt-ng>=0.9"]`
- `grad-cam` 1.5.5 sdist broken with uv → pre-install via `pip install grad-cam==1.5.5` before `uv sync`
- `tensorflow-hub` requires `setuptools<82`
- `uv venv --seed` needed for pip availability

## CLI Commands

```
teacher-bambird extract-features <partition_id>   # ROI extraction + wavelet features
teacher-bambird cluster                            # DBSCAN clustering (unreliable)
teacher-birdnet run <source> <partition_id>        # BirdNET inference
teacher-perch run <source> <partition_id>          # Perch TFLite inference
teacher-compile-perch compile                      # One-time TFLite conversion
teacher-stats run                                  # Aggregate statistics
teacher-benchmark run                              # Timing benchmarks
teacher-indices run <source> <partition_id>         # Acoustic indices (CPU-only)
```

## Relationship to Exp 013

Exp 013 (weak supervision / FlyingSquid) consumes exp 010 outputs:
- **Embeddings**: trains linear probes on BirdNET/Perch embeddings at native resolution
- **BamBird events**: uses raw ROI temporal boundaries as an event detection LF
- **Acoustic indices**: NDSI, spectral contrast, flatness, bioacoustic index as spectral LFs (939K rows, 1:1 join with BirdNET)
- Exp 013 does NOT read predictions (6522/10932 species logits) — probes replace species mapping

## Config

`configs/teacher.yaml`:
```yaml
data:
  focal_dir: ~/scratch/birdclef-2026/data/birdclef-2025/train_audio
  soundscape_dir: ~/scratch/birdclef-2026/data/birdclef-2025/train_soundscapes
output:
  root: ~/scratch/birdclef-2026/outputs/exp010/labels
partitions:
  focal: 50
  soundscape: 20
```

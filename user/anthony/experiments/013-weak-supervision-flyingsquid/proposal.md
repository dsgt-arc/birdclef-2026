# Proposal: Weak Supervision Feasibility with FlyingSquid

**Status**: proposed
**Created**: 2026-02-28
**Author**: Anthony

## Hypothesis

Linear classifiers trained on pre-computed BirdNET/Perch embeddings at their native temporal resolution, combined with acoustic indices and BamBird events via FlyingSquid's triplet method, produce calibrated per-frame soft labels whose estimated LF accuracies are internally consistent and whose fused outputs improve over any single teacher model's predictions.

## Background

Experiment 010 generated three independent teacher outputs on BirdCLEF 2025:
- **BirdNET**: 1024-dim embeddings + 6522-species logits (939K 3s clips)
- **Perch**: 1280-dim embeddings + 10932-species logits (188K 5s clips)
- **BamBird**: temporal event boundaries from wavelet-based ROI detection (660K ROIs, ~10ms precision)

The teacher predictions use internal numeric species IDs that haven't been mapped to BirdCLEF's 206 species. Training linear probes on the saved embeddings sidesteps this entirely — the probes learn BirdCLEF-aligned species predictions directly from the embedding space using focal recording labels as supervision.

**Acoustic indices** (NDSI, spectral contrast, spectral flatness, bioacoustic index, etc.) provide species-agnostic "something is vocalizing" signals that are structurally independent from neural classifiers. These are computed in exp 010 (new stage) and consumed here as pre-computed inputs.

**Key design**: Each source operates at its native resolution. Labels are generated programmatically at a configurable output rate (1-10 Hz) by looking up which source windows overlap each output frame.

FlyingSquid's triplet method estimates each labeling function's accuracy from pairwise/triple agreement statistics alone — no ground truth needed.

### Connections

- **Inputs from**: Experiment 010 (teacher embeddings + BamBird events + acoustic indices)
- **Concept**: `docs/concepts/weak-supervision-label-fusion.md`
- **Downstream**: Experiment 009 (pretrain), Experiment 012 (EfficientNet finetune)

## Method

### Approach

1. Train linear probes on pre-computed embeddings at native resolution (3s BirdNET, 5s Perch)
2. Generate labels at configurable 1-10 Hz by resampling all sources to the output grid
3. Discretize into {-1, 0, +1} vote matrix and run FlyingSquid per species
4. Evaluate internal consistency and calibration

### Setup

| Component | Details |
|-----------|---------|
| Data | Exp 010 outputs: BirdNET embeds (4.3 GB), Perch embeds (1.3 GB), BamBird events, acoustic indices; BirdCLEF 2025 train.csv |
| Compute | CPU-only throughout. Phase 1: 64 GB RAM, 16 CPUs, ~30 min. Phase 2-3: 32 GB RAM, ~15 min. |
| Dependencies | scikit-learn, polars, numpy, typer — no PyTorch, no audio I/O (all pre-computed) |
| Code | `user/anthony/experiments/013-weak-supervision-flyingsquid/src/weaksup/` |

### Procedure

**Phase 1: Train linear probes on embeddings** (~30 min, CPU)

1. Load BirdNET embeddings (939K 3s clips × 1024-dim) and Perch embeddings (188K 5s clips × 1280-dim) from exp 010 parquets
2. Join with `train.csv` to get species labels per recording
3. Train probes at **native resolution** — each clip gets its recording's species label:
   - **Species-level**: 206 binary `LogisticRegression(class_weight='balanced')`, one per species
   - **Taxa-level**: 4-class `LogisticRegression` (Aves/Amphibia/Insecta/Mammalia)
4. Evaluate via 90/10 stratified CV (stratified by recording, not clip), report per-species AUC

**Quality gate**: If probe AUC < 0.6 for most species, stop — embeddings don't separate BirdCLEF species.

**Phase 2: Build label matrix + run FlyingSquid** (~15 min, CPU)

Generate labels at the output resolution (default 1 Hz). For each output frame at time `t`:

1. **Neural probe LFs**: Look up which BirdNET window (3s) and Perch window (5s) contain time `t`. Use the probe's prediction for that window.
2. **BamBird LF**: Check if any ROI event boundary `[min_t, max_t]` overlaps time `t`.
3. **Acoustic index LFs**: Use pre-computed per-frame NDSI, spectral contrast, spectral flatness, bioacoustic index from exp 010.
4. **Recording label**: Constant +1 for the recording's primary species.
5. **Taxa LF**: Use taxa probe prediction from the enclosing window.

Discretize all sources into {-1, 0, +1} and run FlyingSquid per species (206 independent runs).

**Output**: `fused/soft_labels.parquet`, `fused/lf_accuracies.json`.

**Phase 3: Evaluate** (~5 min, CPU)

1. LF accuracy diagnostics — are estimates internally consistent?
2. Calibration — is P_fused monotonically increasing with LF agreement?
3. Sanity check — for species X recordings, is P(X) high? For non-X, is P(X) low?

### Multi-Resolution Label Generation

Each source operates at its native resolution and is resampled to the output grid:

```
Output grid (1 Hz):  |--0--|--1--|--2--|--3--|--4--|--5--|--6--|--7--|--8--|--9--| (seconds)

BirdNET probe (3s):  |------win0------|------win1------|------win2------|
Perch probe (5s):    |----------win0----------|----------win1----------|
BamBird (~10ms):     .....███.......████████......██......................
Acoustic indices:    |--0--|--1--|--2--|--3--|--4--|--5--|--6--|--7--|--8--|--9--|

Output frame t=4:    BirdNET→win1, Perch→win0, BamBird→overlaps, indices→frame4
```

### Labeling Functions

Organized into tiers by error mode independence.

#### Tier 1: Species-Aware Neural Probes

| LF | Source | Native res | Vote logic |
|----|--------|-----------|-----------|
| LF0 | BirdNET species probe | 3s | +1 if P > 0.5, −1 if P < 0.1, else abstain |
| LF1 | Perch species probe | 5s | +1 if P > 0.5, −1 if P < 0.1, else abstain |
| LF2 | BirdNET taxa probe | 3s | +1 if predicted taxa matches, −1 if wrong, else abstain |

*Error mode*: Neural classifiers trained on embeddings. LF0-1 correlated with each other; LF2 is coarser granularity.

#### Tier 2: Metadata

| LF | Source | Native res | Vote logic |
|----|--------|-----------|-----------|
| LF3 | Recording label | whole file | +1 for primary species, −1 for non-target recordings |

*Error mode*: Always +1 for target species regardless of audio content. Independent from all audio-derived LFs.

#### Tier 3: Acoustic Event Detection

| LF | Source | Native res | Vote logic |
|----|--------|-----------|-----------|
| LF4 | BamBird ROI events | ~10ms | +1 if ROI overlaps frame, −1 if no ROIs in file, else abstain |
| LF5 | scikit-maad ROI detection | ~10ms | +1 if double-threshold ROI overlaps, −1 if none, else abstain |

*Error mode*: Unsupervised wavelet/morphological event detection. Species-agnostic — detects "something vocalized" without identifying what. Independent from neural classifiers. LF4 and LF5 use different algorithms (BamBird wavelet vs scikit-maad double-threshold) so partially independent from each other.

Note: scikit-maad's `region_of_interest_index()` uses seed/low double-threshold dilation and doesn't suffer BamBird's DBSCAN clustering issues (34 species with 0% signal rate). Can serve as a complementary or replacement event detector.

#### Tier 4: Spectral Indices (Biophonic Activity)

| LF | Source | Native res | What it measures | Vote logic |
|----|--------|-----------|-----------------|-----------|
| LF6 | NDSI | output Hz | Biophony vs anthropophony ratio | +1 if NDSI > 0.2, −1 if < −0.2, else abstain |
| LF7 | Spectral contrast | output Hz | Harmonic peak prominence | +1 if high (structured vocalization), −1 if flat, else abstain |
| LF8 | Spectral flatness | output Hz | Tonality (Wiener entropy) | +1 if low (tonal/bird-like), −1 if high (noise), else abstain |
| LF9 | Bioacoustic Index | output Hz | Energy in 2-15 kHz bird band | +1 if high, −1 if low, else abstain |

*Error mode*: Pure signal processing. Independent from neural classifiers and event detectors. Partially correlated with each other (all measure aspects of spectral shape) but each captures a different facet:
- NDSI: frequency band energy ratio (biological vs anthropogenic)
- Spectral contrast: peak-to-valley difference (harmonic structure)
- Spectral flatness: geometric/arithmetic mean ratio (tonality)
- Bioacoustic Index: absolute energy in bird vocal range

#### Tier 5: Optional Extensions (if time allows)

| LF | Source | Notes |
|----|--------|-------|
| HawkEars | 5-CNN ensemble, strong-label training | Independent error mode from BirdNET/Perch. 314 NA species. Requires GPU + bioacoustics-model-zoo. |
| RIBBIT | opensoundscape pulse-rate detector | Periodic amplitude modulation. Strong for Amphibia (34 species) and Insecta (17 species). |
| Onset detection | librosa `onset_detect()` | ~16ms event boundaries. Fast BamBird alternative. |
| Harmonic ratio | `RMS(HPSS harmonic) / RMS(total)` | Noise-robust tonal energy fraction. |
| TFSD | scikit-maad `tfsd()` | Transient spectral changes in 2-8 kHz. |

### Error Mode Independence Summary

With 10 core LFs spanning 4 independent error modes:
- **Neural** (LF0-2): species/taxa classifiers on pre-trained embeddings
- **Metadata** (LF3): recording-level label, no audio content
- **Acoustic events** (LF4-5): unsupervised time-frequency event detection
- **Spectral indices** (LF6-9): frequency-domain signal processing

C(10,3) = 120 triplets for FlyingSquid accuracy estimation — very robust.

### Variables

- **Independent**: Number and composition of labeling functions (ablation: core 5 LFs → full 10 LFs); output resolution (1-10 Hz)
- **Dependent**: Estimated LF accuracies, fused label calibration, agreement with recording-level ground truth
- **Controlled**: Embedding source (exp 010), species set (206 BirdCLEF 2025), probe training procedure

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Probe CV AUC | Per-species AUC of linear probes on embeddings (Phase 1 quality gate) |
| LF accuracy estimates | FlyingSquid-estimated accuracy per LF per species (sanity check) |
| Calibration | Monotonicity of P_fused vs LF majority agreement |
| Positive recall | Fraction of frames from species X recordings with P(X) > 0.5 |
| Negative precision | Fraction of frames from non-X recordings with P(X) < 0.1 |

### Baseline

No direct numeric baseline — this is a feasibility study. Internal consistency checks substitute for ground truth comparison:
- Probe AUC provides embedding quality baseline
- Recording-level labels provide a "naive" labeling function (~0.5 expected accuracy from FlyingSquid, since it always votes +1)

### Success Criteria

- **Confirm if**: Probe AUC > 0.7 for >80% of species AND FlyingSquid LF accuracy estimates are internally consistent (probes > recording label > 0.5) AND calibration curve is monotonic
- **Reject if**: Probe AUC < 0.6 for most species (embeddings don't separate BirdCLEF species) OR FlyingSquid estimates degenerate (all accuracies near 0.5)

## Data Dependencies (all from exp 010)

| Source | Path | Resolution | Size |
|--------|------|-----------|------|
| BirdNET embeddings | `exp010/labels/birdnet/train_audio/parts/embed/` | 3s clips | 4.3 GB |
| Perch embeddings | `exp010/labels/perch/train_audio/parts/embed/` | 5s clips | 1.3 GB |
| BamBird features | `exp010/labels/bambird/features/` | ~10ms events | 2.1 GB |
| Acoustic indices | `exp010/labels/indices/` (TODO — new exp 010 stage) | per-frame | TBD |
| Species metadata | `data/birdclef-2025/train.csv` | per recording | — |

**Exp 010 TODO**: Add acoustic indices computation stage — per-frame NDSI, spectral contrast, spectral flatness, bioacoustic index, and optionally scikit-maad ROI detection. Array job over existing partition layout.

## Limitations

- Focal recording labels are noisy positives — species labeled but may not vocalize in every 3s/5s clip
- Acoustic LFs (Tier 3-4) are species-agnostic — they detect "sound activity" not species identity
- FlyingSquid assumes conditional independence between LFs — BirdNET and Perch probes are partially correlated
- Probe predictions are constant within their native window (3s or 5s) — higher output Hz doesn't add neural information, only finer acoustic/event detail
- BamBird clustering quality is limited (34/206 species have 0% signal) — using raw ROI event boundaries, not cluster labels
- No temporal ground truth available — evaluation relies on internal consistency, not accuracy against annotations
- Spectral index LFs (Tier 4) are partially correlated with each other — FlyingSquid's independence assumption is approximate here

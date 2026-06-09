# Marbled White-lipped Frog Motif Mining Synthesis

## Header

| Field | Value |
|---|---|
| Date | 2026-05-10 |
| Status | Current synthesis after simplification + calibration |
| BirdCLEF label | `22967` |
| Species | Marbled White-lipped Frog / *Leptodactylus elenae* |
| Main mining code | `src/data_analysis/marbled_frog/motif_mining.py` |
| Motif bank code | `src/data_analysis/marbled_frog/motif_bank.py` |
| Template calibration | `src/data_analysis/marbled_frog/motif_template_calibration.py` |
| Envelope diagnostics | `src/data_analysis/marbled_frog/motif_recording_diagnostics.py` |
| Main artifact root | `artifacts/marbled_frog/motif_mining_simplified/` |

## What problem we are solving

We want a defensible way to discover recurring Marbled White-lipped Frog acoustic motifs in the focal `22967` recordings, then use those motifs as candidate detectors for soundscape labeling.

The practical target is not yet a final ground-truth labeler. The current target is:

1. Find recurring time-frequency events in focal recordings.
2. Group similar events into reviewable motif families.
3. Build a small motif bank for accepted families.
4. Calibrate those templates on focal recordings.
5. Understand what assumptions must hold before scanning soundscapes.

## Current pipeline

The current pipeline is intentionally simpler than the earlier exploratory stack.

```text
focal audio
  → robust-excess-z spectrogram
  → scikit-maad hysteresis ROI proposal (`create_mask`)
  → scikit-maad ROI boxes (`select_rois`)
  → scikit-maad shape + centroid features
  → StandardScaler + sklearn HDBSCAN
  → human cluster review
  → motif bank from feature-space medoid + nearest neighbors
  → fixed physical support per motif family
  → template matching calibration on focal recordings
  → event consolidation
  → band-envelope diagnostic detector
```

### Authoritative current code

```text
src/data_analysis/marbled_frog/motif_mining.py
src/data_analysis/marbled_frog/motif_bank.py
src/data_analysis/marbled_frog/motif_template_calibration.py
src/data_analysis/marbled_frog/motif_recording_diagnostics.py
```

### Authoritative current artifacts

```text
artifacts/marbled_frog/motif_mining_simplified/tables/candidates.csv
artifacts/marbled_frog/motif_mining_simplified/tables/feature_matrix.csv
artifacts/marbled_frog/motif_mining_simplified/tables/cluster_summary.csv
artifacts/marbled_frog/motif_mining_simplified/clusters/
artifacts/marbled_frog/motif_mining_simplified/motif_bank_approved/
artifacts/marbled_frog/motif_template_calibration/
artifacts/marbled_frog/motif_recording_diagnostics/
```

The curated reference boxes are:

```text
artifacts/marbled_frog/motif_boxes/curated/tables/curated_motif_boxes.csv
```

They cover three focal recordings:

```text
iNat1448422: primary + secondary components, 8 boxes
iNat1697350: primary_independent + secondary_independent, 20 boxes
iNat1698104: chip_1k_1p8k, 5 boxes
```

## Important design decisions

### 1. Raw ROI boxes are provenance, not motif templates

Maad ROIs are thresholded connected regions. They vary with SNR, tails, overlap, and fragmentation. They are useful for saying:

```text
something event-like happened here
```

They are not stable enough to define template support directly.

The motif bank therefore separates:

```text
raw maad ROI box       = detection/provenance
fixed template support = family-level canonical crop used for template matching
```

Motif bank plots use:

```text
cyan rectangle        = fixed family template support
dashed green rectangle = original maad ROI provenance box
```

### 2. We simplified away custom heuristic stacks

We initially explored:

- pairwise template-distance outlier detection,
- recording-aware spectral-mode priors,
- weighted automatic cluster-validity scores,
- diverse exemplar selection.

Those were useful diagnostically but too hand-tuned. They were committed for history and removed from the main pipeline.

The current core relies on published/library primitives:

```text
scikit-maad ROI detection
scikit-maad shape/centroid features
sklearn HDBSCAN
feature-space medoids
fixed-support template matching
```

### 3. Template events and envelope events answer different questions

Template-bank matching detects events that look like accepted motif-family templates.

The band-envelope detector detects repeated energy peaks in a broad empirically useful band, currently:

```text
850–1600 Hz
```

For some recordings these agree well. For `iNat56387`, the envelope detector finds a dense repeated sequence that the template bank only partially detects.

That difference is not just a tuning nuisance. It means we have two related but distinct detectors:

```text
template bank = higher-precision motif-family evidence
band envelope = higher-recall repeated-vocalization evidence
```

## Empirical findings

### Simplified mining

The simplified maad/HDBSCAN run produced:

```text
270 ROI candidates
9 non-noise HDBSCAN clusters
104 candidates in noise / edge / unclustered label -1
```

Cluster 1 was visually rejected. The approved bank currently uses:

```text
clusters 0, 2, 3, 4, 5, 6, 7, 8
```

### Template calibration on focal recordings

Template matching was calibrated on the 8 focal `22967` recordings using the approved motif bank.

Raw template peaks were consolidated across related motif-family detections into biological event boxes.

Current stage metrics:

```text
raw template peaks:                  333
within-family NMS detections:         333
cross-family consolidated events:     154
reduction vs raw:                    53.8%
median components per event:          2
curated recovery after consolidation: 100%
```

Within-family NMS currently does not reduce events because peak-picking already imposes spacing. The useful simplification is cross-family consolidation.

### Curated recovery

Against curated boxes:

```text
chip_1k_1p8k:           5/5 recovered
primary:                4/4 recovered
secondary:              4/4 recovered
primary_independent:   10/10 recovered
secondary_independent: 10/10 recovered
```

This is encouraging but limited. The curated boxes cover only selected seed windows/components in three focal recordings, not full soundscape conditions.

### Recording-level envelope diagnostics

The band-envelope detector peak-picks the 850–1600 Hz robust-z mean envelope with MAD thresholds. It was added because template matching under-detected visually obvious events in `iNat56387`.

Per-recording envelope-vs-template summary:

```text
iNat1448422: 23 envelope peaks,  8 template events, 39% envelope peaks matched
iNat1697350: 19 envelope peaks, 19 template events, 95% envelope peaks matched
iNat1698104:  6 envelope peaks,  5 template events, 83% envelope peaks matched
iNat1698105: 33 envelope peaks, 30 template events, 64% envelope peaks matched
iNat1698107: 53 envelope peaks, 49 template events, 89% envelope peaks matched
iNat321313:  25 envelope peaks, 15 template events, 60% envelope peaks matched
iNat56387:   32 envelope peaks, 10 template events, 31% envelope peaks matched
iNat911059:  28 envelope peaks, 18 template events, 64% envelope peaks matched
```

The key finding is `iNat56387`:

```text
32 high-recall envelope motif events
10 template-bank events
median envelope interval ≈ 0.304 s
```

Human review indicated that essentially all of those envelope peaks look like valid motifs. So `iNat56387` demonstrates that the template bank can substantially under-count low-SNR/dense repeated sequences.

## Load-bearing assumptions

These are the assumptions that must hold for the current approach to be safe, especially before soundscape labeling.

### A1. Focal motifs transfer to soundscapes

The motif bank is learned from focal train audio. Soundscapes have different SNR, background density, overlapping species, and microphones. If focal motif shapes do not transfer, template scores will be miscalibrated.

**Failure mode:** templates miss real soundscape calls or fire on non-target species with similar spectrogram shapes.

### A2. HDBSCAN clusters represent meaningful motif families

Many HDBSCAN clusters are dominated by one or two recordings. Some may be recording-specific variants, overlapping components, or artifacts rather than species-level motif families.

**Failure mode:** bank templates overfit one iNat recording and do not generalize.

### A3. Fixed template support is a reasonable family abstraction

Motif-bank templates use one canonical physical time/frequency support per family. This makes review and matching interpretable, but it assumes family members share roughly the same duration and frequency band.

**Failure mode:** a mixed cluster or variable call type gets truncated, over-widened, or matched poorly.

### A4. Focal-calibrated thresholds are valid outside focal audio

Template thresholds are derived from focal member scores and focal background score tracks. Soundscape background is likely harder.

**Failure mode:** the focal threshold admits many soundscape false positives.

Before soundscape labeling, thresholds should be recalibrated on representative soundscape background windows.

### A5. Cross-family consolidation corresponds to biological events

Consolidation merges detections close in time or with small overlap. This works when multiple motif-family detections are components of one call.

**Failure mode:** dense choruses or overlapping individuals get over-merged into one event.

### A6. The 850–1600 Hz band-envelope detector is target-specific enough

The envelope detector performed well in focal recordings, especially `iNat56387`. But many frogs/insects/birds can produce energy in 850–1600 Hz.

**Failure mode:** band-envelope peaks in soundscapes detect non-target co-occupants.

The envelope detector should not be used as a standalone positive labeler without soundscape-specific negative/background checks.

### A7. Curated recovery validates only part of the problem

Curated recovery is 100%, but the curated set is small and targeted:

- 33 boxes,
- 3 focal recordings,
- selected windows/components.

**Failure mode:** high curated recovery may not imply high full-recording recall or soundscape precision.

`iNat56387` already shows a recall gap that curated recovery alone would miss.

## Authoritative vs deprecated artifacts

### Authoritative for current pipeline

```text
artifacts/marbled_frog/motif_mining_simplified/
artifacts/marbled_frog/motif_mining_simplified/motif_bank_approved/
artifacts/marbled_frog/motif_template_calibration/
artifacts/marbled_frog/motif_recording_diagnostics/
```

### Review/diagnostic, not final labels

```text
artifacts/marbled_frog/motif_mining_simplified/motif_bank/
```

This bank includes all non-noise clusters for review. The approved bank excludes visually rejected cluster 1.

### Deprecated / historical

Older artifact directories were removed from the working tree to avoid confusion:

```text
artifacts/marbled_frog/motif_mining/
artifacts/marbled_frog/motif_mining_maad/
artifacts/marbled_frog/motif_mining_maad_loose/
```

The exploratory code for pairwise outliers and spectral priors was committed for history, then removed from the simplified codebase:

```text
src/data_analysis/marbled_frog/motif_cluster_outliers.py
src/data_analysis/marbled_frog/motif_spectral_prior.py
```

## What should happen before soundscape labeling

### 1. Decide label target

We need to decide whether the soundscape labels should mean:

```text
A. high-precision motif-family template match
B. high-recall repeated vocalization event in 850–1600 Hz
C. hybrid evidence: template OR envelope, with separate confidence fields
```

The `iNat56387` result argues against template-only recall.

### 2. Add more focal curation

At minimum curate/check:

```text
iNat56387: dense low-SNR envelope sequence
iNat1698107 or iNat1698105: high-repeat recordings with many events
```

This is needed to validate the envelope detector and consolidation logic beyond the seed curated windows.

### 3. Recalibrate thresholds on soundscape background

Run template scoring and band-envelope scoring on representative soundscape segments known or assumed negative for `22967`. Estimate score distributions and choose thresholds from soundscape background, not focal background only.

### 4. Check co-occupants in 850–1600 Hz

Before using band-envelope peaks as positives, inspect soundscape examples for other species/insects in the same band.

### 5. Keep both detectors in outputs

For soundscapes, output scores/events separately:

```text
template_match_events
band_envelope_events
hybrid_window_scores
```

Do not collapse them into a single binary label until thresholds and false-positive rates are understood.

### 6. Validate event consolidation on dense scenes

The current consolidation is tuned on focal clips. In soundscapes, multiple callers may overlap. Keep component-level detections alongside consolidated event boxes so over-merging can be audited.

## Current interpretation

The pipeline is now good enough for **focal-audio motif characterization** and for producing **candidate soundscape evidence**, but not yet good enough for unattended soundscape pseudo-labeling.

The most load-bearing empirical finding is that template matching recovers curated seed boxes, while the simple band-envelope detector recovers a much denser valid motif sequence in `iNat56387`. Therefore any soundscape workflow should preserve both signals and treat the envelope detector as high-recall but potentially less specific.

# Concept: Unsupervised Event Detection for Per-Chunk Labeling

**Status**: exploratory/future-work seed
**Created**: 2026-02-08
**Updated**: 2026-02-08

## Idea

Use unsupervised vocalization detection (bambird/scikit-maad) to identify which temporal regions of XenoCanto training recordings contain actual bird vocalizations. When training with sliding windows (e.g., 30s chunks), chunks with no detected events become **natural negatives**, solving the no-negatives problem in BirdCLEF training data.

This approach works for the BirdCLEF competition where training data is weakly labeled (recording-level species tag, no temporal annotations) and no pre-existing event annotations like BirdSet's `detected_events` are available.

## Motivation

Current experiment 006 splits XenoCanto recordings into 30s chunks and assigns the recording-level label to every chunk. This means:
- **Zero negative examples** in training (all 13k+ chunks are positive)
- Validation data (soundscape) is ~55% empty → massive train/val distribution mismatch
- Chunks of pure silence or background noise get positive labels → noisy supervision

BirdSet's solution (bambird) shows this is tractable: unsupervised signal processing can reliably identify vocalization events in focal recordings. We can replicate this for any XenoCanto dataset without needing neural models or manual annotation.

## Background

### BirdSet's approach
BirdSet runs bambird on all XenoCanto focal recordings to produce `detected_events` timestamps. Their training pipeline then extracts segments centered on events. See [vendor doc](../vendor/bambird-scikit-maad.md).

### Why not BirdNET?
BirdNET could also work (and gives species-level predictions), but:
- Heavier dependency (TensorFlow)
- Model license restrictions (CC BY-NC-SA 4.0)
- Overkill — we already know the species from the recording label; we just need to know *when* it vocalizes
- bambird is what BirdSet uses and is proven on XenoCanto data

### Why not just energy thresholding?
A simple energy detector would work for very clean recordings but fails on:
- Recordings with loud background (rain, wind, insects)
- Species with quiet calls
- Recordings with other species vocalizing

bambird's clustering step handles these cases by separating the dominant sound source from noise/other species.

## Open Questions

- How well does bambird generalize to BirdCLEF 2026 species vs the BirdSet subset?
- Should chunks with partial overlap with events get soft labels (e.g., proportion of chunk covered by events)?
- Is the bambird pipeline fast enough to process all BirdCLEF training data (potentially 10k+ recordings)?
- Should we use event timestamps for cropping (BirdSet's approach) or for labeling chunks (our proposed approach)?
- Could we combine both: event-centered crops for positive examples + random non-event crops for negatives?

## Testable Hypotheses

1. **Per-chunk event labeling improves val/cmAP**: Training with bambird-derived per-chunk labels (positive chunks where events detected, negative where not) will significantly outperform recording-level labels on soundscape validation.

2. **Unsupervised detection is sufficient**: bambird's signal-processing approach produces event labels of comparable quality to BirdNET pseudo-labels for the purpose of chunk-level positive/negative assignment.

3. **Negatives matter more than label precision**: Even noisy negative labels (from imperfect event detection) improve model calibration compared to zero negatives.

## Proposed Approach

### Option A: Chunk-level binary labeling (simplest)
```
For each recording:
  1. Run bambird → detected_events [(t_start, t_end), ...]
  2. For each 30s chunk:
     - If any detected event overlaps this chunk → positive (keep recording label)
     - If no events overlap → negative (all-zeros label)
```

### Option B: Chunk-level proportional labeling
```
For each 30s chunk:
  - label_weight = (total event duration within chunk) / (chunk duration)
  - Use label_weight to scale the loss or as a soft label
```

### Option C: Event-centered sampling (BirdSet style)
```
For each recording:
  1. Run bambird → detected_events
  2. Positive samples: center a window on each event
  3. Negative samples: randomly sample windows with no event overlap
  4. Control positive:negative ratio (e.g., 1:1 or 2:1)
```

### Pipeline
```
XenoCanto recordings
    ↓ bambird (CPU, one-time)
per-recording event timestamps (JSON/parquet)
    ↓ chunk assignment
per-chunk labels
    ↓ training
SSAMBA with real negatives
```

## Connections

- **Based on**: [bambird + scikit-maad](../vendor/bambird-scikit-maad.md), [BirdSet benchmark](../references/birdset-benchmark-paper.md)
- **Related to**: [Soundscape domain shift](soundscape-domain-shift.md) (negatives help with domain gap)
- **Could lead to**: Experiment 007 — event-labeled chunk training

## Notes

- BirdSet already has `detected_events` for HSN dataset — could use those directly for exp 006 validation before building the full bambird pipeline for BirdCLEF competition data
- The bambird pipeline is fast on focal recordings (clean SNR, clear events) but may struggle on noisy XenoCanto uploads
- Consider caching event detections as a parquet/JSON file alongside the mel cache
- For BirdCLEF competition, this is a one-time preprocessing cost — events don't change between training runs

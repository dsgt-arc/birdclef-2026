# Experiment 010: Teacher Labels for BirdCLEF 2025

**Status**: proposed
**Created**: 2026-02-15

## Hypothesis

Running established teacher models (BamBird, BirdNET, Perch) on all BirdCLEF 2025 audio
will produce temporal event annotations and species logits suitable for weak supervision
label fusion, improving upon recording-level-only labels.

## Method

Run three teacher models on 28,564 focal recordings + 9,726 soundscapes:

| Model | Output | Window |
|-------|--------|--------|
| BamBird | Temporal event boundaries (ROIs) | Per-event |
| BirdNET | Species confidence + 1024-dim embeddings | 3s window, 1s step |
| Perch | Species confidence + 1280-dim embeddings | 5s window |

All jobs run CPU-only via SLURM array jobs.

## Outputs

- Parquet files with embeddings and predictions per partition
- BamBird ROI events as JSON per recording
- Summary statistics: species frequency, model agreement, event density

## Success Criteria

1. All partitions complete without error
2. Output schemas are consistent and compatible with downstream label fusion
3. BirdNET and Perch top-species predictions correlate on same audio
4. Total storage < 20 GB on scratch

# Activity Scores for Focal Recordings

## What

Per-5s-chunk activity scores for all 35,549 BirdCLEF 2026 focal recordings.
Indicates confidence that a chunk contains an actual vocalization vs silence/noise.

**Output**: `artifacts/focal_activity.parquet` — 244,875 chunks, 34,877 files.

## Method

Activity score = normalized spectral contrast (our best call/no-call predictor,
0.888 AUROC in exp 101). Two normalizations:

| Score | How | Use case |
|-------|-----|----------|
| `activity_score` | Per-clip min/max normalized | "Which frames in this recording are active?" |
| `activity_score_global` | Global p5/p95 quantile normalized | "How active is this frame vs the whole dataset?" |

Raw index values also included for alternative weighting.

## Source data

Acoustic indices computed in exp 010-2026 at 3s window / 1s step resolution,
aggregated to 5s chunks (mean and max within chunk).

## Known issues

**Short recordings produce degenerate per-clip scores:**

| Chunks | Problem | Count | % |
|--------|---------|-------|---|
| 1 | Always 0.5 (no variation to normalize) | 5,115 | 14.7% |
| 2 | Always 0 and 1 (binary, no magnitude info) | ~3,000 | ~8.6% |
| 3+ | Meaningful per-clip variation | ~26,700 | ~76.7% |

For short recordings, `activity_score_global` is more informative than
`activity_score`. A fallback strategy: use per-clip for recordings with ≥4
chunks, global for shorter recordings.

**NDSI provides complementary signal** but wasn't included in the score.
NDSI measures biophony (2-11 kHz) vs anthropophony (1-2 kHz), while spectral
contrast measures tonality. Combining them could help but we lack labeled
data to learn proper weights (same problem as exp 101 call/no-call).

## Spot check results

Per-clip normalization works well for longer recordings (4+ chunks):
- Call bouts get scores 0.7–1.0
- Silent gaps between calls get 0.0–0.1
- NDSI confirms: high during calls, low/negative during silence

See `scripts/spot_check_activity.py` for the full 12-clip analysis.

## Intended use

- **Sample weight** in BCE loss: high-activity chunks contribute more gradient
- **Sampling probability**: draw more chunks with actual calls during training
- **Mixup prior**: mix low-activity focal chunks with soundscape windows

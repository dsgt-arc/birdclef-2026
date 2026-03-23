# Ground Truth: Labeled Soundscape Windows

From exp 100 EDA. Source: `soundscape_labels.parquet` (deduplicated from raw CSV which
has 2x duplicate rows).

## Overview

| Metric | Value |
|--------|-------|
| Labeled files | 66 / 10,658 |
| Labeled 5s windows | **739** |
| Unlabeled gaps in labeled files | 53 windows (exclude from evaluation) |
| Species in labels | 75 (47 evaluable, 28 with zero focal data) |
| Sites | 9 / 23 |

## Species by taxa

| Taxa | Species | Evaluable | Windows present | % of 739 |
|------|---------|-----------|----------------|----------|
| Aves | 28 | 28 | 212 | 28.7% |
| Amphibia | 17 | 14 | 566 | 76.6% |
| Insecta | 25 | 0 | 168 | 22.7% |
| Mammalia | 4 | 4 | 41 | 5.5% |
| Reptilia | 1 | 1 | 13 | 1.8% |

25 Insecta are sonotypes (`47158sonXX`) with zero focal data. 3 Amphibia (1491113
Guaraní Leaf-litter Frog, 25073 *Chiasmocleis mehelyi*, 517063 Southern Orange-legged
Leaf Frog) also lack focal data. These 28 species are excluded from primary evaluation.

~5 additional species have <2 positive windows — AUROC is undefined for these.
Effective evaluable count is ~42 species.

## Window density

Median 8 species/window (range 1–20). Multi-taxa: 240/739 (32.5%).

## Per-species counts

Most species have ~12 windows. 5 species have only 1 window. Species with <5 windows
will have unreliable metrics.

## Artifacts

All in `user/anthony/experiments/100-eda/artifacts/`:
`soundscape_labels.parquet`, `soundscape_window_stats.parquet`,
`soundscape_taxa_summary.parquet`, `soundscape_coverage.parquet`

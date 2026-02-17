---
slug: "seshurajup/birdclef-2020-2025-all-training-extra-npy-dataset"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 83
---

# Kaggle Summary: BirdCLEF 2020-2025 All Training+ extra npy Dataset

**Source**: https://www.kaggle.com/code/seshurajup/birdclef-2020-2025-all-training-extra-npy-dataset
**Competition**: BirdCLEF 2025
**Score**: N/A (data processing pipeline, not a submission)

## Summary

This notebook creates a unified multi-year training dataset by aggregating historical BirdCLEF competition data from 2020-2025 plus additional audio sources. It preprocesses audio files by filtering for species present in the 2025 competition, deduplicating recordings across years, and converting audio to preprocessed numpy arrays for faster loading during training.

## Approach

### Data Sources

The notebook combines seven data sources:
- **BirdCLEF 2025**: Primary competition dataset (28,552 recordings)
- **BirdCLEF 2021**: Historical recordings (62 matching recordings after filtering)
- **BirdCLEF 2024 Additional**: Extra audio from additional MP3 dataset (83 recordings)
- **BirdCLEF 2024**: Previous year competition (13 recordings)
- **BirdCLEF 2023**: Two years prior (10 recordings)
- **BirdCLEF 2020**: Earlier competition (1 recording)
- **BirdCLEF 2022**: Historical data (0 recordings after filtering)

### Data Processing Pipeline

1. **Species Filtering**: Extracts the unique species list from BirdCLEF 2025 (primary_label column) and filters all historical datasets to only include recordings from those target species.

2. **Metadata Harmonization**: Standardizes metadata columns across different competition formats, including:
   - Converting various filename formats to unified record_name identifiers
   - Parsing secondary_labels from string representations to lists
   - For BirdCLEF 2020, mapping species names to eBird codes using a species-to-code dictionary
   - Adding source tags to track data provenance (bc20, bc21, bc22, bc23, bc24, bc25, bc00 for additional audio)

3. **Deduplication Strategy**: Removes duplicate recordings by:
   - Identifying recordings that appear in multiple years using record_name
   - Prioritizing newer competition data over older versions (sorting by source descending)
   - Keeping only the first occurrence of each unique recording

4. **Audio Preprocessing**: Loads audio files at 32kHz sample rate using librosa and extracts temporal chunks:
   - First 10 seconds of each recording
   - Last 10 seconds of each recording
   - Saves chunks as numpy arrays organized by species directory
   - This pre-extraction enables faster data loading during training by avoiding repeated file I/O

5. **Secondary Label Processing**: Preserves multi-label annotations where recordings contain multiple species, mapping historical label formats to current eBird codes.

### Output Format

- **Metadata CSV**: Consolidated dataframe with columns for primary_label, secondary_labels, record_name, source, and rank
- **Audio Arrays**: Organized directory structure with species folders containing pre-extracted 10-second numpy arrays
- **Naming Convention**: Files prefixed with 'first_10_' or 'last_10_' to indicate temporal position

## Key Ideas

- **Multi-year augmentation**: Expands training data by carefully integrating historical competition datasets, though the actual yield is relatively small (only 169 additional recordings pass filtering and deduplication)
- **Temporal chunk extraction**: Pre-extracting first and last 10-second segments reduces I/O overhead and provides natural data augmentation through temporal variation
- **Provenance tracking**: Source tags enable training strategies that weight or sample differently based on data origin (e.g., prioritizing current year data)
- **Format normalization**: Handles the evolving metadata schemas across BirdCLEF competitions (different filename structures, label formats, file extensions)
- **Deduplication priority**: Newer recordings are preferred over older versions when the same recording appears in multiple years, under the assumption that newer releases may have better quality or metadata

## Results

| Metric | Value |
|--------|-------|
| Total recordings | 28,721 (28,552 from bc25 + 169 historical) |
| Species coverage | 182 species (from bc25) |
| Additional gain | +0.6% recordings from historical data |

## Connections

- **Related concepts**: Multi-year training data aggregation, temporal audio augmentation
- **Informs experiments**: Any experiment using extended historical training data or pre-extracted audio chunks
- **Similar approaches**: Common pattern in BirdCLEF competitions to leverage historical data, though gains are typically modest due to species distribution shift across years

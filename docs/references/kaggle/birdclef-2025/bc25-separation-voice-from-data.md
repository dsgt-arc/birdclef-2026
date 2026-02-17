---
slug: "kdmitrie/bc25-separation-voice-from-data"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 181
---

# Kaggle Summary: BC25: Separation Voice from Data

**Source**: https://www.kaggle.com/code/kdmitrie/bc25-separation-voice-from-data
**Competition**: BirdCLEF 2025

## Summary

This notebook identifies and segments human voice contamination in the BirdCLEF 2025 training data. It provides techniques for detecting voice segments in both focused recordings (from specific authors) and soundscape data, enabling data cleaning and selective masking during training.

## Approach

### Problem Identification

The notebook addresses a data quality issue where recordings from certain authors (particularly Fabio A. Sarria-S and other CSA collection contributors) contain human voice annotations or commentary alongside bird vocalizations. This contamination can confuse models and degrade performance.

### Voice Detection Methodology

**Fabio's Recordings - Energy-Based Segmentation**:
- Computes short-time power by squaring audio samples
- Chunks the signal into 0.05-second segments and sums energy within each chunk
- Converts to dB scale and identifies intersections with a -50 dB threshold
- Finds the first two threshold crossings to bracket the voice segment
- Outputs start/stop timestamps for each recording in `fabio.csv`

**General CSA Collection - VAD Approach**:
- Uses Silero VAD (Voice Activity Detection) model from `snakers4/silero-vad` via torch.hub
- Processes recordings author-by-author to check for voice contamination
- Visualizes power envelope alongside VAD segmentation output
- VAD provides precise timestamp ranges for detected speech segments

**Full Dataset Scanning**:
- Scans all train_audio files and train_soundscapes
- Applies Silero VAD with default threshold of 0.5 (suggested reduction to 0.4 for higher sensitivity)
- Returns speech timestamps in seconds via `return_seconds=True`
- Saves results to `train_voice_summary.txt` (file list) and `train_voice_data.pkl` (timestamp dictionary)

### Output Format

The notebook generates:
1. `fabio.csv` - start/stop times for Fabio's recordings
2. `train_voice_summary.txt` - list of training files containing voice
3. `train_voice_data.pkl` - pickled dictionary mapping filenames to speech timestamp ranges
4. `ss_voice_summary.txt` - list of soundscape files containing voice
5. `ss_voice_data.pkl` - pickled dictionary for soundscape voice timestamps

## Key Ideas

- **Author-specific contamination patterns**: Certain fieldwork authors consistently include verbal annotations, requiring systematic cleaning
- **Energy-based segmentation for predictable patterns**: When voice appears in a consistent temporal position (e.g., start of recording), simple power thresholding suffices
- **VAD for comprehensive detection**: Silero VAD generalizes well to arbitrary voice positions and provides high-quality timestamps
- **Threshold tuning**: Default VAD threshold of 0.5 may miss subtle speech; lowering to 0.4 increases sensitivity
- **Visualization for validation**: Overlaying power curves with VAD segmentation allows manual verification before bulk processing
- **Data cleaning vs. masking**: Timestamps enable both exclusion of contaminated files and selective masking of voice segments during training

## Results

The notebook is a data analysis/cleaning tool rather than a competition entry, so no CV or leaderboard scores are reported. It successfully identifies voice contamination in:
- All 74 recordings by Fabio A. Sarria-S
- Multiple recordings across other CSA collection authors
- A subset of training audio and soundscape files (exact counts depend on threshold)

## Connections

- **Related concepts**: Data quality, audio segmentation, voice activity detection
- **Potential applications**:
  - Pre-filter training data by removing voice-contaminated files
  - Apply time-domain masking to exclude voice segments while retaining bird vocalizations
  - Weight loss contributions by voice probability to reduce contamination impact
  - Use voice timestamps as negative labels (ensure these time ranges predict "no bird")
- **Similar approaches**: Any data cleaning or quality filtering pipeline for audio competitions

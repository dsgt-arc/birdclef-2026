---
slug: "docxian/birdclef-2025-eda-geography"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 60
---

# Kaggle Summary: BirdCLEF+ 2025 - EDA & Geography

**Source**: https://www.kaggle.com/code/docxian/birdclef-2025-eda-geography
**Competition**: BirdCLEF+ 2025
**Score**: N/A (EDA notebook, not a submission)

## Summary

This notebook provides exploratory data analysis of the BirdCLEF+ 2025 training dataset with special focus on geographic distribution patterns. The analysis examines metadata structure, taxonomic composition, recording quality ratings, and spatial distributions of species, including interactive mapping capabilities.

## Approach

### Data Exploration

The notebook analyzes the training metadata by merging it with taxonomy information to understand the dataset composition. Key aspects examined include:

- **Taxonomic diversity**: The dataset contains multiple taxonomic classes beyond birds (Aves), including amphibians (Amphibia), insects (Insecta), and mammals (Mammalia), with the distribution heavily weighted toward bird species
- **Missing coordinates**: Over 800 recordings lack latitude/longitude information, requiring data cleaning to convert string "None" values to proper NaN entries
- **Recording quality**: Rating distribution shows quality assessment scores for recordings, visualized as a bar chart
- **Collection sources**: Analysis of different data collection sources contributing to the dataset

### Geographic Analysis

The notebook emphasizes spatial distribution patterns through multiple visualization approaches:

- **Global scatter plots**: Simple longitude-latitude plots showing the worldwide distribution of all recordings
- **Species-specific maps**: Separate visualizations for the top species by recording count, plotted in groups of 10 to show regional clustering
- **Taxonomic class maps**: Geographic distributions broken down by taxonomic class (birds, amphibians, insects, mammals) reveal that most taxa are concentrated in specific regions while birds have broader global coverage
- **Interactive maps**: Uses folium to create zoomable, interactive world maps with circle markers where marker size encodes the recording quality rating (radius proportional to square root of rating). Popups provide detailed metadata including primary label, secondary labels, recording type, and source URL
- **Multi-species overlays**: Demonstrates adding multiple species to the same interactive map with color-coded markers for comparison

### Audio Visualization

The notebook includes basic audio analysis techniques:

- **Waveform display**: Time-series plots of audio amplitude for individual recordings
- **Spectrogram generation**: Uses librosa to compute short-time Fourier transforms and convert amplitudes to decibel scale, visualizing frequency content over time with logarithmic frequency axis
- **Audio playback**: Embedded audio players to listen to example recordings directly in the notebook

## Key Ideas

- **Geographic bias awareness**: The dataset exhibits strong geographic clustering, with certain species concentrated in specific regions. This spatial distribution pattern is important for understanding potential domain shift between train and test data.
- **Missing coordinate handling**: Significant portions of the dataset lack location metadata, which requires careful string-to-numeric conversion and NaN handling before geographic analysis.
- **Multi-taxon dataset**: BirdCLEF+ 2025 extends beyond just bird species to include other sound-producing taxa, requiring awareness of different acoustic characteristics across taxonomic classes.
- **Rating as quality signal**: Recording quality ratings are distributed across the dataset and could be used as a weighting factor during training or for data filtering.
- **Interactive exploration value**: Folium maps with popup metadata enable efficient manual inspection of species distributions and recording quality patterns, useful for understanding data quirks before modeling.
- **Species frequency imbalance**: The frequency distribution of primary labels shows substantial class imbalance, with top species having significantly more recordings than rare ones.

## Results

N/A - This is an exploratory analysis notebook, not a predictive model.

## Connections

- **Related concepts**: Geographic distribution patterns could inform data augmentation strategies or domain adaptation approaches for soundscape evaluation
- **Informs experiments**: The multi-taxon composition (birds, frogs, insects, mammals) discovered here is relevant to understanding the full scope of the BirdCLEF+ challenge
- **Similar approaches**: Other EDA notebooks in the competition likely explore similar metadata patterns but this one uniquely emphasizes the geographic dimension with interactive mapping

# Notebook: Selected Species Train-Audio PaCMAPs

## Header

| Field | Value |
|---|---|
| Date | 2026-05-10 |
| Status | initial plots generated |
| Species set | `src/data_analysis/selected_species.py` |
| Analysis code | `src/data_analysis/soundscapes/per_species_train_pacmap.py` |
| Input embeddings | `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/train_audio/embeddings.parquet` |
| Input metadata | `/mnt/data/scratch/birdclef-2026/train.csv` |
| Output directory | `artifacts/soundscapes/per_species_train_pacmap/` |
| Projection | selected species train-audio embeddings → standardized → PCA up to 50 dims → PaCMAP |

## What was generated

For each representative train-supported labeled-soundscape species, there is one figure with:

1. **left panel**: all train-audio 5-second Perch window embeddings for that species,
2. **right panel**: one average-pooled embedding per train-audio file for that species.

The two panels for a species use a **shared PaCMAP coordinate system**, fitted jointly on that species' windows plus file-pooled means.

Point color shows whether the species is the `train.csv` primary label or appears only in `secondary_labels` for that file.

## Selected species

```text
primary_label|common_name|class_name|n_windows|n_files|plot
65380|Dwarf Tree Frog|Amphibia|102|23|65380-dwarf-tree-frog.png
22973|Whistling Grass Frog|Amphibia|355|63|22973-whistling-grass-frog.png
555146|Chaco Tree Frog|Amphibia|107|18|555146-chaco-tree-frog.png
23158|Pale-legged Weeping Frog|Amphibia|199|26|23158-pale-legged-weeping-frog.png
24279|Lesser Snouted Tree Frog|Amphibia|336|50|24279-lesser-snouted-tree-frog.png
22967|Marbled White-lipped Frog|Amphibia|33|8|22967-marbled-white-lipped-frog.png
chacha1|Chaco Chachalaca|Aves|1574|163|chacha1-chaco-chachalaca.png
whtdov|White-tipped Dove|Aves|11893|957|whtdov-white-tipped-dove.png
undtin1|Undulated Tinamou|Aves|8386|518|undtin1-undulated-tinamou.png
compau|Pauraque|Aves|6433|558|compau-pauraque.png
```

## Notes

- These plots are for `train_audio`, not labeled soundscape windows.
- File-pooled points are means over all Perch 5-second chunks in a train-audio recording.
- Secondary-label points are weak: the species is listed for the file, but may not be present in every chunk.
- The figure CSVs contain PaCMAP coordinates for both window-level and file-pooled points.

# Notebook: Labeled Soundscape Species Training Coverage

## Header

| Field | Value |
|---|---|
| Date | 2026-05-10 |
| Status | initial count complete |
| Analysis code | `src/data_analysis/soundscapes/species_coverage.py` |
| Labeled soundscape labels | `/mnt/data/scratch/birdclef-2026/train_soundscapes_labels.csv` |
| Training metadata | `/mnt/data/scratch/birdclef-2026/train.csv` |
| Train-audio windows | `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/train_audio/embeddings.parquet` |
| Output table | `artifacts/soundscapes/species_coverage/coverage.csv` |
| Output plot | `artifacts/soundscapes/species_coverage/coverage_scatter.png` |
| Output bar plot | `artifacts/soundscapes/species_coverage/train_audio_windows_by_label.png` |
| Scope | 75 labels appearing in labeled soundscapes |

## At-a-glance takeaway

For every species/taxon label that appears in the labeled soundscapes, I counted:

- how many labeled 5-second soundscape windows contain it,
- how many labeled soundscape files contain it,
- how many `train_audio` source recordings contain it as a primary or secondary label,
- how many Perch 5-second train-audio chunks those recordings produce.

Key finding: many soundscape labels have little or no matching isolated-ish `train_audio` coverage. This is especially true for insect sonotypes and a few high-count amphibians.

## Overall

| Metric | Count |
|---|---:|
| Labels found in labeled soundscapes | 75 |
| Soundscape segment-label windows | 6,244 |
| Labels with no primary-label `train_audio` files | 28 |
| Labels with no primary-or-secondary `train_audio` files | 28 |
| Sum of per-label primary-or-secondary `train_audio` files | 8,897 |
| Sum of per-label primary-or-secondary Perch 5s windows | 88,136 |

## Count definitions

| Column | Meaning |
|---|---|
| `soundscape_labeled_windows` | Number of labeled 5-second soundscape segments where the label appears. |
| `soundscape_labeled_files` | Number of labeled soundscape files containing the label. |
| `soundscape_labeled_sites` | Number of labeled soundscape sites containing the label. |
| `train_audio_primary_files/windows` | `train.csv` recordings where this is the primary label, plus their Perch 5-second chunk count. |
| `train_audio_secondary_files/windows` | recordings where this appears in `secondary_labels`; weak because the species may not occupy every chunk. |
| `train_audio_any_files/windows` | primary or secondary recordings, deduplicated per file/label. |

## Coverage scatter plot

Generated plot:

`artifacts/soundscapes/species_coverage/coverage_scatter.png`

Plot encoding:

- x-axis: `train_audio_any_windows + 1`, log scale. Points on the dashed left line have no train-audio windows.
- y-axis: `soundscape_labeled_windows`, log scale.
- color: taxon class.
- point size: number of labeled soundscape files containing the label.

The plot makes the main imbalance visible: birds mostly sit far to the right with large train-audio support; insect sonotypes stack on the left with zero train-audio support; several high-count frogs sit high on the plot but have little or no train-audio support.

## Train-audio windows by species bar plot

Generated plot:

`artifacts/soundscapes/species_coverage/train_audio_windows_by_label.png`

This horizontal bar plot shows `train_audio_any_windows` for every label found in the labeled soundscapes. It is sorted by window count and uses a symlog x-axis so the many zero/low-count labels remain visible while the high-coverage bird labels still fit on the same chart.

## Top labels by labeled-soundscape window count

```text
primary_label|common_name|class_name|soundscape_labeled_windows|soundscape_labeled_files|train_audio_any_files|train_audio_any_windows
65380|Dwarf Tree Frog|Amphibia|666|30|23|102
517063|Southern Orange-legged Leaf Frog|Amphibia|626|40|0|0
22973|Whistling Grass Frog|Amphibia|426|21|63|355
555146|Chaco Tree Frog|Amphibia|420|22|18|107
23158|Pale-legged Weeping Frog|Amphibia|350|17|26|199
24279|Lesser Snouted Tree Frog|Amphibia|346|19|50|336
24321|Mato Grosso Snouted Tree Frog|Amphibia|344|19|2|6
22967|Marbled White-lipped Frog|Amphibia|310|14|8|33
66971|Paraguayan Swimming Frog|Amphibia|298|18|5|23
47158son25|Insect sonotype25|Insecta|168|7|0|0
1491113|Guaraní leaf-litter frog|Amphibia|158|8|0|0
chacha1|Chaco Chachalaca|Aves|130|10|163|1574
whtdov|White-tipped Dove|Aves|126|6|957|11893
47158son07|Insect sonotype07|Insecta|96|4|0|0
undtin1|Undulated Tinamou|Aves|86|11|518|8386
47158son17|Insect sonotype17|Insecta|86|4|0|0
compau|Pauraque|Aves|76|7|558|6433
litnig1|Little Nightjar|Aves|74|4|126|2183
22961|Pointedbelly Frog|Amphibia|72|3|6|36
47158son13|Insect sonotype13|Insecta|72|3|0|0
```

## Taxon-level coverage pattern

```text
class_name|n_labels|soundscape_windows|train_audio_files_sum|train_audio_windows_sum|median_train_audio_files
Amphibia|17|4174|330|2062|17.0
Insecta|25|1136|0|0|0.0
Aves|28|824|8495|85408|237.5
Mammalia|4|84|71|664|12.5
Reptilia|1|26|1|2|1.0
```

## Important sparse-coverage labels

These labels occur often in labeled soundscapes but have almost no `train_audio` coverage:

```text
primary_label|common_name|class_name|soundscape_labeled_windows|train_audio_any_files|train_audio_any_windows
517063|Southern Orange-legged Leaf Frog|Amphibia|626|0|0
24321|Mato Grosso Snouted Tree Frog|Amphibia|344|2|6
66971|Paraguayan Swimming Frog|Amphibia|298|5|23
47158son25|Insect sonotype25|Insecta|168|0|0
1491113|Guaraní leaf-litter frog|Amphibia|158|0|0
47158son07|Insect sonotype07|Insecta|96|0|0
47158son17|Insect sonotype17|Insecta|86|0|0
47158son13|Insect sonotype13|Insecta|72|0|0
47158son11|Insect sonotype11|Insecta|72|0|0
47158son03|Insect sonotype03|Insecta|66|0|0
47158son10|Insect sonotype10|Insecta|66|0|0
```

## Interpretation

- The labeled soundscapes are not just a validation/diagnostic set; for some taxa they are the main or only labeled evidence.
- Insect sonotypes have soundscape labels but no matching `train_audio` recordings in `train.csv`, so isolated-audio prototype training will not cover them.
- Several amphibians are heavily represented in soundscapes but sparse or absent in `train_audio`.
- Bird labels generally have much stronger `train_audio` coverage, often hundreds of recordings and thousands of Perch windows.
- Secondary-label window counts should be treated as weak support, not frame-accurate evidence.

## Follow-up checks

- Add a rarity flag to modeling metadata: no train audio, <=5 train-audio files, <=25 train-audio windows.
- Compare model performance/confusion against this coverage table.
- For labels with no train audio but many soundscape windows, prototype or pseudo-label methods need to use soundscape-derived evidence directly.

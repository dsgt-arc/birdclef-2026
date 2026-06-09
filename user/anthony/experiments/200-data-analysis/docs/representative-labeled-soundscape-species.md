# Representative Train-Supported Labeled-Soundscape Species

## Header

| Field | Value |
|---|---|
| Date | 2026-05-10 |
| Status | selected working set |
| Source analysis | `docs/labeled-species-training-coverage.md` |
| Code constant | `src/data_analysis/selected_species.py` |
| Selection rule | representative labels from labeled soundscapes with nonzero train-audio support |

## Selected 10-label working set

This set intentionally focuses on **birds and amphibians** because they are the train-supported taxa that best represent the labeled soundscapes. Insect sonotypes are excluded because they have no matching `train_audio` coverage.

```text
primary_label|common_name|class_name|reason
65380|Dwarf Tree Frog|Amphibia|most common train-supported soundscape label
22973|Whistling Grass Frog|Amphibia|high coverage, multi-site frog
555146|Chaco Tree Frog|Amphibia|high-count frog with training support
23158|Pale-legged Weeping Frog|Amphibia|common frog, decent train support
24279|Lesser Snouted Tree Frog|Amphibia|common frog, strong train support
22967|Marbled White-lipped Frog|Amphibia|multi-site frog with sparse train support
chacha1|Chaco Chachalaca|Aves|best bird for cross-site/context analysis
whtdov|White-tipped Dove|Aves|strong train support, common in soundscapes
undtin1|Undulated Tinamou|Aves|good bird with multiple soundscape contexts
compau|Pauraque|Aves|nocturnal bird context with strong train support
```

## Notes

- This is a modeling/analysis working set, not a claim that these are the globally most important species.
- It deliberately excludes soundscape-only labels like insect sonotypes and some frogs, because the current task requires train-audio data.
- Mammals/reptiles are omitted from the core 10 because their labeled-soundscape support is comparatively small and/or train support is weak.

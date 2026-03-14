---
title: "BirdCLEF 2026 EDA Analysis"
source: "user/murilo/notebooks/003-birdclef2026-eda.ipynb"
companion_to: "docs/references/birdclef-2026-competition.md"
date: "2026-03-14"
---

# BirdCLEF 2026 — EDA Analysis

Findings from [003-birdclef2026-eda.ipynb](../../user/murilo/notebooks/003-birdclef2026-eda.ipynb), covering species distribution, soundscape labels, soundscape-only species, and secondary labels. See the [competition analysis](birdclef-2026-competition.md) for data structure and task description.

## 1. Species Distribution & Class Balance

### Taxonomic class breakdown (train_audio)

| Class | Recordings | Species | % of Data |
|---|---|---|---|
| Aves | 34,799 | 162 | 97.9% |
| Amphibia | 451 | 32 | 1.3% |
| Insecta | 199 | 3 | 0.6% |
| Mammalia | 99 | 8 | 0.3% |
| Reptilia | 1 | 1 | <0.01% |

Aves has 97.9% of all focal recordings. Non-bird taxa combined account for just 2.1% (750 recordings across 44 species). This extreme imbalance means models will be overwhelmingly biased toward birds unless explicitly corrected with loss reweighting or oversampling.

### Species gap: train_audio vs submission

| Class | In train_audio | In submission | Gap |
|---|---|---|---|
| Aves | 162 | 162 | 0 |
| Amphibia | 32 | 35 | 3 (soundscape-only) |
| Insecta | 3 | 28 | **25 (soundscape-only)** |
| Mammalia | 8 | 8 | 0 |
| Reptilia | 1 | 1 | 0 |

The Insecta gap is the starkest: 28 submission species but only 3 with focal recordings. The other 25 are unnamed sonotypes.

### Long-tail distribution

- **Mean**: 172.6 recordings/species
- **Median**: 125 recordings/species
- **Max**: 499 (rubthr1 — Rufous-bellied Thrush)
- **Min**: 1 (116570 — Southern Spectacled Caiman)
- **25 species** have <10 recordings
- **52 species** have <50 recordings
- **82 species** have <100 recordings

The distribution follows a classic long tail on log scale. Nearly 40% of species (82/206) have fewer than 100 focal recordings.

### Collection breakdown: iNat blind spot

| Class | % iNaturalist | % Xeno-Canto |
|---|---|---|
| Insecta | **100%** | 0% |
| Reptilia | **100%** | 0% |
| Amphibia | **87.1%** | 12.9% |
| Mammalia | **66.7%** | 33.3% |
| Aves | 34.0% | **66.0%** |

**Key finding**: All Insecta and Reptilia recordings come exclusively from iNaturalist. This matters because:

- **iNat has zero secondary labels** (0.0% vs 19.0% for XC) — background species in iNat recordings are completely unidentified
- **iNat has no quality ratings** — XC provides 1–5 ratings, but iNat recordings have none
- Non-bird taxa effectively have no metadata quality signal at all

## 2. Soundscape Label Analysis

### Label density

- 1,478 labeled 5s segments across 66 soundscape files
- 75 unique species annotated (of 234 submission species)
- **Mean 4.2 species per segment**, max 10
- Distribution peaks at 4–5 species/segment (46.2% of all labels)
- Only 10.6% of segments have a single species

The dense multi-label nature confirms that multi-label training (BCE loss) is essential, not optional. Single-label training would discard 89.4% of the label information.

### Co-occurrence: frog choruses dominate

**The top 15 co-occurring species pairs are all Amphibia-Amphibia.** This is the single most striking finding:

| Pair | Co-occurrences |
|---|---|
| Dwarf Tree Frog + Chaco Tree Frog | 382 |
| Dwarf Tree Frog + Southern Orange-legged Leaf Frog | 376 |
| Dwarf Tree Frog + Mato Grosso Snouted Tree Frog | 316 |
| Dwarf Tree Frog + Lesser Snouted Tree Frog | 302 |
| Dwarf Tree Frog + Pale-legged Weeping Frog | 300 |
| Dwarf Tree Frog + Paraguayan Swimming Frog | 282 |

Dwarf Tree Frog (Dendropsophus nanus) co-occurs with nearly every other frog species. Nocturnal frog choruses are the dominant multi-species acoustic event in the Pantanal soundscapes.

**Cross-class co-occurrences are 4× rarer:**

| Pair | Classes | Co-occurrences |
|---|---|---|
| White-tipped Dove + Insect sonotype07 | Aves + Insecta | 96 |
| Insect sonotype07 + Chestnut-vented Conebill | Insecta + Aves | 70 |
| Chaco Chachalaca + Insect sonotype07 | Aves + Insecta | 64 |
| Marbled White-lipped Frog + Pauraque | Amphibia + Aves | 50 |
| Whistling Grass Frog + Pauraque | Amphibia + Aves | 48 |

Cross-class pairs involve nocturnal birds (Pauraque, Tropical Screech Owl) co-occurring with frogs, or daytime insects co-occurring with birds. This confirms that **time-of-day priors could cleanly separate acoustic regimes** (nocturnal frog chorus vs diurnal bird dawn chorus).

### Implications for MixIT

The co-occurrence patterns strongly support MixIT source separation:

1. **Dense frog choruses** (300+ co-occurrences) are exactly the scenario MixIT was designed for — overlapping vocalizations from many species simultaneously
2. Cross-class separation (bird from insect, frog from bird) may be easier than within-class separation (frog from frog), since the spectral signatures differ more across taxa
3. The 4.2 average species per segment means MixIT's 4-source or 8-source models are well-matched to the actual complexity

## 3. Soundscape-Only Species

### All 28 species have labeled data

Good news: every soundscape-only species appears in at least some labeled segments. Zero species have truly zero training data.

### Data availability is wildly skewed

| Species | Class | Segments | Files |
|---|---|---|---|
| Southern Orange-legged Leaf Frog (517063) | Amphibia | 626 | 40 |
| Insect sonotype25 | Insecta | 168 | 7 |
| Guaraní leaf-litter frog (1491113) | Amphibia | 158 | 8 |
| Insect sonotype07 | Insecta | 96 | 4 |
| Insect sonotype17 | Insecta | 86 | 4 |
| ... | ... | ... | ... |
| Insect sonotype09 | Insecta | 12 | 1 |
| Insect sonotype12 | Insecta | 10 | 1 |
| Insect sonotype19 | Insecta | 10 | 2 |
| **Insect sonotype05** | **Insecta** | **6** | **2** |

The 3 amphibians are relatively well-represented (24–626 segments). Most insect sonotypes live in few-shot territory (6–96 segments), and critically many come from only 1–3 soundscape files, limiting acoustic diversity.

### Co-occurrence complexity

Soundscape-only species co-occur with an average of **12.9 unique partners** per species. They are almost never heard in isolation — they exist in dense multi-species mixtures. This means:

- Training on raw multi-label segments risks confusing a sonotype's signal with its co-occurring partners
- MixIT or similar separation would help isolate the actual sonotype signal for training
- Few-shot/metric learning approaches may need to operate on separated sources, not raw mixtures

## 4. Secondary Labels

### Coverage

- **4,372 recordings** (12.3%) have secondary labels
- **XC only**: 19.0% of XC recordings have secondary labels; **0.0% of iNat** recordings do
- 161 unique species appear as secondary labels, all within the 234 submission species

### Most common background species

| Species | As Secondary | As Primary | Common Name |
|---|---|---|---|
| grekis | 624 | 482 | Great Kiskadee |
| whtdov | 468 | 491 | White-tipped Dove |
| undtin1 | 315 | 204 | Undulated Tinamou |
| yecpar | 226 | 243 | Yellow-chevroned Parakeet |
| rufhor2 | 225 | 315 | Rufous Hornero |

The most common background species are ubiquitous Pantanal birds. Great Kiskadee appears as a background species more often (624) than as a primary species (482) — it is literally more often "in the background" than the target of a recording.

### Implications

1. **Missing labels problem**: 87.7% of recordings have no secondary labels, and 100% of iNat recordings (35% of data) have none. The actual rate of background species presence is certainly much higher. Models trained only on primary labels learn to *ignore* frequently co-occurring background species
2. **Secondary labels as weak supervision**: the 4,372 XC recordings with secondary labels could provide multi-label training signal from focal recordings, bridging toward the multi-label soundscape regime
3. **All secondaries are in submission**: no wasted labels — every background species annotation is useful for the competition

## Summary: Top Actionable Findings

1. **iNat blind spot** — non-bird taxa have zero secondary labels, zero quality ratings, and come exclusively from iNat. Quality filtering and multi-label training for these taxa must rely on other signals
2. **Frog choruses dominate soundscapes** — nocturnal Amphibia co-occurrence is 4× stronger than any cross-class pair. Time-of-day priors and MixIT separation are high-value interventions
3. **Insect sonotypes are extreme few-shot** — 6 to 168 segments each, from 1–7 files, always in dense mixtures. Few-shot learning or metric learning on MixIT-separated sources is likely needed
4. **Secondary labels are XC-only** — 12.3% coverage overall but 0% for iNat. Using them as weak supervision helps but doesn't solve the missing-label problem for non-bird taxa
5. **Dense multi-label is the norm** — 89.4% of soundscape segments have 2+ species. Multi-label BCE training is mandatory; MixIT is well-matched (4-source model for 4.2 avg species)

# Soundscape-Realistic Mixup

## Idea

Instead of random mixup, create synthetic training examples that match the
statistical properties of real Pantanal soundscapes. The labeled soundscapes
(739 windows) define the target distribution. Focal recordings are the source
material. The mixup process samples from a model of what real soundscapes
look like.

## Why

Random mixup (2 focal clips, alpha=0.15) is proven and simple. But it creates
unrealistic combinations: a Brazilian bird mixed with an African frog at equal
loudness. The resulting audio doesn't match what the model will see at inference.

Soundscape-realistic mixup would create training examples with:
- Realistic species density (8 species, not 2)
- Realistic taxa proportions (mostly frogs, some birds, some insects)
- Realistic co-occurrence (species that actually live together)
- Realistic loudness relationships (background chorus vs foreground calls)

## Target Distribution (from exp 100/101 EDA)

### Species density per window

| p25 | p50 | p75 | p90 | max |
|-----|-----|-----|-----|-----|
| 6 | 8 | 10 | 14 | 20 |

### Taxa presence rates

| Taxa | Windows present | % of 739 |
|------|----------------|----------|
| Amphibia | 566 | 76.6% |
| Aves | 212 | 28.7% |
| Insecta | 168 | 22.7% |
| Mammalia | 41 | 5.5% |
| Reptilia | 13 | 1.8% |

### Multi-taxa windows

32.5% of windows contain species from 2+ taxa. The typical Pantanal
soundscape is a frog chorus with occasional bird calls and insect sonotypes.

## Approach: Taxa-Proportional Sampling

A middle ground between random mixup and a full generative model.

### Step 1: Sample species count and taxa composition

Draw from the observed soundscape distribution:

```
N ~ empirical distribution of species-per-window (median 8, range 1-20)

For each sample:
  n_amphibia ~ Binomial(N, 0.55)  # most species are frogs
  n_aves     ~ Binomial(N - n_amphibia, 0.45)
  n_insecta  = N - n_amphibia - n_aves  # remainder
```

Or simpler: sample a real labeled window's species composition, then
substitute each species with a random focal recording of that species
(or a species from the same taxon if that exact species has no focal data).

### Step 2: Sample focal recordings for each species

For each species in the composition:
- Pick a random focal recording of that species
- Extract a random 5s chunk
- Use the activity score to prefer chunks with actual calls

### Step 3: Mix at realistic relative levels

Not all species are equally loud in a soundscape. Options:

**Simple**: Equal mixing (Dirichlet uniform) — each species contributes
equally. This is unrealistic but simple.

**Moderate**: Taxa-level loudness prior — amphibians are typically the
loudest (they're the background chorus), birds are intermittent and
quieter, insects are continuous but quiet. Draw mixing weights from
a taxa-conditioned distribution.

**Data-driven**: Use the acoustic indices from real soundscapes to
estimate the typical energy distribution. Spectral contrast and
bioacoustic index per labeled window tell us how "loud" each window is.
Cross-reference with species composition to estimate per-taxa loudness.

### Step 4: Create the mixed example

```
mixed_audio = sum(weight[i] * focal_chunk[i] for i in selected_species)
mixed_labels = union of all selected species labels
```

Labels could be:
- **Hard**: 1.0 for all selected species
- **Weighted**: weight[i] for species i (louder species get higher label)
- **Binary threshold**: 1.0 if weight > threshold, 0.0 otherwise

## Co-Occurrence Model (future refinement)

The 739 labeled windows give us a species co-occurrence matrix. Species
that co-occur frequently in real soundscapes should co-occur in synthetic
ones too.

**Simple co-occurrence**: Build a 75x75 matrix of P(species_j | species_i)
from the labeled windows. When sampling additional species for a mix,
condition on species already selected.

**Limitation**: 739 windows with 75 species is thin. Many species pairs
have zero or 1 co-occurrence. The taxa-level approach (sample amphibians
together, birds together) is more robust with this data size.

**Site-specific profiles**: Different sites have different species
communities. S22 (3,383 files) has different species than S13 (1,873 files).
A site-conditioned model could create site-realistic mixes.

## Loudness Calibration

The key challenge: focal recordings have varying loudness (some recorded
close-up, some distant). Mixing them at raw levels creates unrealistic
combinations.

Options:
- **Normalize per recording**: Divide each focal chunk by its RMS energy
  before mixing, then scale by the sampled weight
- **Match to soundscape energy**: Scale the mixed audio to have similar
  spectral contrast / bioacoustic index as real soundscape windows
- **Per-species SNR model**: Learn the typical loudness of each species in
  soundscapes from the teacher model predictions + acoustic indices

## Implementation Priority

1. **Start simple**: taxa-proportional sampling with N=8 species, equal
   mixing weights. Compare against random 2-clip mixup.
2. **If simple helps**: add loudness calibration (RMS normalization +
   taxa-level weights).
3. **If loudness helps**: add co-occurrence conditioning.
4. **If co-occurrence helps**: add site-specific profiles.

Each step adds complexity. Stop when the marginal gain drops below ~0.5%
on soundscape AUROC.

## Data Sources

| Source | What it provides |
|--------|-----------------|
| Labeled soundscapes (739 windows) | Species composition, density, co-occurrence |
| Exp 100 EDA artifacts | Taxa distribution, per-species window counts |
| Focal activity scores | Per-chunk activity confidence for selecting call-containing chunks |
| Soundscape acoustic indices | Energy profiles for loudness calibration |
| Teacher predictions (exp 101) | Per-species probabilities on soundscapes for soft co-occurrence |

## Relationship to Other Ladder Steps

- **Phase 4 (focal mixup)**: tests whether any mixup helps. If random
  2-clip mixup doesn't help, soundscape-realistic mixup won't either.
- **Phase 6 (no-call)**: the no-call examples could be mixed into the
  soundscape-realistic mixes as "background noise only" components.
- **Phase 7 (pseudo-labeling)**: iterative pseudo-labeling may subsume
  soundscape mixup — if the model trains directly on real soundscapes
  with pseudo-labels, synthetic soundscapes become less valuable.

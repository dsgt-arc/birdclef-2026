# Error Analysis: Exp 012 Models on Labeled Soundscapes

**Date**: 2026-03-22
**Notebook**: `user/murilo/notebooks/005-error-analysis.ipynb`
**Data**: 66 labeled soundscape files, 792 segments, 75/234 species evaluable

## Macro AUROC (75 evaluable species)

| Model | AUROC | Kaggle LB |
|-------|-------|-----------|
| **Ensemble (mean)** | **0.877** | — |
| ProtoCLR | 0.854 | 0.767 |
| ConvNeXt | 0.822 | 0.755 |
| EAT | 0.777 | 0.720 |

## Per-Taxon Per-Model Breakdown

| Taxon | Species | ProtoCLR | ConvNeXt | EAT | Ensemble |
|-------|---------|----------|----------|-----|----------|
| Mammalia | 4 | 0.899 | 0.869 | **0.913** | **0.927** |
| Aves | 28 | 0.854 | 0.809 | 0.823 | **0.899** |
| Amphibia | 17 | 0.844 | 0.820 | 0.834 | 0.864 |
| Insecta | 25 | 0.856 | 0.848 | **0.673** | 0.857 |
| Reptilia | 1 | 0.816 | 0.411 | 0.578 | 0.785 |

## Key Findings

### 1. EAT fails on insects (AUROC 0.673)

EAT scores 0.673 on insects — 0.18 lower than ProtoCLR (0.856) and ConvNeXt (0.848).
Root cause: EAT uses kaldi fbank at sr=16kHz (fmax=8000Hz). Insect stridulation has
significant energy above 8kHz. ProtoCLR also caps at 8kHz but handles insects better,
possibly due to Xeno-Canto pretraining exposure.

| Model | Insecta AUROC | fmax |
|-------|--------------|------|
| ProtoCLR | 0.856 | 8,000 Hz |
| ConvNeXt | 0.848 | 16,000 Hz |
| EAT | **0.673** | 8,000 Hz |

### 2. Simple mean ensemble hurts (−0.039 vs best single)

Mean ensemble helps only 8 species but hurts 39. EAT's low scores on insects and
several birds contaminate the average. A simple mean is the wrong ensemble strategy.

**Implication**: Use weighted ensemble, `min()` reduction (2024 winner pattern), or
drop EAT for non-bird taxa. Never use unweighted mean.

### 3. Low positive-segment confidence (median 0.05–0.19)

| Model | Pos Mean | Pos Median | Neg Mean | Separation |
|-------|----------|-----------|----------|-----------|
| ProtoCLR | 0.297 | 0.140 | 0.008 | 36× |
| ConvNeXt | 0.225 | 0.050 | 0.006 | 36× |
| EAT | 0.354 | 0.075 | 0.009 | 38× |
| Ensemble | 0.292 | 0.187 | 0.008 | 37× |

Separation ratio is good (36–38×), but median positive predictions are very low.
Most true-positive segments get confidence < 0.2. This is the clip-level averaging
problem: the model pools over the full 5s, diluting brief vocalizations.

**Implication**: SED with framewise-max would produce higher confidence on segments
where a species vocalizes even briefly — directly addressing this bottleneck.

### 4. Species coverage is heavily biased

| Taxon | In Labeled | Total | Coverage |
|-------|-----------|-------|----------|
| Insecta | 25 | 28 | 89% |
| Amphibia | 17 | 35 | 49% |
| Aves | 28 | 162 | **17%** |
| Mammalia | 4 | 8 | 50% |
| Reptilia | 1 | 1 | 100% |

134/159 unevaluable species are birds. Our main LB performance comes from birds,
but we can only diagnose 28/162 of them here. The labeled soundscape evaluation
over-represents insects and amphibians relative to the actual species distribution.

### 5. Worst species are mixed across taxa

| Species | Taxon | n_pos | Ensemble AUROC |
|---------|-------|-------|---------------|
| 47158son06 | Insecta | 18 | 0.470 |
| trsowl | Aves | 26 | 0.558 |
| 65377 | Amphibia | 9 | 0.602 |
| 1491113 | Amphibia | 79 | 0.611 |
| 47158son04 | Insecta | 17 | 0.660 |
| limpki | Aves | 3 | 0.675 |

Failures aren't concentrated in one taxon — insects, amphibians, and some birds
all appear in the bottom 15. Species 1491113 (Amphibia) has 79 positive segments
but only 0.611 AUROC, suggesting a representation problem, not a data scarcity issue.

## Implications for Future Experiments

1. **SED architecture (Phase A)**: The low positive confidence (median 0.05–0.19)
   is the strongest signal that clip-level classification leaves performance on the
   table. Framewise-max directly addresses this.

2. **Ensemble strategy**: Never use unweighted mean. Explore weighted averaging,
   `min()` reduction, or per-taxon model selection (drop EAT for insects).

3. **Frequency coverage**: EAT's 8kHz cap is a hard limitation for insects. Future
   models should use sr=32kHz or at least fmax=16kHz for full spectral coverage.

4. **Pseudo-labeling quality**: When using these models as teachers for exp 013,
   EAT's insect predictions will introduce noise. Consider excluding EAT from the
   teacher ensemble for insect species, or using ProtoCLR + ConvNeXt only.

5. **Need more bird evaluation data**: 134/162 bird species have no labeled
   soundscape data. The LB is our only signal for most birds.

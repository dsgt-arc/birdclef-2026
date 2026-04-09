# Experiment 024: Data-Centric Exploration — Results

## Completed Analyses

*(updated as phases complete)*

### Pre-experiment: v5j Inference EDA (2026-04-09)

Full findings in `user/murilo/notes/2026-04-09-inference-eda-deep-dive.md`.

| Finding | Value | Implication |
|---------|-------|-------------|
| Macro-AUC on 75 labeled-soundscape species | 0.9958 | Model saturated on known species |
| Species with zero soundscape data | 159 / 234 (68%) | Core LB gap is data coverage |
| v5j unseen species mean max-prob | 0.036 | Model confidently predicts "no" for unseen |
| v5j pseudo-label coverage @0.5 | 80 / 159 | Half get some labels, half get nothing |
| 4-fold avg vs single coverage | -12 species | Use max-across-folds for pseudo-labeling |
| Focal "confusions" that are secondary labels | 12.6% | Model is more multi-label capable than metrics show |
| Focal count vs coverage (Spearman) | r=0.459 | More data helps, but domain shift is the wall |
| High focal + zero soundscape suspects | 54 species | Need ecological sanity check |

### Phase A: Max-Across-Folds

*(pending)*

### Phase B-C: Perch Inference + Probes

*(pending)*

### Phase D: Perch vs SED Comparison

*(pending)*

### Phase E: Ecological Sanity Check

*(pending)*

---

## Synthesis & Recommendations

*(to be written after all phases complete)*

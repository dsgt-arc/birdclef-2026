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

### Phase A: Max-Across-Folds (2026-04-09) ✅

Max-across-folds (`max(fold0, fold1, fold2, fold3)`) dramatically outperforms both
single-fold and averaged-folds for pseudo-label coverage of unseen species.

| Strategy | Coverage @>0.5 | Zero Coverage @>0.5 | Zero @>0.3 |
|----------|---------------|--------------------|-----------:|
| Single-fold | 80 / 159 | 79 | 55 |
| Avg-4fold | 68 / 159 | 91 (worse!) | — |
| **Max-4fold** | **109 / 159** | **50** | **34** |

Key findings:
- **+29 species recovered** at >0.5 vs single-fold (80 → 109)
- **34 species newly covered** that single-fold missed entirely
- Several "data quality suspects" (House Sparrow, Swallow-tailed Manakin) recovered —
  they DO exist in soundscapes but are too rare for any single fold
- Each fold contributes 2-6 unique species; all folds needed
- Only **34 truly hard species** remain at >0.3 — targets for Perch teacher
- Max-folds produces 223K predictions at >0.5 (vs 155K single) — larger pseudo-label pool

**Decision**: Use max-across-folds as the SED component of the pseudo-labeling teacher.
Apply P^1.5 power transform after max to control false positive rate.

Full analysis: `notes/2026-04-09-inference-eda-deep-dive.md` Section 7

### Phase B-C: Perch Inference + Probes

*(pending)*

### Phase D: Perch vs SED Comparison

*(pending)*

### Phase E: Ecological Sanity Check

*(pending)*

---

## Synthesis & Recommendations

*(to be written after all phases complete)*

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

### Phase B: Perch v2 TF Inference (2026-04-10) ✅

Full Perch v2 TF inference on both data sources. See tasks.md Phase B for details.

| Source | Rows | Embeddings | Logits |
|--------|------|------------|--------|
| train_soundscapes | 127,896 (10,658 × 12) | 1.19 GB | 11.3 GB |
| train_audio | 265,924 (35,549 files) | 2.4 GB | 23.2 GB |

### Phase C: Apply MLP Probes (2026-04-10) ✅

Applied full Perch→234 pipeline from exp 023 submission kernel to Phase B embeddings.
Script: `user/murilo/scripts/apply_perch_probes.py`

**Key discovery**: MLP probes require 142 features (128 PCA + 14 contextual), not just
PCA embeddings. Full pipeline: logit mapping → prior fusion → temporal smoothing → probes.

**Validation**: Phase B TF embeddings are **identical** to exp 020 probe training embeddings
(cosine similarity = 1.000000). No retraining needed.

| Output | Description | Size |
|--------|-------------|------|
| `perch_234_raw_probs.parquet` | Pure Perch (mapped logits + genus proxy) | 158 MB |
| `perch_234_scores.parquet` | Full pipeline (logit-space) | 172 MB |
| `perch_234_probs.parquet` | Full pipeline (prob-space) | 171 MB |

**Species mapping**: 203 direct, 3 genus proxy, 28 unmapped (sonotypes)

| Metric | Raw Perch | Full Pipeline |
|--------|-----------|---------------|
| Species with max prob >0.5 | 195 / 234 | 212 / 234 |
| Species with max prob >0.3 | 199 / 234 | 220 / 234 |
| Species with max prob >0.9 | 189 / 234 | 201 / 234 |
| True zeros (<0.001) | 28 | 6 |

**Pipeline helps most**: sonotype species (prior tables rescue from 0.0 to >0.95).
**Pipeline can hurt**: unmapped inactive species get suppressed to -8.0, discarding valid
genus proxy signal. Use raw_probs for fair detection comparison in Phase D.

### Phase D: Perch vs SED Comparison

*(pending)*

### Phase E: Ecological Sanity Check

*(pending)*

---

## Synthesis & Recommendations

*(to be written after all phases complete)*

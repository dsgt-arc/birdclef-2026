# Experiment 025: Iterative Pseudo-Labeling — Results

## Phase B: Row-Addition (FAILED)

Added pseudo-labeled samples as separate dataset rows with per-sample loss weighting.
50/50 GT:PL batch composition, loss_weight=0.5 for PL samples.

| Model | Loss | MixUp | val/auroc | val/cmap | LB | Delta vs v5j |
|-------|------|-------|-----------|----------|-----|-------------|
| v5j baseline (exp 019) | BCE | interp spec | 0.874 | — | **0.906** | — |
| v5j-pl-v1 (fold 0) | BCE | interp spec | 0.872 | 0.620 | **0.897** | **-0.009** |

**Conclusion**: PLs as separate rows regressed -0.009 on LB. Hypothesis: noisy Perch
Tier C labels (148K from 6%-precision model) and/or gradient balance between GT and PL
samples disrupted the learned v5j representation.

---

## Phase C: Mario-Style Waveform Mixing (IN PROGRESS)

Mix pseudo-labeled soundscape audio INTO focal training waveforms at the waveform level.
30% of samples get overlaid with a random PL segment (amplitude-controlled).
Labels: max(focal_label, pl_soft_labels). Standard 3-tuple — v5j pipeline unchanged.

Reference: Mario Lasseck, BirdCLEF 2024 2nd place (+8% from this technique).

### v3-waveform-mix: Round 1 (90/10 split, 20 epochs)

**Config**: `configs/v3_waveform_mix.yaml`
**SLURM**: Job 6492104, RTX 6000, 39m09s walltime
**Checkpoint**: epoch 16 (best by val/auroc = 0.866)

| Metric | Value |
|--------|-------|
| val/auroc (focal) | 0.862 (epoch 19) / **0.866** (best, epoch 16) |
| val/cmap | 0.603 |
| val/f1_macro | 0.524 |
| val/top1_acc | 0.751 |
| val/top5_acc | 0.862 |
| val/loss | 0.012 |
| LB | pending |

#### val/auroc progression

| Epoch | val/auroc | Note |
|-------|-----------|------|
| 0 | 0.505 | Random |
| 3 | 0.822 | Rapid improvement |
| 5 | 0.844 | MixUp warmup ends |
| 10 | 0.862 | +0.018 jump |
| 14 | 0.864 | Still improving |
| 15 | 0.865 | Still improving |
| **16** | **0.866** | **Best — checkpoint saved** |
| 17-19 | <0.866 | Slight plateau but model still training |

**Observation**: Model was still improving at epoch 20 (new bests at epochs 14, 15, 16).
The slow but steady improvement from epoch 10→16 (+0.004) suggests more epochs could help.
Next run should use max_epochs=30 or 40.

#### Comparison with baselines

| Model | PL Method | val/auroc | val/cmap | LB |
|-------|-----------|-----------|----------|-----|
| v5j baseline (exp 019) | None | 0.874 | — | **0.906** |
| v5j-pl-v1 Phase B | Row-addition (50/50, w=0.5) | 0.872 | 0.620 | 0.897 |
| **v3-waveform-mix** | **Waveform mixing (30%, amp-ctrl)** | **0.866** | **0.603** | **pending** |

**Note**: val/auroc is -0.008 vs v5j baseline (0.866 vs 0.874). However, offline metrics
have been **anti-correlated with LB** across 19+ experiments (FINDINGS.md insight #2).
Lower offline metrics with training changes that alter what data the model sees (not how
it trains) have NOT been tested before — this is the first data-centric training change.
The LB score is the only meaningful comparison.

**Note**: val_sc/auroc not available — `dual_validation: true` was not set in the config.
Will add for next run.

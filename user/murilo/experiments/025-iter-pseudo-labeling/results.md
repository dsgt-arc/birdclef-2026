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

### v3-waveform-mix: Round 1b (90/10 split, 40 epochs, dual validation)

**Config**: `configs/v3_waveform_mix.yaml` (updated: 40ep, dual_validation=true)
**SLURM**: Job 6494042, RTX 6000, 1h14m walltime
**Checkpoint**: epoch 38 (best by val/auroc = 0.851)

| Metric | Value |
|--------|-------|
| val/auroc (focal) | 0.845 (epoch 39) / **0.851** (best, epoch 38) |
| val/cmap | 0.612 |
| val/f1_macro | 0.561 |
| val/top1_acc | 0.775 |
| val/top5_acc | 0.870 |
| val/loss | 0.011 |
| **val_sc/auroc** | **0.991** |
| LB | **0.894** ❌ (-0.012 vs v5j baseline) |

#### val/auroc progression

| Epoch | val/auroc | Note |
|-------|-----------|------|
| 0 | 0.481 | Random |
| 3 | 0.801 | Rapid improvement |
| 8 | 0.827 | |
| 10 | 0.835 | |
| 14 | 0.843 | |
| 21 | 0.846 | Still improving past epoch 20 |
| 22 | 0.847 | |
| 24 | 0.848 | |
| 26 | 0.848 | |
| 28 | 0.849 | |
| 35 | 0.850 | |
| 36 | 0.850 | |
| **38** | **0.851** | **Best — checkpoint saved. Still improving.** |

**Observation**: Model set new val/auroc bests throughout training — never fully plateaued.
Improvements from epoch 20→38: +0.005 (0.846→0.851). Slow but steady. Even more epochs
could help, but should submit to LB first to confirm waveform mixing works before investing
more compute.

**val_sc/auroc = 0.991** — extremely high, but this is on 86 soundscape segments with only
~35 species having positives. Not directly comparable to LB. Still, the model clearly
learned soundscape patterns well.

#### Comparison: 20ep vs 40ep

| Run | Epochs | val/auroc | val/cmap | val/f1_macro | val/top1_acc | val_sc/auroc |
|-----|--------|-----------|----------|-------------|-------------|-------------|
| 20ep (job 6492104) | 20 | 0.866 | 0.603 | 0.524 | 0.751 | — |
| 40ep (job 6494042) | 40 | 0.851 | 0.612 | 0.561 | 0.775 | 0.991 |

Note: different random initialization between runs. The 40ep run has lower val/auroc but
higher val/cmap (+0.009), val/f1_macro (+0.037), and val/top1_acc (+0.024). This suggests
the 40ep model is better calibrated despite the lower AUROC number — different init +
OneCycleLR schedule over 40ep produces different optimization trajectory.

#### Comparison with baselines (updated)

| Model | PL Method | val/auroc | val/cmap | val_sc/auroc | LB |
|-------|-----------|-----------|----------|-------------|-----|
| v5j baseline (exp 019) | None | 0.874 | — | — | **0.906** |
| v5j-pl-v1 Phase B | Row-addition (50/50, w=0.5) | 0.872 | 0.620 | — | 0.897 |
| v3-waveform-mix 20ep | Waveform mixing (30%, amp) | 0.866 | 0.603 | — | pending |
| **v3-waveform-mix 40ep** | **Waveform mixing (30%, amp)** | **0.851** | **0.612** | **0.991** | **0.894** |

**Result: REGRESSED -0.012 vs v5j baseline (0.906).** Worse than Phase B row-addition (0.897).
Both pseudo-label approaches have now failed to improve LB.

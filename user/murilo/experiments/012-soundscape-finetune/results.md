# Results: Labeled Soundscape Fine-Tuning

**Date Completed**: 2026-03-22
**Author**: Murilo Gustineli

## Summary

Soundscape fine-tuning confirmed: all 3 models improved on the Kaggle leaderboard by expanding
species coverage from 206 → 234 and adapting to the soundscape domain. ConvNeXt showed the
largest gain (+0.060), followed by ProtoCLR (+0.031) and EAT (+0.022). ProtoCLR remains the
best single-model at **0.767 LB**.

**Verdict**: confirmed

## Kaggle Leaderboard Results

Competition metric: **macro-averaged ROC-AUC** (skips classes with no true positive labels).

### Exp 012: Soundscape Fine-Tuned Models (234 classes)

| Model | Training Data | Species | Public LB Score | vs Exp 011 Fulldata | Date |
|-------|--------------|---------|----------------:|--------------------:|------|
| ProtoCLR LoRA attn | Focal + soundscape | 234/234 | **0.767** | +0.031 | 2026-03-21 |
| ConvNeXt LoRA MLP+MLP head | Focal + soundscape | 234/234 | **0.755** | +0.060 | 2026-03-22 |
| EAT LoRA attn+MLP head | Focal + soundscape | 234/234 | **0.720** | +0.022 | 2026-03-22 |

### Exp 011 Baselines (for comparison)

| Model | Training Data | Species | Public LB Score |
|-------|--------------|---------|----------------:|
| ProtoCLR fulldata | Focal only | 206/234 | **0.736** |
| EAT 80/10/10 | Focal only | 206/234 | **0.701** |
| EAT fulldata | Focal only | 206/234 | **0.698** |
| ConvNeXt 80/10/10 | Focal only | 206/234 | **0.720** |
| ConvNeXt fulldata | Focal only | 206/234 | **0.695** |

## Training Results (offline test split, 234 classes)

All models fine-tuned from exp 011 best checkpoint (80/10/10 split) with LoRA rank=8,
alpha=16, lr=1e-4, effective batch size=256, early stopping patience=8 on val/auroc.

| Run | Encoder | AUROC | cmAP | F1 macro | F1 micro | Top-1 | Top-5 | Best Epoch | Stopped |
|-----|---------|-------|------|----------|----------|-------|-------|------------|---------|
| `protoclr-soundscape` | ProtoCLR/CvT-13 | **0.845** | **0.481** | **0.447** | **0.562** | **0.678** | **0.809** | 11 | 13 |
| `convnext-soundscape` | ConvNeXt-Base | 0.844 | 0.470 | 0.432 | 0.553 | 0.663 | 0.796 | 2 | 11 |
| `eat-soundscape` | EAT-base | 0.838 | 0.452 | 0.428 | 0.541 | 0.648 | 0.795 | 3 | 12 |

**Note**: Test AUROC (0.838–0.845) is not directly comparable to exp 011 (0.929–0.938) because
exp 012 evaluates on 234 classes including 28 barely-trained soundscape-only species, while
exp 011 evaluated on 206 focal species only.

### Convergence Speed

All models converge much faster than exp 011 due to warm-starting from pretrained checkpoints
at a lower LR (1e-4 vs 3e-4):

| Model | Exp 011 Best Epoch | Exp 012 Best Epoch | Speedup |
|-------|-------------------|-------------------|---------|
| ConvNeXt | 9 | 2 | 4.5× |
| EAT | 17 | 3 | 5.7× |
| ProtoCLR | 21 | 11 | 1.9× |

## Observations

- **All 3 models improved on LB**: Soundscape fine-tuning is universally beneficial.
  The 206→234 species expansion and soundscape domain adaptation both contribute to the
  improvement. Every model gained despite training on only 739 labeled soundscape segments.

- **ConvNeXt had the largest LB gain (+0.060)**: ConvNeXt was most hurt by fulldata
  overfitting in exp 011 (0.720 → 0.695, −0.025). Soundscape fine-tuning starting from
  the non-overfitted 80/10/10 checkpoint recovers and surpasses the original score.
  This also suggests ConvNeXt had the biggest focal→soundscape domain gap.

- **ProtoCLR is the best single model (0.767)**: Consistent with exp 011 — ProtoCLR's
  Xeno-Canto pretraining gives strong bird-specific representations. It's also the
  smallest model (~20M params) and cheapest to run on Kaggle.

- **EAT converged very fast (best epoch 3)**: The lower LR (1e-4) combined with warm
  start means EAT adapts in just 3 epochs. This fast convergence leaves little room
  for the soundscape data to further refine representations, which may explain its
  smaller LB gain (+0.022).

- **ConvNeXt converged fastest (best epoch 2)**: Even faster than EAT, yet still achieved
  the largest LB improvement. ConvNeXt's MLP-targeted LoRA adapts very efficiently.

- **Offline test AUROC does not predict LB ranking**: ConvNeXt (0.844 offline) beats EAT
  (0.838 offline) by 0.006, but the LB gap is much larger (0.755 vs 0.720 = 0.035).
  The real evaluation on diverse soundscapes amplifies differences that the test split
  underestimates.

- **Exp 012 uses 80/10/10, not fulldata**: Unlike exp 011 fulldata runs, exp 012 fine-tunes
  from the 80/10/10 checkpoint and keeps the val/test split for early stopping. A fulldata
  soundscape run could yield further gains but risks the same overfitting seen in exp 011.

## Analysis

The hypothesis is confirmed: labeled soundscape fine-tuning improves all 3 models on the
Kaggle leaderboard. The two-factor improvement (species expansion + domain adaptation)
validates the exp 012→013 sequencing decision — these soundscape-adapted models will be
stronger teachers for pseudo-labeling in exp 013.

**Key findings for next steps:**

1. **ProtoCLR-soundscape (0.767) is the best teacher for pseudo-labeling**: Highest LB score,
   smallest model, and broadest species coverage. Should anchor the exp 013 ensemble.

2. **ConvNeXt-soundscape (0.755) adds ensemble diversity**: CNN architecture (vs ProtoCLR's
   conv-transformer and EAT's pure transformer) and ImageNet pretraining provide complementary
   error profiles for teacher averaging.

3. **EAT-soundscape (0.720) may be the weakest ensemble member**: Smallest LB gain, lowest
   absolute score. Still worth including for architectural diversity (pure transformer +
   AudioSet pretraining) but could be dropped if inference budget is tight.

4. **Fulldata soundscape fine-tuning is a potential follow-up**: If exp 013 pseudo-labeling
   doesn't yield sufficient gains, retraining exp 012 on 100% focal data (no val/test split)
   with capped epochs could push scores higher.

### Confounders

- **Single seed (42)**: Rankings could shift with different seeds, especially between
  ConvNeXt (0.755) and EAT (0.720) where the gap is meaningful but still within
  seed-to-seed variance.

- **No hyperparameter tuning**: All models use the same LR (1e-4), patience (8), and
  LoRA config. Per-model tuning (especially for EAT which converges in 3 epochs) could
  improve results.

- **Public LB uses ~34% of test data**: Final rankings on the remaining ~66% may differ.

- **28 soundscape-only species are barely trained**: Only 739 soundscape segments contribute
  to learning these species, and they appear in dense multi-label contexts (avg 4.2
  species/segment). Their actual per-class performance is unknown.

## Artifacts

| Type | Location |
|------|----------|
| Checkpoints | `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/012-soundscape-finetune/checkpoints/` |
| Metrics (ProtoCLR) | `artifacts/training/protoclr_soundscape/metrics.json` |
| Metrics (EAT) | `artifacts/training/eat_soundscape/metrics.json` |
| Test results (ConvNeXt) | `outputs/test_results/2026-convnext-soundscape.json` |
| W&B | Project `dsgt-arc/birdclef-2026-soundscape-finetune` |
| SLURM logs | `logs/slurm_{eat,convnext,protoclr}_soundscape_*.log` |
| Code | Branch `mg/012-soundscape-finetune`, PR #17 |

## Next Steps

- [ ] Exp 013: Use these 3 models as teacher ensemble for pseudo-labeling 10,592 unlabeled soundscapes
- [ ] Check focal test AUROC vs exp 011 baseline (catastrophic forgetting check)
- [ ] Check 28 soundscape-only species have non-zero predictions
- [ ] Consider fulldata soundscape run if exp 013 gains are insufficient

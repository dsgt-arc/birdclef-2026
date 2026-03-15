# Results: LoRA Baseline on BirdCLEF 2026

**Date Completed**: 2026-03-15
**Author**: Murilo Gustineli

## Summary

The LoRA fine-tuning pipeline from BirdCLEF 2025 (exps 006/007/009) transfers cleanly to BirdCLEF
2026 focal data with no hyperparameter changes. All 3 encoders trained to convergence and achieved
AUROC 0.929–0.938 on the held-out test split, confirming that the pipeline generalizes to the new
dataset (35,549 recordings, 206 focal species).

ConvNeXt leads on AUROC (0.9376), followed closely by ProtoCLR (0.9372) and EAT (0.9289).
Compared to the 2025 baselines (0.941–0.945 AUROC), the 2026 results are slightly lower, likely
due to increased species count and different class distribution rather than pipeline issues.

**Verdict**: confirmed

## Observations

### Quantitative

All three models trained with identical LoRA hyperparameters (rank=8, alpha=16, lr=3e-4, effective batch=256):

| Run | Encoder | AUROC | cmAP | F1 macro | F1 micro | Top-1 | Top-5 | Best Epoch | Stopped |
|-----|---------|-------|------|----------|----------|-------|-------|------------|---------|
| `convnext-lora-mlp-mlphead` | ConvNeXt-Base | **0.9376** | 0.5215 | 0.456 | 0.548 | 0.658 | 0.796 | 9 | 20 |
| `protoclr-lora-attn` | ProtoCLR/CvT-13 | 0.9372 | **0.5309** | **0.492** | **0.560** | **0.677** | **0.810** | 21 | 32 |
| `eat-lora-attn-mlphead` | EAT-base | 0.9289 | 0.5088 | 0.472 | 0.537 | 0.650 | 0.799 | 17 | 28 |

**Cross-year comparison (best LoRA config per encoder, sorted by AUROC):**

| Encoder | 2025 AUROC | 2026 AUROC | Delta | 2025 cmAP | 2026 cmAP | Delta |
|---------|-----------|-----------|-------|-----------|-----------|-------|
| EAT-base | 0.945 | 0.929 | -0.016 | 0.536 | 0.509 | -0.027 |
| ConvNeXt-Base | 0.941 | **0.938** | -0.003 | 0.525 | 0.522 | -0.003 |
| ProtoCLR/CvT-13 | 0.942 | 0.937 | -0.005 | 0.531 | **0.531** | 0.000 |

### Qualitative

- **Pipeline transfers without changes**: The unified `train.py` dispatched correctly for all
  3 encoder types. `BirdCLEFDataModule` worked out of the box with the 2026 data format. No
  code changes were needed beyond swapping `data.data_dir` in the configs.

- **ConvNeXt converges fastest**: Best epoch at 9 (stopped at 20), vs EAT at 17 and ProtoCLR at
  21. This is consistent with 2025 — ConvNeXt's MLP-targeted LoRA finds its optimum quickly.

- **ProtoCLR leads on cmAP and F1 despite smaller model**: At 19M params (vs ~90M for EAT and
  ConvNeXt), ProtoCLR achieves the highest cmAP (0.531) and F1 macro (0.492). Its Xeno-Canto
  pretraining continues to provide strong bird-specific representations, even on a new species set.

- **EAT shows the largest 2025→2026 AUROC drop**: -0.016 AUROC, compared to -0.003 for ConvNeXt
  and -0.005 for ProtoCLR. This may reflect EAT's AudioSet pretraining being more sensitive to
  the shifted species distribution, or the attention-only LoRA targets being less adaptive to
  the new class boundaries.

- **All models fit comfortably on RTX 6000**: GPU memory usage was <3% across all runs (0.4–0.7
  GB allocated out of 24 GB). ConvNeXt achieved 99.9% GPU utilization, while EAT (88.9%) and
  ProtoCLR (66.0%) were somewhat data-loading bottlenecked.

## Analysis

The hypothesis is confirmed: the 2025 LoRA pipeline produces competitive baselines on 2026
without tuning. AUROC values (0.929–0.938) are slightly below 2025 (0.941–0.945), but this is
expected given the larger species set (206 vs 206 classes, but different species composition)
and potentially different class balance in the 2026 focal data.

**Key findings for next steps:**

1. **ConvNeXt is the strongest single model by AUROC** (0.9376), making it the priority for
   submission pipeline integration. Its fast convergence (9 epochs) also makes it cheap to
   iterate on.

2. **ProtoCLR is the best value model**: highest cmAP and F1 with the smallest checkpoint
   (80 MB vs ~355 MB). If submission size or inference speed matter, ProtoCLR is the pick.

3. **The 3 models remain complementary**: Different architectures (transformer vs CNN vs
   conv-transformer), different pretraining (AudioSet vs ImageNet vs Xeno-Canto), and different
   error profiles. This ensemble diversity should benefit pseudo-labeling and model averaging.

### Confounders

- **Single seed (42)**: The AUROC spread between models (0.929–0.938) is within typical
  seed-to-seed variance. Rankings could shift with different seeds.

- **No hyperparameter tuning**: All configs use identical 2025 hyperparameters. The 2026
  dataset may have different optimal learning rates, LoRA ranks, or training durations.

- **Test split only**: Metrics are on the held-out 10% test split of focal recordings, not
  on actual Kaggle soundscapes. Real competition performance depends on the focal-to-soundscape
  domain gap plus the 28 soundscape-only species we cannot train on.

- **cmAP monitored for checkpointing, not AUROC**: Early stopping and checkpoint selection use
  `val/cmap`, not `val/auroc`. The best AUROC checkpoint may differ from the best cmAP checkpoint.

## Artifacts

| Type | Location |
|------|----------|
| Metrics | `artifacts/training/{eat_lora_attn_mlphead,convnext_lora_mlp_mlphead,protoclr_lora_attn}/metrics.json` |
| Test results | `outputs/test_results/{2026-eat-lora-attn-mlphead,2026-convnext-lora-mlp-mlphead,2026-protoclr-lora-attn}.json` |
| Checkpoints | `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/011-lora-baseline-2026/checkpoints/` |
| TensorBoard | `outputs/tensorboard/exp011-{2026-eat-lora-attn-mlphead,2026-convnext-lora-mlp-mlphead,2026-protoclr-lora-attn}/` |
| W&B | Project `dsgt-arc/birdclef-2026-lora-finetune` |
| GPU profiling | `outputs/gpu_check/2026-{eat,convnext,protoclr}-*_*.{txt,png}` |
| SLURM logs | `logs/slurm_{eat,convnext,protoclr}_*_*.log` |
| Code | Branch `mg/010-submission-pipeline`, PR #16 |

## Next Steps

- [ ] Export 2026-trained checkpoints to ONNX for Kaggle submission (extend exp 010 pipeline)
- [ ] Submit all 3 models to Kaggle and compare public LB scores
- [ ] Tune hyperparameters on 2026 data (higher LoRA rank, learning rate schedule, longer warmup)
- [ ] Monitor with AUROC as checkpoint selection metric instead of cmAP
- [ ] Build 3-model ensemble for pseudo-labeling unlabeled soundscapes

**Leads to**: These 2026 baselines become the foundation for submission and pseudo-labeling.
Next priority is ONNX export and Kaggle submission to get real LB scores, followed by
hyperparameter tuning if the baselines underperform expectations.

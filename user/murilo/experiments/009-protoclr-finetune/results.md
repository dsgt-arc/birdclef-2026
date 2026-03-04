# Results: ProtoCLR (CvT-13) Fine-Tuning on BirdCLEF 2025

**Date Completed**: 2026-03-03
**Author**: Murilo Gustineli

## Summary

ProtoCLR with LoRA fine-tuning reaches AUROC=0.942 (attn-only config), matching EAT's best
(0.945) and ConvNeXt (0.941) despite having only 19M parameters (~5x smaller).
Domain-invariant prototypical contrastive pretraining on Xeno-Canto transfers remarkably well
to BirdCLEF, confirming ProtoCLR as a viable teacher model for pseudo-labeling.

Full fine-tuning (all 19M params) underperformed LoRA (AUROC=0.925 vs 0.942), confirming that
LoRA's low-rank constraint acts as critical regularization when the pretrained representations
are already well-aligned with the target task.

**Verdict**: confirmed

## Observations

### Quantitative

All three configs (two LoRA, one full fine-tune) vs. Exp 006 (EAT) and Exp 007 (ConvNeXt) baselines:

| Run | FT Mode | Trainable Params | Head | AUROC | cmAP | F1 macro | F1 micro | Top-1 | Top-5 | Best Epoch | Stopped | Walltime |
|-----|---------|-----------------|------|-------|------|----------|----------|-------|-------|------------|---------|----------|
| `protoclr-lora-attn` | LoRA (Q/K/V/proj) | ~148K | Linear | **0.942** | 0.531 | 0.463 | 0.581 | 0.659 | 0.808 | 20 | 31 | 2h 41m |
| `protoclr-lora-all` | LoRA (Q/K/V/proj + MLP) | ~247K | Linear | 0.939 | 0.537 | 0.504 | 0.592 | 0.668 | 0.803 | 28 | 30 | 3h 24m |
| `protoclr-full-finetune` | Full (all params) | 19M | Linear | 0.925 | 0.525 | 0.502 | 0.584 | 0.651 | 0.782 | 47 | 50 | ~4h 30m |

**Cross-experiment comparison (best configs, sorted by AUROC):**

| Method | Encoder | Params | Pretraining | AUROC | cmAP | Δ AUROC vs ProtoCLR best |
|--------|---------|--------|-------------|-------|------|--------------------------|
| EAT LoRA attn+MLP head (adapters) | EAT-base | ~90M | AudioSet | **0.945** | 0.536 | +0.003 |
| **ProtoCLR LoRA attn (PEFT)** | **CvT-13** | **19M** | **Xeno-Canto (ProtoCLR)** | **0.942** | **0.531** | **—** |
| EAT LoRA attn+MLP head (PEFT) | EAT-base | ~90M | AudioSet | 0.942 | 0.511 | 0.000 |
| ConvNeXt LoRA MLP+MLP head (PEFT) | ConvNeXt-Base | ~89M | ImageNet-22K+1K | 0.941 | 0.525 | -0.001 |
| ProtoCLR LoRA all (PEFT) | CvT-13 | 19M | Xeno-Canto (ProtoCLR) | 0.939 | 0.537 | -0.003 |
| ProtoCLR full fine-tune | CvT-13 | 19M | Xeno-Canto (ProtoCLR) | 0.925 | 0.525 | -0.017 |
| Frozen linear probe (Exp 005) | EAT-base | ~90M | AudioSet | 0.890 | 0.301 | -0.052 |

### Qualitative

- **Domain-invariant pretraining is highly effective**: ProtoCLR, pretrained on Xeno-Canto with
  prototypical contrastive learning (~685K segments, 10K species), matches models pretrained on
  much larger general audio (AudioSet) or vision (ImageNet-22K) datasets. The Xeno-Canto
  pretraining directly targets the focal-to-soundscape domain shift that BirdCLEF tests.

- **Remarkable parameter efficiency**: At 19M parameters, ProtoCLR is ~5x smaller than EAT
  (~90M) and ConvNeXt (~89M), yet achieves equivalent AUROC (0.942 vs 0.945 and 0.941). The
  CvT-13 architecture with convolutional tokenization and progressive downsampling is compact
  but expressive.

- **Attention-only LoRA slightly better by AUROC; all-layers better by F1**: The attn-only
  config achieves higher AUROC (0.942 vs 0.939) while the all-layers config achieves higher
  F1 macro (0.504 vs 0.463, +8.8%). Adding MLP targets helps discriminate underrepresented
  classes at some cost to overall ranking quality.

- **Both configs converge smoothly**: lora_attn peaks at epoch 20 (stopped 31), lora_all peaks
  at epoch 28 (stopped 30). Early stopping patience of 10 was well-calibrated — neither run hit
  the 50-epoch budget.

- **Training is CPU-bound**: GPU profiling showed the CPU mel spectrogram transform is the
  bottleneck (data loading spikes up to 4651ms). Throughput plateaued at ~500 samples/s
  regardless of batch size (bs=128 vs bs=512 yielded 503 vs 511 samples/s). Despite this,
  training is fast (2.7-3.4h) because the model is small. EAT required ~9-11h and ConvNeXt
  required ~5-7h for comparable runs.

- **Train-test gap indicates some overfitting**: Both LoRA configs show a gap between training
  cmAP (0.849 / 0.895) and test cmAP (0.531 / 0.537). This is consistent with Exp 006 and 007
  and expected for the BirdCLEF domain shift between clean training data and noisier evaluation.

- **Full fine-tuning degrades performance**: Training all 19M parameters with lr=1e-4 (3× lower
  than LoRA) yields AUROC=0.925, a -1.7% drop from LoRA attn (0.942). The model ran all 50
  epochs without triggering early stopping (best val cmAP=0.475 at epoch 47), indicating it
  continued fitting training data without improving generalization. This is catastrophic
  forgetting: the pretrained Xeno-Canto representations are overwritten when all parameters
  are unconstrained.

## Analysis

The hypothesis is confirmed: ProtoCLR provides viable, complementary features for BirdCLEF
teacher model ensembles. The result is stronger than expected — ProtoCLR achieves AUROC=0.942,
nearly matching EAT's best (0.945) despite being 5x smaller.

**Why does ProtoCLR punch above its weight?**

1. **Task-aligned pretraining**: ProtoCLR was trained on Xeno-Canto bird recordings with
   prototypical contrastive learning designed for domain-invariant representations. This is
   precisely the focal-to-soundscape transfer problem in BirdCLEF. AudioSet and ImageNet
   pretraining are more general but less targeted.

2. **CvT-13 architecture advantages**: Convolutional tokenization (no fixed patch size) with
   progressive downsampling captures multi-scale spectro-temporal patterns efficiently. The
   absence of positional embeddings makes it naturally position-agnostic, useful for
   variable-length bird vocalizations.

3. **384-dim embeddings are sufficient**: Despite smaller embeddings (384 vs 768 for EAT,
   1024 for ConvNeXt), the representations are rich enough for 206-class classification.
   This suggests the bottleneck in prior experiments was not embedding dimensionality.

**The MLP LoRA effect**: Adding LoRA to MLP layers (fc1, fc2) improves F1 macro significantly
(+8.8%) but slightly lowers AUROC (0.939 vs 0.942). This suggests the MLP LoRA helps adapt
per-class decision boundaries, improving tail-class performance at a slight cost to ranking
quality. For the Kaggle competition (AUROC metric), the attention-only config is preferable.

**LoRA vs full fine-tuning**: Full FT underperforms LoRA by -1.7% AUROC (0.925 vs 0.942)
despite having 130× more trainable parameters (19M vs 148K). Three factors explain this:

1. **Catastrophic forgetting**: ProtoCLR's Xeno-Canto pretraining is already well-aligned with
   BirdCLEF (bird focal recordings → bird soundscapes). With all 19M params trainable, the
   model overwrites these domain-invariant representations to fit the 206-class training set.
   LoRA's low-rank constraint (rank=8) preserves the backbone while making targeted adaptations.

2. **Implicit regularization**: LoRA's rank-8 bottleneck limits each weight update to an 8-dim
   subspace, acting as strong regularization. With ~135K BirdCLEF recordings across 206 classes,
   this constraint prevents overfitting. Full FT's 19M free parameters are excessive for this
   dataset size.

3. **Training dynamics**: LoRA attn converges cleanly (best epoch 20, stopped 31), while full FT
   never plateaued (best epoch 47, ran all 50 epochs, best val cmAP=0.475 vs LoRA's 0.531).
   The model kept fitting without improving generalization — a hallmark of forgetting.

This result is consistent with recent PEFT literature: LoRA tends to match or outperform full FT
when pretrained representations are task-aligned and downstream data is moderate in size.

**Ensemble value**: ProtoCLR, EAT, and ConvNeXt use fundamentally different architectures and
pretraining objectives:
- **ProtoCLR**: CvT-13 (conv tokenization), bird-specific contrastive pretraining
- **EAT**: Transformer (patch tokenization), general audio pretraining
- **ConvNeXt**: Pure CNN, vision pretraining on spectrograms

This architectural and pretraining diversity should produce complementary errors, making all
three strong candidates for a teacher ensemble.

### Confounders

- **Single seed**: All runs use seed=42. The 0.003 AUROC gap between ProtoCLR (0.942) and EAT
  (0.945) is within typical seed-to-seed variance. The key finding is parity, not superiority.

- **Different LoRA backends**: ProtoCLR uses PEFT while EAT's best config uses the adapters
  library. Exp 006 showed adapters gives ~0.01-0.025 cmAP improvement over PEFT on EAT,
  so ProtoCLR with adapters might push even higher.

- **No MLP head tested**: Exp 006 showed a +0.03-0.05 cmAP boost from an MLP classification
  head (768->512->206) on EAT. ProtoCLR was only tested with a linear head (384->206). An
  MLP head could further improve results.

- **CPU bottleneck**: The dataloader mel transform was the throughput bottleneck (~500
  samples/s). With GPU-based mel transforms (nnAudio), training would be significantly faster,
  though final metrics should be unaffected.

- **Batch size difference**: ProtoCLR used bs=256 (accum=1, effective=256) while EAT used
  bs=32 (accum=8, effective=256) and ConvNeXt used bs=128 (accum=2, effective=256). Same
  effective batch size across all experiments.

## Artifacts

| Type | Location |
|------|----------|
| Metrics | `artifacts/training/{protoclr_lora_attn,protoclr_lora_all,protoclr_full_finetune}/metrics.json` |
| Test results | `outputs/test_results/{protoclr-lora-attn,protoclr-lora-all,protoclr-full-finetune}.json` |
| Checkpoints | `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/009-protoclr-finetune/checkpoints/` |
| TensorBoard | `outputs/tensorboard/exp009-{protoclr-lora-attn,protoclr-lora-all,protoclr-full-finetune}/` |
| W&B | `outputs/wandb/wandb/run-2026030{2,3}-*/` |
| GPU profiling | `outputs/gpu_check/{protoclr-lora-attn,protoclr-full-finetune}_*.{txt,png}` |
| SLURM logs | `logs/slurm_protoclr_{lora_attn,lora_all,full_finetune}_*.log` |
| Code | Branch `mg/009-protoclr-finetune` |

## Next Steps

- [x] ~~Full fine-tuning run~~ — completed; underperformed LoRA (AUROC=0.925 vs 0.942), confirming LoRA is the right approach for ProtoCLR
- [ ] Test ProtoCLR with MLP classification head (384->256->206) to push beyond AUROC=0.942
- [ ] Test ProtoCLR with adapters backend instead of PEFT for apples-to-apples EAT comparison
- [ ] Build 3-model teacher ensemble (ProtoCLR + EAT + ConvNeXt) for pseudo-labeling
- [ ] Evaluate ensemble diversity: correlation between per-sample predictions across models

**Leads to**: ProtoCLR LoRA attn (AUROC=0.942) joins EAT (AUROC=0.945) and ConvNeXt
(AUROC=0.941) as the third teacher model. Full fine-tuning confirmed that LoRA is preferable
for ProtoCLR — the pretrained representations are too valuable to risk overwriting. The three
models cover distinct inductive biases (conv tokenization vs patch tokenization vs pure CNN) and
pretraining domains (bird audio vs general audio vs vision). Next priority is building the
teacher ensemble for pseudo-labeling unlabeled soundscapes.

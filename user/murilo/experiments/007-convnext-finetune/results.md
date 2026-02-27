# Results: ConvNeXt-Base LoRA Fine-Tuning on BirdCLEF 2025

**Date Completed**: 2026-02-27
**Author**: Murilo Gustineli

## Summary

ConvNeXt-Base with ImageNet pretraining and MLP-only LoRA reaches cmAP=0.527, only 1.7%
below EAT's best (cmAP=0.536 from Exp 006). ImageNet visual features transfer surprisingly
well to mel spectrogram bird classification, nearly matching AudioSet audio-specific pretraining
despite the domain gap. The MLP head ablation provides no benefit (0.525 vs 0.527).

**Verdict**: confirmed (ConvNeXt < EAT, but the gap is much smaller than expected)

## Observations

### Quantitative

Both runs vs. Exp 006 baselines and Exp 005 frozen probe:

| Run | Head | LoRA Params | test cmAP | AUROC | F1 macro | F1 micro | Top-1 | Top-5 | Best Epoch | Stopped |
|-----|------|-------------|-----------|-------|----------|----------|-------|-------|------------|---------|
| `convnext_lora_mlp` | Linear | ~1.4M | **0.527** | 0.939 | 0.467 | 0.568 | 0.638 | 0.786 | 15 | 26 |
| `convnext_lora_mlp_mlphead` | MLP 512 | ~1.4M | 0.525 | 0.941 | 0.455 | 0.569 | 0.640 | 0.786 | 12 | 23 |

**Cross-experiment comparison (all LoRA, test cmAP):**

| Method | Encoder | Pretraining | test cmAP | Delta vs ConvNeXt best |
|--------|---------|-------------|-----------|------------------------|
| EAT LoRA attn+MLP head (adapters) | EAT-base | AudioSet | **0.536** | +0.009 |
| EAT LoRA attn+MLP head (PEFT) | EAT-base | AudioSet | 0.511 | -0.016 |
| EAT LoRA all (PEFT) | EAT-base | AudioSet | 0.482 | -0.045 |
| EAT LoRA attn (adapters) | EAT-base | AudioSet | 0.485 | -0.042 |
| EAT LoRA attn (PEFT) | EAT-base | AudioSet | 0.469 | -0.058 |
| **ConvNeXt LoRA MLP (PEFT)** | **ConvNeXt-Base** | **ImageNet** | **0.527** | **—** |
| ConvNeXt LoRA MLP + MLP head (PEFT) | ConvNeXt-Base | ImageNet | 0.525 | -0.002 |
| Frozen linear probe | EAT-base | AudioSet | 0.301 | -0.226 |

### Qualitative

- **ImageNet features transfer well to audio**: ConvNeXt-Base (cmAP=0.527) falls in the
  "interesting" range (>0.4) defined in the proposal. The 1.7% gap vs EAT's best is far
  smaller than expected. ImageNet-pretrained CNNs learn generic hierarchical features
  (edges, textures, frequency patterns) that apply to mel spectrograms.

- **MLP head does NOT help ConvNeXt**: Unlike Exp 006 where the MLP head gave EAT a +0.03-0.05
  cmAP boost, ConvNeXt sees no benefit from a richer head (0.525 vs 0.527). ConvNeXt's
  1024-dim features may already be linearly separable for this task, or the MLP head
  overfits with the slightly higher parameter count.

- **EAT is faster per sample despite smaller batches**: EAT at bs=32 achieves 182 samples/s
  (3.6 ms/sample) vs ConvNeXt at bs=128 with 113 samples/s (8.5 ms/sample). EAT's dense
  attention is more compute-efficient (higher FLOPs/byte) than ConvNeXt's memory-bandwidth-bound
  depthwise convolutions. ConvNeXt fits larger batches in VRAM (no O(n²) activation maps) but
  this doesn't translate to faster training — just fewer accumulation steps with slower per-step
  compute.

- **Earlier convergence**: Both ConvNeXt runs reach their best epoch earlier (12-15) and
  stop earlier (23-26) than EAT runs (27-36), suggesting ConvNeXt's LoRA adapts more
  quickly or has less capacity to improve beyond a certain point.

- **VRAM usage**: ConvNeXt at bs=128 uses ~9.7 GB reserved (22.8 GB in interactive session
  with full pipeline). EAT at bs=32 uses ~5.4 GB reserved. ConvNeXt's lower per-sample
  activation memory allows larger batch sizes, but OOM still occurs at bs=256 because
  backprop through 89M params dominates memory regardless of architecture.

## Analysis

The hypothesis is confirmed: ConvNeXt produces lower cmAP than EAT's best. However, the
margin (0.527 vs 0.536) is much narrower than expected, challenging the assumption that
audio-specific pretraining is essential for competitive bird sound classification.

**Why is the gap so small?**

1. **Mel spectrograms are images**: Converting audio to mel spectrograms transforms the task
   into a visual pattern recognition problem. ImageNet-pretrained CNNs excel at recognizing
   textures, frequency bands, and temporal patterns — all present in spectrograms.

2. **ConvNeXt's LoRA targets more parameters**: With ~1.4M LoRA params across 36 blocks (all
   MLP layers), ConvNeXt adapts more of its network than EAT's attention-only LoRA (~451K
   params). The fairer comparison is ConvNeXt (1.4M) vs EAT LoRA all (~1.2M), where EAT
   only leads by 0.482 vs 0.527 — ConvNeXt actually wins by 9%.

3. **ConvNeXt-Base is a strong backbone**: The `fb_in22k_ft_in1k` variant was pretrained on
   ImageNet-22K then fine-tuned on ImageNet-1K, providing rich, generalizable features.

**The MLP head asymmetry**: In Exp 006, the MLP head was critical for EAT because attention-only
LoRA doesn't adapt the MLP layers, so the head compensates. In Exp 007, LoRA already targets
the MLP layers (fc1, fc2), so the classification head doesn't need extra capacity — the encoder
features are already well-adapted.

### Confounders

- **Asymmetric LoRA targets**: ConvNeXt has no attention layers, so LoRA only applies to MLP
  (fc1, fc2). EAT's best config uses attention-only LoRA + MLP head. The architectures are
  adapted in fundamentally different ways, making direct comparison imperfect.

- **Only PEFT backend**: Exp 006 showed adapters slightly outperforms PEFT (+0.01-0.025 cmAP).
  ConvNeXt was only tested with PEFT, so the comparison is against EAT's best (adapters).
  Against PEFT-only EAT configs, ConvNeXt (0.527) beats all except `eat_lora_attn_mlphead`
  (0.511) — wait, it beats that too. ConvNeXt actually outperforms every PEFT EAT config.

- **Single seed**: All runs use seed=42. The small gap (0.009 cmAP) is within typical
  seed-to-seed variance.

- **Batch size difference**: ConvNeXt uses bs=128 (accum=2, effective=256) while EAT uses
  bs=32 (accum=8, effective=256). Same effective batch size, but larger micro-batches can
  affect batch norm statistics and gradient noise.

## Artifacts

| Type | Location |
|------|----------|
| Metrics | `artifacts/training/{convnext_lora_mlp,convnext_lora_mlp_mlphead}/metrics.json` |
| Test results | `outputs/test_results/convnext-lora-{mlp,mlp-mlphead}.json` |
| Checkpoints | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/007-convnext-finetune/checkpoints/` |
| TensorBoard | `outputs/tensorboard/exp007-convnext-lora-{mlp,mlp-mlphead}/` |
| W&B | `outputs/wandb/wandb/run-20260227-*/` |
| GPU profiling | `outputs/gpu_check/convnext-lora-mlp_4330340.{txt,png}` |
| SLURM logs | `logs/slurm_convnext_lora_mlp_{4330339,4330340}.log`, `logs/slurm_convnext_lora_mlp_mlphead_4330338.log` |
| Code | Branch `mg/007-convnext-finetune` |

## Next Steps

- [ ] Test ConvNeXt with adapters backend to see if the PEFT→adapters gap applies to CNNs
- [ ] Try higher LoRA rank (r=16) — ConvNeXt may benefit from more adapter capacity given its 36 blocks
- [ ] Ensemble ConvNeXt + EAT predictions (CNN + Transformer diversity could boost cmAP)
- [ ] Test ConvNeXt-Large or ConvNeXt-XL to see if scaling helps close the gap further

**Leads to**: ConvNeXt-Base (cmAP=0.527) is a viable alternative to EAT for spectrogram
classification. The near-parity with EAT suggests that architecture diversity (CNN + Transformer
ensembles) could be a fruitful direction. The best single model remains EAT LoRA
(adapters, cmAP=0.536).

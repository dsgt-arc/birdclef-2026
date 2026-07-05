# Concept: Vision Transformer + Masked Autoencoding for Bioacoustics Pre-training

**Status**: deferred (as of 2026-02-01)
**Created**: 2026-01-31
**Updated**: 2026-02-01

## Idea

Leverage self-supervised pre-training via Masked Autoencoders (MAE) to learn robust audio representations from BirdSet spectrograms, enabling better generalization to rare species and soundscape audio. The approach treats mel-spectrograms as images, adapts Vision Transformers (ViT) for single-channel audio, and uses aggressive masking (75%) to force semantic understanding of bird vocalizations.

**Key insight**: Rather than relying solely on supervised learning (which struggles with long-tail species distribution in BirdSet), we can leverage the abundance of unlabeled audio through self-supervised pre-training. MAE's aggressive masking forces the encoder to learn global semantic features rather than local texture patterns, making it ideal for soundscape audio where birds appear in cluttered acoustic scenes.

---
**⚠️ EXPERIMENT STATUS (2026-02-01)**

Experiment 003 (Audio-MAE Pre-training) was planned but **deferred** after discussion with Anthony. We decided to:
1. Focus on validating Experiment 002 (Supervised ViT Baseline) infrastructure first
2. Ensure models fit on single GPU (competition constraint for CPU inference)
3. Complete HSN baseline and evaluate results before pursuing self-supervised pre-training

**Planning artifacts**: Archived at `user/murilo/experiments/_archived/003-audio-mae-pretraining/`

**Concept preservation**: This document remains valuable for future reference if we decide to pursue MAE pre-training after Experiment 002 completes.

---

## Motivation

**Problems with current approaches**:
1. **Long-tail distribution**: BirdSet has ~9,736 species in XCM, many with <10 training examples
2. **Focal → Soundscape gap**: Models trained on clean focal recordings fail on noisy soundscapes
3. **Limited labeled data**: Annotating bioacoustics data is expensive and time-consuming
4. **Transfer learning gaps**: ImageNet-pretrained models don't understand audio structure

**Why self-supervised pre-training helps**:
- Can leverage all available audio (labeled or not)
- Learns semantic representations of bird vocalizations
- Encoder trained to reconstruct masked regions must understand frequency patterns, temporal structure, and species-specific characteristics
- After pre-training, encoder can be fine-tuned on small labeled datasets or used for zero-shot retrieval

**Strategic value for BirdCLEF 2026**:
- Pre-trained encoder provides strong foundation for downstream tasks
- Zero-shot retrieval enables query-by-example without retraining
- Fine-tuning on BirdCLEF data should improve generalization vs. training from scratch

## Background

**Masked Autoencoders (MAE)**:
- Original MAE paper (He et al., 2022): 75% masking, asymmetric encoder-decoder, learned on ImageNet
- AudioMAE (Huang et al., 2022): Adapted MAE for audio spectrograms, used 80% masking
- SSAMBA: Self-supervised learning for bird vocalizations (prior work on BirdSet)

**Vision Transformers for Audio**:
- Audio Spectrogram Transformer (AST): ViT applied to mel-spectrograms
- Patch-based processing: 16×16 patches capture local frequency-time patterns
- Positional embeddings encode temporal order

**BirdSet Context**:
- XCM dataset: ~160k recordings, ~9,736 species (long-tail distribution)
- Focal vs. soundscape: Training on focal, testing on soundscape (domain shift)
- Current best: Supervised ConvNeXT achieves cmAP ~0.36 on average

**Key References**:
- MAE: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2022)
- AudioMAE: "Masked Autoencoders that Listen" (Huang et al., 2022)
- BirdSet paper: Benchmarking bioacoustic models
- AST: "AST: Audio Spectrogram Transformer" (Gong et al., 2021)

## Open Questions

**Architecture & Training**:
- Is 75% masking optimal for bird audio, or should we explore 60-80%?
- Should we use patch normalization (MAE recommendation) or global normalization?
- What decoder depth is sufficient? (4 layers vs. 6-8 layers)
- Should we pre-train on XCM only, or include external datasets (Xeno-Canto, AudioSet)?

**Evaluation & Downstream Tasks**:
- How do we quantitatively measure representation quality beyond reconstruction loss?
- Will MAE pre-training outperform ImageNet pre-training for BirdSet fine-tuning?
- Can we use the encoder for zero-shot retrieval without any labeled data?
- Does pre-training help with focal → soundscape transfer?

**Computational Efficiency**:
- Can we reduce encoder size (ViT-Small) while maintaining quality?
- What's the minimum pre-training duration (epochs) before seeing benefits?
- Should we explore block masking vs. random masking for efficiency?

**Domain-Specific Adaptations**:
- Should we mask frequency bands differently than time steps?
- Can we incorporate bird-specific priors (e.g., mask vocalizations preferentially)?
- Should we use multi-scale patches (mix of 8×8 and 16×16)?

## Testable Hypotheses

**Pre-training Quality**:
1. **H1**: MAE pre-trained on XCM will produce visually coherent mel-spectrogram reconstructions (not gray blur) after 100 epochs
2. **H2**: Reconstruction MSE loss will decrease to <0.5 (50% reduction from random initialization)
3. **H3**: Encoder embeddings will cluster by species in t-SNE visualization

**Downstream Performance**:
4. **H4**: MAE pre-trained encoder + fine-tuning will achieve higher cmAP than ImageNet pre-trained ViT on BirdSet
5. **H5**: MAE pre-trained encoder will enable zero-shot retrieval with >0.20 mean Average Precision
6. **H6**: Pre-training will improve performance on rare species (tail of distribution) more than common species

**Supervised Baseline**:
7. **H7**: Supervised ViT-Base trained from scratch on XCM will achieve cmAP ≥0.25 (competitive with BirdSet baselines)
8. **H8**: On-the-fly mel-spectrogram generation will maintain GPU utilization >80% during training

**Infrastructure**:
9. **H9**: PyTorch Lightning pipeline will handle 8-16 workers for data loading without bottlenecks
10. **H10**: Training can complete on single RTX 6000 GPU without OOM (batch_size ≥32 for supervised, ≥64 for MAE)

## Connections

**Based on**:
- MAE paper (He et al., 2022): Masked autoencoding framework
- AudioMAE (Huang et al., 2022): Audio-specific adaptations
- BirdSet benchmarks: Evaluation protocol and datasets
- Experiment 001: Validated BirdSet data pipeline and metrics

**Related to**:
- `soundscape-domain-shift`: MAE may help bridge focal → soundscape gap via robust representations
- `data-and-compute-scale`: Self-supervised learning maximizes data efficiency (unlabeled audio)
- `research-leads`: Positions ViT + MAE as foundation model candidate for distillation

**Could lead to**:
- **Experiment 002**: Supervised ViT Baseline (infrastructure + supervised performance)
- **Experiment 003**: Audio-MAE Pre-training (self-supervised encoder learning)
- **Experiment 004** (future): Fine-tuning MAE encoder on BirdCLEF 2026
- **Experiment 005** (future): Zero-shot retrieval with MAE embeddings
- **Experiment 006** (future): Distillation of MAE encoder to MLP-Mixer (CPU inference)

**Planning connections**:
- Could inform architecture decisions for BirdCLEF 2026 submission
- MAE encoder could be base model for the MLP-Mixer distillation concept

## Notes

**Implementation Strategy**:
- Start with supervised ViT to validate infrastructure (GPU data loading, metrics, W&B)
- Use HSN dataset for rapid iteration (21 species, ~5k samples) before scaling to XCM
- Reuse infrastructure from Experiment 002 (DataModule, mel-spec generation) in Experiment 003

**Critical Design Decisions**:
1. **Map-style dataset** (download ~90GB XCM) vs. streaming: Chose map-style for simplicity and speed
2. **On-the-fly mel-spec generation** vs. pre-compute: Chose on-the-fly to save ~500GB storage
3. **AudioSet normalization** (mean=-4.268, std=4.569) vs. BirdSet-specific: Chose AudioSet for consistency with existing models
4. **75% masking ratio**: Balances MAE (75%) and AudioMAE (80%) recommendations
5. **Lightweight decoder** (4 layers, 512 dim): MAE paper shows decoder depth doesn't affect encoder quality

**Risks & Mitigations**:
- **Risk**: MAE reconstructions are gray blur (averaging, not learning)
  - **Mitigation**: Monitor W&B visualizations from epoch 1, reduce mask ratio if needed
- **Risk**: Pre-training doesn't transfer to downstream tasks
  - **Mitigation**: Design Experiment 004 to compare MAE vs. ImageNet vs. random init
- **Risk**: Computational cost too high for single GPU
  - **Mitigation**: Start with HSN (quick validation), use gradient accumulation if needed

**Alternative Approaches Considered**:
1. **Contrastive learning** (SimCLR, MoCo): Requires careful augmentation design for audio, harder to implement
2. **Autoregressive modeling** (GPT-style): Doesn't leverage 2D structure of spectrograms as well as MAE
3. **BERT-style masking** (15% random tokens): Less aggressive than MAE, may not force semantic understanding
4. **Supervised multi-task learning**: Still requires labels, doesn't leverage unlabeled data

**Why MAE over alternatives**:
- Simple to implement (encoder-decoder with random masking)
- Proven effective in vision (MAE) and audio (AudioMAE) domains
- Aggressive masking (75%) forces semantic understanding
- Works well with ViT architecture (patch-based, no CNN inductive bias)
- Can visualize reconstructions to validate learning

**Progressive Complexity**:
```
Experiment 001 (Complete)
    └─> BirdSet benchmarks (Phase 2A)
         └─> Infrastructure validated

Experiment 002 (Implementing)
    └─> Supervised ViT Baseline
         └─> PyTorch Lightning pipeline
         └─> On-the-fly mel-spec generation
         └─> Multi-label metrics (cmAP, AUROC)
         └─> HSN → XCM workflow (HSN first)

Experiment 003 (DEFERRED - 2026-02-01)
    └─> Audio-MAE Pre-training
         └─> Planning complete, implementation deferred
         └─> See archived planning docs
         └─> May revisit after Exp 002 completes

Future (TBD)
    └─> Based on Exp 002 results
    └─> Options: Scale to XCM, iterate architecture, or explore BEATS/EATS
```

**Success Criteria Evolution**:
- **Experiment 002 success**: cmAP ≥0.25 on XCM, GPU util >80%, infrastructure robust
- **Experiment 003 success**: Reconstruction MSE <0.5, visual quality good, encoder embeddings cluster by species
- **Future work success**: MAE pre-training improves downstream cmAP by ≥5% vs. random init

**Timeline Estimate** (if pursuing):
- Experiment 002: ~1 week (implementation + HSN validation + XCM training)
- Experiment 003: ~1 week (MAE implementation + HSN test + XCM pre-training)
- Evaluation & analysis: ~3-5 days per experiment
- Total: ~3 weeks for both experiments + analysis

**Resource Requirements**:
- GPU: RTX 6000 (24GB VRAM) sufficient for both experiments
- Storage: ~90GB for XCM dataset (map-style), ~50GB for checkpoints
- Compute: ~24-48 hours per full XCM training run (supervised or MAE)
- W&B: Track metrics, reconstructions, system utilization

# Proposal: Pretrained Audio Model Fine-Tuning on BirdCLEF 2025

**Status**: proposed
**Created**: 2026-02-21
**Author**: Murilo Gustineli

## Hypothesis

Fine-tuning a pretrained self-supervised audio model (BEATs or EAT) on the BirdCLEF 2025 dataset will significantly outperform training from scratch (exp 002/003/004), because these models have learned general-purpose audio representations from AudioSet-2M that transfer well to bioacoustics.

## Background

Experiments 002–004 trained ViT and ConvNeXT models from scratch or with ImageNet pretraining on BirdSet HSN (~21 species). The best result was cmAP=0.118 (exp 004, streaming 128x4). However:

- **ImageNet pretraining** captures visual features, not audio semantics
- **Self-supervised audio models** (BEATs, EAT) are pretrained on AudioSet-2M with audio-specific objectives (acoustic tokenization, masked prediction)
- **BirdCLEF 2025** is a larger, more realistic dataset than BirdSet HSN, with hundreds of species and real-world soundscape evaluation

### Model Candidates

| Model | Params | Pretraining | Training Cost | Key Innovation |
|-------|--------|-------------|---------------|----------------|
| BEATs iter3+ | ~90M | AudioSet-2M (AS2M fine-tuned) | 342 epochs (high) | Iterative acoustic tokenizer |
| EAT | ~90M | AudioSet-2M | 10 epochs (15x cheaper) | Sparse encoding + UFO loss |

**Initial choice**: EAT — same quality as BEATs with 15x less training cost. If EAT proves difficult to integrate, fall back to BEATs which has simpler usage patterns.

## Method

### Approach

1. **Download** BirdCLEF 2025 competition data (train_audio + metadata)
2. **EDA** to understand species distribution, audio properties, class imbalance
3. **Fine-tune** pretrained EAT (or BEATs) with a new classification head for BirdCLEF species
4. **Compare** frozen encoder (linear probe) vs full fine-tuning

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdCLEF 2025 train_audio (~30-50 GB, hundreds of species) |
| Compute | 1x RTX 6000, 64 GB RAM, 8 CPUs |
| Dependencies | torch, torchaudio, nnAudio, BEATs/EAT checkpoints |
| Code | `user/murilo/experiments/005-pretrained-finetune/` |

### Procedure

1. Download BirdCLEF 2025 data to shared storage
2. Run EDA: species counts, duration distribution, class imbalance
3. Create PyTorch Dataset + DataModule for BirdCLEF 2025
4. Load pretrained encoder, attach classification head
5. Train linear probe (frozen encoder) as baseline
6. Train full fine-tuning with lower learning rate for encoder
7. Evaluate on BirdCLEF 2025 validation split

### Variables

- **Independent**: training strategy (frozen vs fine-tuned encoder), pretrained model (BEATs vs EAT)
- **Dependent**: cmAP, AUROC, F1 on validation set
- **Controlled**: data preprocessing (5s chunks, 32kHz, streaming), optimizer (AdamW), training duration

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| cmAP | Class-mean average precision (primary, matches competition metric) |
| AUROC | Area under ROC curve |
| F1 (macro) | Macro-averaged F1 score |

### Baseline

- Exp 003 best: ConvNeXT-Tiny pretrained (ImageNet) on HSN — cmAP=0.073
- Exp 004 best: Vertical 128x4 streaming on HSN — cmAP=0.118
- Note: direct comparison is limited since BirdCLEF 2025 has different species/data than HSN

### Success Criteria

- **Confirm if**: fine-tuned model achieves cmAP > 0.15 on BirdCLEF 2025 validation
- **Reject if**: fine-tuned model performs no better than a randomly initialized model of same architecture

## Limitations

- BirdCLEF 2025 evaluation is on soundscapes (different domain than focal recordings in train)
- Pretrained models use 16kHz; BirdCLEF audio is 32kHz — may need resampling
- EAT/BEATs expect specific input format (fbank features) that may differ from our mel pipeline
- Class imbalance in BirdCLEF may require weighted loss or sampling strategies

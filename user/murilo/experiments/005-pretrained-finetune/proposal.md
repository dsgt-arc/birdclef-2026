# Proposal: Pretrained Audio Model Fine-Tuning on BirdCLEF 2025

**Status**: in-progress
**Created**: 2026-02-21
**Updated**: 2026-02-22
**Author**: Murilo Gustineli

## Hypothesis

Fine-tuning a pretrained self-supervised audio encoder (EAT, pretrained on AudioSet-2M) on BirdCLEF 2025 focal recordings will achieve cmAP > 0.15 on a held-out test split, demonstrating that audio-domain pretraining provides useful representations for bird species classification.

## Background

Experiments 002–004 trained ViT and ConvNeXT models on BirdSet HSN (~21 Hawaiian species). Those results are **not directly comparable** to this experiment because:

- **Different dataset**: BirdCLEF 2025 has 206 species (vs 21 in HSN), different geographic coverage (South America vs Hawaii), and far more training data (28k recordings vs ~2k)
- **Different model family**: EAT is a self-supervised audio transformer (pretrained on AudioSet-2M), whereas exp 002–004 used vision models (ViT, ConvNeXT) with ImageNet or no pretraining

This experiment establishes the **first baseline on BirdCLEF 2025** and validates whether audio-domain self-supervised pretraining (EAT) produces useful representations for bird species classification. The key comparisons are:

1. **Frozen encoder (linear probe)** vs **full fine-tuning** — measures how much task-specific adaptation helps beyond the pretrained features
2. **Training curves and generalization** — does the model learn meaningful patterns (val metrics improving) or overfit?

Future experiments can compare against other architectures on the same BirdCLEF 2025 data for fair comparisons.

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
| Data | BirdCLEF 2025 train_audio (~11.5 GB, 206 species, 28k recordings) |
| Compute | 1x RTX 6000, 64 GB RAM, 8 CPUs |
| Dependencies | torch, torchaudio, nnAudio, BEATs/EAT checkpoints |
| Code | `user/murilo/experiments/005-pretrained-finetune/` |

### Procedure

1. Download BirdCLEF 2025 data to shared storage
2. Run EDA: species counts, duration distribution, class imbalance
3. Create PyTorch Dataset + DataModule for BirdCLEF 2025
4. Create **80/10/10 stratified train/val/test split** from focal recordings (by primary_label)
   - Train: model training
   - Val: hyperparameter tuning, early stopping, checkpoint selection
   - Test: held-out final evaluation (never used during training)
5. Load pretrained encoder, attach classification head
6. Train linear probe (frozen encoder) as baseline
7. Train full fine-tuning with lower learning rate for encoder
8. Evaluate best checkpoint on held-out test set

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

This experiment establishes the **first baseline on BirdCLEF 2025**. There is no prior result on this dataset to compare against.

For reference only (different dataset, not a fair comparison):
- Exp 003 best: ConvNeXT-Tiny (ImageNet pretrained) on BirdSet HSN — cmAP=0.073
- Exp 004 best: Vertical 128x4 streaming on BirdSet HSN — cmAP=0.118

### Success Criteria

- **Confirm if**: fine-tuned EAT achieves cmAP > 0.15 on held-out BirdCLEF 2025 test split, AND fine-tuning outperforms frozen linear probe
- **Reject if**: model fails to learn (val metrics do not improve over random), or test cmAP is no better than chance (1/206 ≈ 0.005)

## Limitations

- **Train/test domain gap**: training data is focal recordings (clean, single species), but the actual competition evaluates on soundscapes (noisy, multi-species, continuous). Our held-out test split is also focal recordings, so it measures in-domain generalization only.
- **No Kaggle leaderboard comparison yet**: true competition performance requires a Kaggle submission (future work).
- Pretrained models use 16kHz; BirdCLEF audio is 32kHz — resampling handled in data pipeline.
- EAT expects Kaldi fbank features — handled in encoder wrapper.
- Severe class imbalance (max 990, min 2 per species) may require weighted loss or sampling in future iterations.

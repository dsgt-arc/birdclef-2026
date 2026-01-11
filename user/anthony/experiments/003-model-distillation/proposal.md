# Proposal: Distilled Sequence Model for Context Recovery

**Status**: proposed
**Created**: 2026-01-11
**Author**: Anthony

## Hypothesis

Distilling a large, pretrained Audio Transformer (BEATs) into a lightweight student model (e.g., Mamba, CNN-Pooler, or Tiny Transformer) using **unlabeled data from AudioSet** will allow us to recover >90% of the teacher's performance while maintaining the order-of-magnitude speedup gained from static embedding lookups.

Specifically:
- **Teacher**: Full BEATs Transformer (12 layers, 768-dim), pretrained on AudioSet.
- **Student**: Lightweight sequence model operating on discrete tokens/static embeddings.
- **Distillation Data**: AudioSet (Unbalanced Train, ~2M clips).
- **Hypothesis**: The student model will achieve >45% mAP on AudioSet (Teacher is ~50%) and >75% accuracy on ESC-50, while keeping inference latency <10ms on CPU.

## Background

In Experiment 002, we successfully accelerated inference by 32x using static embeddings (Model2Vec approach) but suffered a massive 48% accuracy drop due to "Context Collapse." We found that training a lightweight "Contextual Pooler" (LightCNN) on the labeled ESC-50 dataset recovered some accuracy (~56%), but was likely overfitting due to the small dataset size (2,000 samples).

To bridge the gap to the teacher's performance, we need to learn robust temporal grammar from a much larger dataset. Since we have a strong teacher (BEATs) pretrained on AudioSet, we can use knowledge distillation on the same domain to train a student that mimics the teacher's contextualization.

Reference: `docs/concepts/distilled-sequence-model.md`

## Method

### Approach

We will implement a Teacher-Student distillation pipeline on AudioSet:
1.  **Pre-compute Teacher Targets**: Run the full BEATs model on AudioSet (Unbalanced Train) and save the sequence of contextualized embeddings.
2.  **Pre-compute Student Inputs**: Run the BEATs Tokenizer on the same data to get discrete token indices.
3.  **Train Student**: Train lightweight sequence models (Mamba, CNN, Transformer) to predict the teacher's embeddings given the static embeddings of the tokens.
4.  **Evaluate**:
    *   **Distillation**: Loss on held-out AudioSet data.
    *   **Downstream**: mAP on AudioSet Eval (finetuned head) and Accuracy on ESC-50 (linear probe).

### Setup

| Component | Details |
|-----------|---------|
| Data | AudioSet (Unbalanced Train ~2M clips, Eval ~20k clips) |
| Compute | GPU (L40S) for teacher inference & student training |
| Dependencies | `torch`, `BEATs`, `transformers`, `mamba-ssm` |
| Code | New `distillation.py` script |

### Procedure

1.  **Dataset Preparation**: Download/Prepare AudioSet shards (webdataset format preferred).
2.  **Teacher Inference**: Generate targets for the training set.
    *   Input: Raw Audio.
    *   Output: `(Batch, 248, 768)` contextual embeddings.
3.  **Student Architecture Selection**: Implement candidate students:
    *   **Mamba**: State Space Model (linear complexity), ~1-3M params.
    *   **MLP-Mixer**: All-MLP architecture with token and channel mixing.
    *   **CNN-Pooler**: Scaled-up version of Exp 002's LightCNN (residual 1D CNNs).
    *   **TinyTransformer / EAT-Student**: 2 layers, 256-dim, 4 heads. Will explore EAT-style sparse training (masking 70% of tokens) to accelerate training and force global context learning.
4.  **Distillation Training**:
    *   Loss: MSE / Cosine Similarity between Student(TokenEmbeddings) and Teacher(Embeddings).
    *   Optimizer: AdamW.
5.  **Evaluation**:
    *   **AudioSet mAP**: Train a linear classifier on frozen student features using AudioSet Balanced Train.
    *   **ESC-50 Accuracy**: Train a linear classifier on frozen student features using ESC-50.

### Variables

- **Independent**: Student architecture (Mamba vs CNN vs Transformer).
- **Dependent**: AudioSet mAP, ESC-50 Accuracy, CPU Inference Latency.
- **Controlled**: Teacher model (BEATs iter3), Tokenizer, Input Audio Config (16kHz, 5s).

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **AudioSet mAP** | Mean Average Precision on AudioSet Eval set (primary benchmark). |
| **Accuracy (ESC-50)** | Top-1 accuracy on 5-fold CV (transfer learning benchmark). |
| **Inference Time** | Latency per 5s sample on CPU (batch size 1). |

### Baseline

- **Lower Bound**: Static Embeddings (Exp 002) -> 36% acc (ESC-50).
- **Current Best**: LightCNN trained on ESC-50 (Exp 002) -> ~56% acc (ESC-50).
- **Upper Bound**: Full BEATs Teacher -> 50.6% mAP (AudioSet), 84% acc (ESC-50).

### Success Criteria

- **Confirm if**: Student achieves >45% mAP on AudioSet and >75% accuracy on ESC-50 with <10ms latency.
- **Reject if**: Student fails to outperform LightCNN (~56% on ESC-50) or latency >20ms.

## Limitations

- **Distillation Gap**: Student capacity might be too small to capture full teacher complexity.
- **Storage**: Storing teacher embeddings for 2M clips is huge (~1.5TB). Will prioritize on-the-fly distillation or streaming if storage is tight.
- **Mamba Support**: `mamba-ssm` kernels usually require GPU; need to verify pure PyTorch CPU performance for inference.

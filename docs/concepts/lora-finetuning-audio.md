# Concept: LoRA Finetuning for Audio Transformers

**Status**: historical
**Created**: 2026-01-10

## Related Documentation

- **Experiment**: [001-encoder-benchmark](../../user/anthony/experiments/001-encoder-benchmark/proposal.md) - Testing LoRA on ESC-50
- **References**:
  - [SSL Pretraining Objectives](../references/ssl-pretraining-objectives.md) - Why EAT needs finetuning (reconstruction-based SSL)

## Motivation

Large audio transformers (BEATs, EAT, ~86M params) overfit on small datasets like ESC-50 (2,000 samples). Full finetuning has ~53,000 parameters per training sample - a recipe for memorization.

**Linear probe results show the gap:**
- BEATs: 95.8% (frozen) vs 98.1% (finetuned) = 2.3% gap
- EAT: 77.9% (frozen) vs 95.9% (finetuned) = 18.0% gap

EAT needs finetuning but can't afford full parameter updates on small datasets.

## LoRA: Low-Rank Adaptation

LoRA constrains weight updates to a low-rank subspace:

```
Original:  y = Wx
LoRA:      y = Wx + BAx    where B ∈ R^(d×r), A ∈ R^(r×d), r << d
```

Only A and B are trained. W is frozen. The low-rank bottleneck (r=8-16) acts as regularization.

### Parameter Reduction

| Approach | Trainable | Per Sample (n=1600) | Overfit Risk |
|----------|-----------|---------------------|--------------|
| Full finetune | 86M | 53,750 | Very High |
| LoRA (r=8) | ~300K | 188 | Low |
| LoRA (r=16) | ~600K | 375 | Low |
| Linear probe | ~40K | 25 | Minimal |

## EAT Model Structure

```
EATModel
└── model (EAT)
    ├── local_encoder.proj          # Conv2d patch embedding
    ├── blocks.0-11 (AltBlock)      # 12 transformer layers
    │   ├── attn.qkv                # Linear(768, 2304) - Q,K,V combined
    │   ├── attn.proj               # Linear(768, 768)  - attention output
    │   ├── mlp.fc1                 # Linear(768, 3072) - MLP up
    │   └── mlp.fc2                 # Linear(3072, 768) - MLP down
    └── pre_norm                    # Final LayerNorm
```

### Target Modules for LoRA

**Option 1: Attention only (conservative)**
```python
target_modules = ["qkv", "proj"]
# ~12 layers × 2 modules × 2 matrices (A, B) × r × d
# r=8: ~295K trainable params
```

**Option 2: Attention + MLP (more capacity)**
```python
target_modules = ["qkv", "proj", "fc1", "fc2"]
# ~12 layers × 4 modules × 2 matrices × r × d
# r=8: ~590K trainable params
```

## Recommended Configuration

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,                              # Rank (8-16 for small datasets)
    lora_alpha=16,                    # Scaling (typically 2×r)
    lora_dropout=0.1,                 # Regularization
    target_modules=["qkv", "proj"],   # Attention layers
    bias="none",                      # Don't train biases
    task_type=TaskType.FEATURE_EXTRACTION,
)
```

### Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-4 to 5e-4 | Higher than full finetune (LoRA allows) |
| Batch size | 32 | Same as linear probe |
| Epochs | 50-100 | Early stopping on val loss |
| Weight decay | 0.01 | Light regularization |
| Warmup | 10% of steps | Stabilize early training |
| Scheduler | Cosine | Smooth decay |

### Data Augmentation

Still beneficial but less critical than full finetuning:
- **Mixup** (α=0.3): Interpolate samples and labels
- **SpecAugment**: Mask frequency/time bands
- **Time shift**: Random crop within clip

## Implementation Notes

### Adding Classification Head

EAT's `head` is `Identity()` by default. Need to add classification head:

```python
class EATForClassification(nn.Module):
    def __init__(self, encoder, num_classes, embed_dim=768):
        super().__init__()
        self.encoder = encoder  # PEFT-wrapped EAT
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.encoder.extract_features(x)
        # Mean pool over sequence (excluding CLS)
        pooled = features[:, 1:].mean(dim=1)
        return self.classifier(pooled)
```

### Gradient Checkpointing

For memory efficiency on V100:
```python
model.gradient_checkpointing_enable()
```

## Expected Results

Based on similar LoRA experiments on small datasets:

| Method | Expected Accuracy | Notes |
|--------|------------------|-------|
| Linear probe | 77.9% | Baseline (current) |
| LoRA r=8 | 85-90% | Conservative |
| LoRA r=16 | 88-92% | More capacity |
| Full finetune + aug | 93-96% | If regularized well |

## Wav2Vec2 Considerations

### Can we LoRA finetune wav2vec2 for environmental sounds?

**Yes, but with caveats:**

| Factor | Assessment |
|--------|------------|
| Architecture | Compatible - standard transformer with q, k, v, proj |
| Pretraining domain | Speech (LibriSpeech) - poor match for env. sounds |
| Linear probe baseline | 48.2% on ESC-50 (near random for 50 classes) |
| Expected LoRA improvement | Uncertain - starting from weak representations |

### Why wav2vec2 struggles with environmental sounds

1. **Pretraining objective**: Contrastive learning on masked speech learns phoneme/word boundaries
2. **Acoustic patterns**: Speech has harmonic structure, formants; env. sounds are often noisy, impulsive
3. **Temporal dynamics**: Speech has syllable-level structure; env. sounds vary widely (rain vs dog bark)

### Wav2Vec2 Model Structure (for LoRA)

```
Wav2Vec2Model
├── feature_extractor (CNN)      # Frozen - extracts local features
├── feature_projection           # Linear projection
└── encoder (Transformer)
    └── layers.0-11
        ├── attention.q_proj     # Target for LoRA
        ├── attention.k_proj     # Target for LoRA
        ├── attention.v_proj     # Target for LoRA
        ├── attention.out_proj   # Target for LoRA
        ├── feed_forward.intermediate_dense
        └── feed_forward.output_dense
```

Target modules: `["q_proj", "k_proj", "v_proj", "out_proj"]`

### Experimental Hypothesis

LoRA finetuning wav2vec2 on ESC-50 will likely:
- Improve over linear probe (48% → 60-75%?)
- Still underperform EAT/BEATs (domain mismatch too large)
- Provide useful baseline for "how much can adaptation help a mismatched model"

Worth testing as a comparison point.

## References

- [HuggingFace PEFT LoRA Guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Soxan: Wav2Vec2 for audio classification](https://github.com/m3hrdadfi/soxan)
- EAT model: `worstchan/EAT-base_epoch30_pretrain`
- wav2vec2 model: `facebook/wav2vec2-base-960h`

## Next Steps

1. ~~Implement LoRA wrapper for EAT~~ Done
2. ~~Add classification head~~ Done
3. Run ESC-50 5-fold CV with LoRA (EAT)
4. Add wav2vec2 LoRA finetuning for comparison
5. Compare r=8 vs r=16
6. Add mixup augmentation if needed

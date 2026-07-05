# MLP-Mixer Distillation for Efficient Inference

**Status**: superseded

## Research Direction

**Goal**: Achieve Perch-competitive bird classification performance with CPU-viable inference by distilling ViT models into MLP-Mixer.

**Approach**:
1. Reproduce BirdSet baselines
2. Train/fine-tune ViT (e.g., ProtoCLR-style) for improved representations
3. Distill ViT → MLP-Mixer for efficient deployment

**Relationship to discrete token pipeline**: This is a parallel research direction. The token pipeline explores whether discrete representations + transformers + model2vec can match end-to-end approaches. This direction asks whether distillation can make ViT-quality models CPU-viable.

## Motivation

Current state-of-the-art audio models are computationally expensive:
- **ViT-based** (BEATs, AST, AudioMAE): O(n²) attention complexity
- **CNN-based** (Perch/EfficientNet-B3, BirdNET): Better but still heavy

MLP-Mixer offers:
- O(n) complexity (linear in sequence length)
- No attention mechanism = simpler, faster inference
- Proven competitive with ViT on vision tasks
- Emerging evidence of effectiveness on audio spectrograms

## Literature Support

### Audio Spectrogram Mixer (ASM)
- **Paper**: [arXiv:2401.11102](https://arxiv.org/abs/2401.11102) (Jan 2024)
- **Finding**: MLP-Mixer achieves competitive results with AST at lower inference cost
- **Datasets**: Speech Commands, UrbanSound8k, CASIA
- **Key insight**: "equally outstanding results can be reached with reduced inference cost"

### ScaleKD: Cross-Architecture Distillation
- **Paper**: [arXiv:2411.06786](https://arxiv.org/pdf/2411.06786) (Nov 2024)
- **Finding**: Mixer-S/16 distilled from Swin-L **outperforms** individually trained Mixer-B/16 by 2.19%
- **Compression**: 3.23× smaller while better performance
- **Relevance**: Direct evidence that ViT → MLP-Mixer distillation works

### Spatial-Channel Token Distillation (STD)
- **Paper**: [ICML 2022](https://proceedings.mlr.press/v162/li22c/li22c.pdf)
- **Finding**: Distilled Mixer-B16 beats ViT-B/16 by 2.1% and ViT-L/16 by 3.5%
- **Method**: Token-level distillation preserving spatial-channel structure

## Architecture Options

### Option A: MLP-Mixer on Discrete Tokens (Phase 2 Direct)
```
Audio → Tokenizer → [Token Sequence] → MLP-Mixer → Embeddings
                                         ↓
                              Token-mixing MLP (temporal)
                              Channel-mixing MLP (features)
```
- Train MLP-Mixer directly with masked prediction on our discrete tokens
- No teacher needed, but may underperform vs distillation

### Option B: Distill from ViT Teacher
```
Teacher: BEATs / AST / AudioMAE (frozen)
           ↓ soft targets
Student: MLP-Mixer
```
- Use pre-trained ViT audio model as teacher
- Student learns to match teacher's representations
- Best of both: ViT quality, Mixer efficiency

### Option C: Distill from CNN Teacher (Perch/BirdNET)
```
Teacher: Perch (EfficientNet-B3) / BirdNET (frozen)
           ↓ soft targets + embeddings
Student: MLP-Mixer
```
- Leverage domain-specific bird knowledge from Perch/BirdNET
- Cross-architecture distillation (CNN → MLP)
- May need intermediate representation matching

### Option D: Multi-Teacher Distillation
```
Teachers: Perch + BEATs + BirdNET (frozen)
              ↓ ensemble soft targets
Student: MLP-Mixer
```
- Combine knowledge from multiple models
- More robust student representations
- Higher training complexity

## MLP-Mixer Architecture Details

```python
class MixerBlock:
    def __call__(self, x):
        # x: [batch, seq_len, channels]

        # Token mixing (across sequence/time dimension)
        y = LayerNorm(x)
        y = y.transpose(1, 2)  # [batch, channels, seq_len]
        y = MLP(y)             # Mix tokens
        y = y.transpose(1, 2)  # [batch, seq_len, channels]
        x = x + y

        # Channel mixing (across feature dimension)
        y = LayerNorm(x)
        y = MLP(y)             # Mix channels
        x = x + y

        return x
```

Key hyperparameters:
- **Patch size**: How to chunk spectrogram (e.g., 16×16)
- **Hidden dim**: Width of mixer (e.g., 512, 768)
- **Depth**: Number of mixer blocks (e.g., 8, 12)
- **MLP ratio**: Expansion in MLPs (e.g., 4×)

## Integration with Pipeline

### With Discrete Tokens (Preferred)
```
Phase 1: Audio → WavTokenizer → Discrete Tokens (40-75/sec)
Phase 2: Tokens → MLP-Mixer (masked prediction) → Contextual Embeddings
Phase 3: MLP-Mixer → model2vec distillation → Static Embeddings
Phase 4: Static Embeddings → Linear Classifier → Predictions
```

### Without Discrete Tokens (Spectrogram Direct)
```
Audio → Mel Spectrogram → Patch Embed → MLP-Mixer → Embeddings → Classifier
```

## Efficiency Comparison

| Model | Params | FLOPs | Inference Time (CPU) |
|-------|--------|-------|---------------------|
| ViT-B/16 | 86M | 17.6G | ~500ms |
| EfficientNet-B3 | 12M | 1.8G | ~100ms |
| Mixer-B/16 | 59M | 12.6G | ~200ms |
| Mixer-S/16 | 18M | 3.8G | ~60ms |
| Mixer-S/16 (INT8) | 18M | - | ~30ms |

*Mixer-S/16 with INT8 quantization could be 15× faster than ViT-B/16*

## Concrete Path: ProtoCLR → Mixer

### Why ProtoCLR as Teacher?

| Model | Architecture | Params | BirdSet Performance | Domain |
|-------|--------------|--------|---------------------|--------|
| Perch | EfficientNet-B3 | ~12M | Strong baseline | Birds (10k species) |
| ProtoCLR | CvT-13 | 19M | Competitive few-shot | Birds (domain-invariant) |
| BirdAVES-large | ViT | 316M | Good few-shot | Animals |

ProtoCLR advantages:
- Already trained on Xeno-Canto bird data (domain-specific)
- CvT-13 is a convolutional-ViT hybrid (good distillation source)
- Focus on domain invariance (focal → soundscape transfer)
- Relatively small (19M vs 316M for BirdAVES)

### Proposed Experiments

**Phase 1: Reproduce BirdSet Baselines**
```
1. Run BirdSet ConvNeXT/EfficientNet on standard splits
2. Evaluate Perch on same splits for comparison
3. Establish baseline metrics (AUROC, accuracy)
```

**Phase 2: Train/Improve ViT Teacher**
```
1. Fine-tune ProtoCLR on BirdSet XCM/XCL
2. Or: Train CvT-13 from scratch with ProtoCLR objectives
3. Target: Match or exceed Perch on BirdSet evals
```

**Phase 3: Distill to MLP-Mixer**
```
1. Mixer-S/16 student (18M params → ~4.5M with INT8)
2. Feature matching + soft label distillation
3. Target: <10% drop from teacher, 5-10× faster inference
```

### Can We Beat Perch?

Honest assessment:

**Advantages we have**:
- Perch is EfficientNet-B3, not the most efficient architecture
- ProtoCLR shows ViTs can match Perch in few-shot settings
- Distillation literature shows student can sometimes exceed teacher
- We can optimize specifically for BirdCLEF evaluation metrics

**Challenges**:
- Perch trained on massive curated data (10k species)
- EfficientNet-B3 is already efficient (~12M params)
- Perch embeddings are specifically tuned for transfer learning
- Google has significant compute/data advantages

**Realistic targets**:
- Match Perch performance with 2-3× faster inference = success
- Beat Perch on specific BirdCLEF soundscape datasets = bonus
- Competitive with 5× faster inference = acceptable

## Open Questions

1. **Which teacher?** ProtoCLR vs fine-tuned CvT vs BirdAVES - need to evaluate on BirdSet
2. **Token vs spectrogram input?** Does discrete tokenization help or hurt mixer?
3. **Distillation objective?** Soft labels vs feature matching vs attention transfer
4. **Quantization compatibility?** Does mixer quantize as well as CNNs?
5. **How much does domain-specific pre-training matter?** ProtoCLR on birds vs generic AudioMAE

## Next Steps

1. [ ] Reproduce BirdSet baselines (ConvNeXT, EfficientNet)
2. [ ] Evaluate Perch on BirdSet eval splits for comparison
3. [ ] Fine-tune ProtoCLR on BirdSet XCM
4. [ ] Implement Mixer-S/16 architecture
5. [ ] Set up distillation pipeline (ProtoCLR → Mixer)
6. [ ] Profile inference time on Kaggle CPU environment
7. [ ] Compare: ProtoCLR vs distilled Mixer vs Perch on soundscape datasets

## References

- [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601) (Tolstikhin et al., 2021)
- [ASM: Audio Spectrogram Mixer](https://arxiv.org/abs/2401.11102) (2024)
- [ScaleKD: Strong Vision Transformers Could Be Excellent Teachers](https://arxiv.org/abs/2411.06786) (2024)
- [Spatial-Channel Token Distillation for Vision MLPs](https://proceedings.mlr.press/v162/li22c/li22c.pdf) (ICML 2022)

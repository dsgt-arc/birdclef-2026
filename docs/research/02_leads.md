# Research Leads

Potential repositories, papers, and concepts to explore for the discrete token pipeline. These complement the [vendored repositories](../vendor/README.md) already documented.

## Tokenization (Phase 1)

### EnCodec
- **Source**: https://github.com/facebookresearch/encodec
- **Paper**: High Fidelity Neural Audio Compression (2022)
- **Why**: The foundational neural codec that WavTokenizer and DAC improve upon. Understanding EnCodec's RVQ design is essential for evaluating alternatives.
- **Priority**: Medium (for baseline understanding)

### SoundStream
- **Source**: https://github.com/google/lyra (uses SoundStream)
- **Paper**: SoundStream: An End-to-End Neural Audio Codec (2021)
- **Why**: Google's foundational codec; influenced EnCodec and subsequent work. Different design choices than EnCodec worth understanding.
- **Priority**: Low (historical context)

### Mimi
- **Source**: https://github.com/kyutai-labs/moshi (includes Mimi codec)
- **Paper**: Moshi: A Speech-Text Foundation Model for Real-Time Dialogue (2024)
- **Why**: Very recent codec with built-in semantic tokens at 12.5 Hz. Single codebook design like WavTokenizer. Could be ideal for our pipeline.
- **Priority**: High (novel approach, potentially better fit)

### AudioLM
- **Paper**: AudioLM: A Language Modeling Approach to Audio Generation (2022)
- **Why**: Introduces semantic vs acoustic token separation. The two-stage hierarchy (semantic tokens from w2v-BERT, acoustic from SoundStream) may inform our architecture.
- **Priority**: Medium (architectural insights)

## Self-Supervised Learning (Phase 2)

### BEATs
- **Source**: https://github.com/microsoft/unilm/tree/master/beats
- **Paper**: BEATs: Audio Pre-Training with Acoustic Tokenizers (2022)
- **Why**: Closest to our Phase 1+2 vision. Uses iterative training: (1) train tokenizer, (2) train transformer with masked prediction on tokens, (3) repeat. Achieves SOTA on AudioSet.
- **Priority**: High (directly relevant architecture)

### data2vec 2.0
- **Source**: https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec
- **Paper**: data2vec 2.0: Highly Efficient Self-Supervised Learning (2023)
- **Why**: 16x more efficient than original. Works across modalities including audio. Could accelerate our pre-training.
- **Priority**: Medium (efficiency gains)

### AudioMAE
- **Source**: https://github.com/facebookresearch/AudioMAE
- **Paper**: Masked Autoencoders that Listen (2022)
- **Why**: Simpler than contrastive approaches (just reconstruction). 80% masking ratio works well. May be easier to train than HuBERT-style.
- **Priority**: Medium (simpler alternative)

### MERT
- **Source**: https://github.com/yizhilll/MERT
- **Paper**: MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training (2023)
- **Why**: Transformer on audio tokens for music. Similar architecture goals to ours. Pre-trained checkpoints available.
- **Priority**: Low (music-focused, but transferable ideas)

## Bird/Bioacoustics-Specific

### BirdAVES
- **Source**: https://github.com/earthspecies/biodenoising (related)
- **Paper**: AVES: Animal Vocalization Encoder based on Self-Supervision (2023)
- **Why**: Self-supervised model specifically for animal vocalizations. Direct baseline mentioned in ProtoCLR. Two variants: biox-base (95M) and bioxn-large (316M).
- **Priority**: High (direct competitor/baseline)

### Animal2Vec
- **Source**: (Research project, limited public code)
- **Paper**: Self-supervised learning for animal vocalizations
- **Why**: Specifically designed for animal sounds rather than human speech. May handle bird-specific acoustic patterns better.
- **Priority**: Medium (if code becomes available)

### BEANS Benchmark
- **Source**: https://github.com/earthspecies/beans
- **Paper**: BEANS: The Benchmark of Animal Sounds (2022)
- **Why**: Complements BIRB benchmark. Broader animal sound evaluation. Useful for measuring generalization.
- **Priority**: Low (evaluation infrastructure)

## Inference Optimization (Phase 4)

### ONNX Export & Runtime
- **Source**: https://github.com/onnx/onnx, https://github.com/microsoft/onnxruntime
- **Why**: Critical for Kaggle CPU deployment. ONNX Runtime has optimized operators and can be 2-4x faster than PyTorch CPU inference.
- **Priority**: High (deployment requirement)

### Quantization Techniques
- **Concepts**: INT8 dynamic quantization, INT4 weight-only, QAT
- **Why**: model2vec mentions INT8 quantization halves size without performance loss. We should explore this systematically.
- **Priority**: High (Kaggle constraints)

### Knowledge Distillation
- **Concepts**: Teacher-student, feature matching, attention transfer
- **Why**: Beyond model2vec's approach. Could distill our transformer into even smaller models. DistilBERT-style approaches.
- **Priority**: Medium (if model2vec insufficient)

## Unexplored Directions

### Contrastive Language-Audio Pretraining (CLAP)
- **Source**: https://github.com/LAION-AI/CLAP
- **Why**: Text-audio alignment. Could enable zero-shot classification using species names/descriptions.
- **Priority**: Low (different paradigm, but interesting)

### Spectrogram Augmentation
- **Concepts**: SpecAugment, mixup, time-frequency masking
- **Why**: Standard but important. Should document best practices for bird audio specifically.
- **Priority**: Medium (training infrastructure)

### Multi-task Learning
- **Concepts**: Joint training on multiple datasets, auxiliary objectives
- **Why**: BirdSet provides multiple evaluation sets. Could improve generalization by training jointly.
- **Priority**: Low (optimization after core pipeline works)

---

## Next Steps

1. **Immediate**: Deep dive on BEATs and Mimi - closest to our architecture vision
2. **Short-term**: Set up BirdAVES as evaluation baseline alongside BirdNET/Perch
3. **Medium-term**: Investigate ONNX export pipeline for inference optimization
4. **Long-term**: Explore CLAP for zero-shot capabilities

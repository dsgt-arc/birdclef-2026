# Research Leads

**Status**: historical

Potential repositories, papers, and concepts to explore for the discrete token pipeline. These complement the [vendored repositories](../vendor/README.md) already documented.

> **Last updated**: December 2025. See also: [BEATs architecture](../references/deep-research/beats-architecture.md) for deep dive on BEATs architecture.

## Current SOTA Landscape (Dec 2025)

Based on recent benchmarks and the [Foundation Models for Bioacoustics comparative review](https://arxiv.org/abs/2508.01277):

| Model | Type | AudioSet mAP | Bioacoustics | Params | Training Cost | Priority |
|-------|------|--------------|--------------|--------|---------------|----------|
| **EAT** | General audio | 40.2% (AS-20K) | Good transfer | ~90M | 15× cheaper than BEATs | 🔴 High |
| **OpenBEATs** | General audio | SOTA | **Best on 6 bio datasets** | 300M | ~20k GPU-hrs | 🔴 High |
| **BirdMAE** | Bird-specific | - | **Best on BirdSet** | ~90M | Unknown | 🔴 High |
| **NatureLM-audio** | Bioacoustics LLM | - | SOTA zero-shot | 8B (LLM) | Large | 🟡 Medium |
| **BEATs** | General audio | 50.6% | Strong | 90M | ~20k GPU-hrs | 🟢 Completed |

**Key finding**: EAT achieves similar performance to BEATs with 15× less compute. OpenBEATs scales BEATs to 300M params and dominates bioacoustics benchmarks. BirdMAE wins on BirdSet specifically.

---

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

### EAT (Efficient Audio Transformer) ⭐ NEW
- **Source**: https://github.com/cwx-worst-one/EAT
- **Paper**: EAT: Self-Supervised Pre-Training with Efficient Audio Transformer (IJCAI 2024)
- **Why**: **15× more efficient than BEATs**, matches or exceeds performance. Uses bootstrap self-supervised training with Utterance-Frame Objective (UFO). Only 10 epochs vs BEATs' 342. Hugging Face integration available (May 2025).
- **Key insight**: Large inverse block masks + sparse encoding = faster training without quality loss.
- **Priority**: 🔴 **Critical** - may replace BEATs as our target architecture

### OpenBEATs ⭐ NEW
- **Source**: https://shikhar-s.github.io/OpenBEATs/
- **Paper**: OpenBEATs: A Fully Open-Source General-Purpose Audio Encoder (WASPAA 2025)
- **Why**: Scales BEATs to 300M params, trained on 20k hours including bioacoustics. **SOTA on 6 bioacoustic datasets**. Fully open-source training code, checkpoints, and logs. Outperforms billion-param models at 1/4 size.
- **Priority**: 🔴 **Critical** - best available bioacoustics encoder, open weights

### BEATs
- **Source**: https://github.com/microsoft/unilm/tree/master/beats
- **Paper**: BEATs: Audio Pre-Training with Acoustic Tokenizers (ICML 2023)
- **Why**: Iterative tokenizer training creates semantic audio vocabulary. Deep dive completed in [BEATs architecture](../references/deep-research/beats-architecture.md).
- **Status**: ✅ Research completed. Consider EAT or OpenBEATs as more efficient/powerful alternatives.
- **Priority**: 🟢 Completed (see [BEATs architecture](../references/deep-research/beats-architecture.md))

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

### NatureLM-audio ⭐ NEW
- **Source**: https://github.com/earthspecies/NatureLM-audio
- **Paper**: NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics (ICLR 2025)
- **Demo**: https://huggingface.co/EarthSpeciesProject/NatureLM-audio
- **Why**: First audio-language model for bioacoustics. Uses BEATs encoder + Q-Former + Llama-3.1-8B. SOTA zero-shot on BEANS-Zero. Can do species classification, detection, captioning via natural language prompts.
- **Key insight**: Combines audio encoder with LLM for flexible task specification. Trained on Xeno-canto, iNaturalist, marine mammal databases.
- **Priority**: 🟡 Medium (interesting for zero-shot, but 8B params too large for CPU inference)

### BirdMAE ⭐ NEW
- **Source**: (Check arxiv/github - mentioned in Foundation Models for Bioacoustics review)
- **Paper**: Referenced in [Foundation Models for Bioacoustics comparative review](https://arxiv.org/abs/2508.01277)
- **Why**: **Best performance on BirdSet benchmark**. Bird-specific MAE trained on large-scale bird song data.
- **Priority**: 🔴 **High** - need to find source and evaluate as potential teacher model

### BirdAVES / AVES
- **Source**: https://github.com/earthspecies/biodenoising (related)
- **Paper**: AVES: Animal Vocalization Encoder based on Self-Supervision (2023)
- **Why**: Self-supervised model specifically for animal vocalizations. Direct baseline mentioned in ProtoCLR. Two variants: biox-base (95M) and bioxn-large (316M).
- **Priority**: 🟡 Medium (superseded by OpenBEATs and BirdMAE on benchmarks)

### Animal2Vec
- **Source**: (Research project, limited public code)
- **Paper**: Self-supervised learning for animal vocalizations
- **Why**: Specifically designed for animal sounds rather than human speech. May handle bird-specific acoustic patterns better.
- **Priority**: Low (limited availability)

### Foundation Models for Bioacoustics Review ⭐ NEW
- **Source**: https://arxiv.org/abs/2508.01277
- **Paper**: Foundation Models for Bioacoustics – A Comparative Review (Aug 2025)
- **Why**: Comprehensive benchmark of all major audio encoders on BEANS and BirdSet. Compares AudioMAE, BEATs, EAT, Animal2Vec, AVES, BioLingual, BirdAVES, BirdMAE.
- **Key findings**: BirdMAE best on BirdSet, BEATs beats AudioMAE on bioacoustics, 32kHz sample rate and 5s windows recommended.
- **Priority**: 🔴 **High** - essential reference for model selection

### BEANS Benchmark
- **Source**: https://github.com/earthspecies/beans
- **Paper**: BEANS: The Benchmark of Animal Sounds (2022)
- **Why**: Complements BIRB benchmark. Broader animal sound evaluation. Useful for measuring generalization.
- **Priority**: 🟡 Medium (evaluation infrastructure)

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

## Next Steps (Updated Dec 2025)

### Immediate
1. ✅ Deep dive on BEATs - completed in [BEATs architecture](../references/deep-research/beats-architecture.md)
2. 🔴 **Deep dive on EAT** - 15× more efficient, may be better choice than BEATs
3. 🔴 **Evaluate OpenBEATs checkpoints** - best bioacoustics encoder available
4. 🔴 **Find BirdMAE source** - best on BirdSet, need to locate paper/code

### Short-term
5. Read [Foundation Models for Bioacoustics](https://arxiv.org/abs/2508.01277) review in full
6. Benchmark OpenBEATs vs Perch vs BirdNET on BirdSet evaluation sets
7. Prototype distillation: OpenBEATs → MLP-Mixer or model2vec

### Medium-term
8. Investigate whether EAT's efficiency gains apply to bioacoustics domain
9. ONNX export pipeline for inference optimization
10. Evaluate NatureLM-audio's BEATs encoder as standalone feature extractor

### Research Questions Refined
- Does EAT's 15× efficiency hold for bird audio, or only AudioSet?
- Can OpenBEATs (300M) be distilled to CPU-viable size while retaining bioacoustics performance?
- Is BirdMAE's domain-specific training worth it vs general audio encoders?

# Key References

Essential papers and resources for the BirdCLEF 2026 research. Organized by relevance to current research direction.

> **Last updated**: January 2026

---

## Core Architecture Papers

| Paper | Year | ArXiv | Why It Matters |
|-------|------|-------|----------------|
| **BEATs: Audio Pre-Training with Acoustic Tokenizers** | 2023 | [2212.09058](https://arxiv.org/abs/2212.09058) | Iterative tokenizer training, semantic audio vocabulary. Best for frozen features (95.8% ESC-50 linear probe). See [ssl-pretraining-objectives.md](ssl-pretraining-objectives.md) |
| **EAT: Self-Supervised Pre-Training with Efficient Audio Transformer** | 2024 | [2401.03497](https://arxiv.org/abs/2401.03497) | 15× more efficient than BEATs, but requires finetuning (77.9% vs 95.9% linear vs finetuned). See [ssl-pretraining-objectives.md](ssl-pretraining-objectives.md) |
| **OpenBEATs: A Fully Open-Source General-Purpose Audio Encoder** | 2025 | [2507.14129](https://arxiv.org/abs/2507.14129) | Scaled BEATs (300M), SOTA on bioacoustics, fully open |
| **AudioMAE: Masked Autoencoders that Listen** | 2022 | [2207.06405](https://arxiv.org/abs/2207.06405) | Reconstruction-based SSL, simpler but less semantic |
| **HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction** | 2021 | [2106.07447](https://arxiv.org/abs/2106.07447) | Iterative clustering targets, foundational for discrete audio SSL |
| **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech** | 2020 | [2006.11477](https://arxiv.org/abs/2006.11477) | Contrastive SSL with quantization, foundational architecture |
| **data2vec 2.0: Highly Efficient Self-Supervised Learning** | 2023 | [2212.07525](https://arxiv.org/abs/2212.07525) | 16× more efficient multi-modal SSL |

## Bioacoustics-Specific

| Paper | Year | ArXiv | Why It Matters |
|-------|------|-------|----------------|
| **Foundation Models for Bioacoustics – A Comparative Review** | 2025 | [2508.01277](https://arxiv.org/abs/2508.01277) | Comprehensive benchmark of all audio encoders on BEANS/BirdSet |
| **NatureLM-audio: Audio-Language Foundation Model for Bioacoustics** | 2025 | [2411.07186](https://arxiv.org/abs/2411.07186) | First bioacoustics LLM, SOTA zero-shot, uses BEATs encoder |
| **BioLingual: Transferable Models for Bioacoustics with Human Language** | 2023 | [2308.04978](https://arxiv.org/abs/2308.04978) | CLIP-style language-audio alignment for zero-shot classification |
| **ProtoCLR: Prototypical Contrastive Learning for Bird Sound** | 2024 | [2409.08589](https://arxiv.org/abs/2409.08589) | Domain-invariant representations, focal→soundscape transfer |
| **BirdSet: A Multi-Task Benchmark for Avian Bioacoustics** | 2024 | [2403.10380](https://arxiv.org/abs/2403.10380) | Standard evaluation benchmark, XCL/XCM datasets |
| **AVES: Animal Vocalization Encoder based on Self-Supervision** | 2023 | [2210.14493](https://arxiv.org/abs/2210.14493) | Self-supervised model for animal vocalizations |
| **BEANS: The Benchmark of Animal Sounds** | 2023 | [2210.12300](https://arxiv.org/abs/2210.12300) | Evaluation benchmark for bioacoustic models |
| **Biodenoising: Removing Noise from Bioacoustic Recordings** | 2024 | [2410.03427](https://arxiv.org/abs/2410.03427) | Earth Species Project denoising for wildlife audio |
| **State Space Models for Bioacoustics** | 2025 | [2512.03563](https://arxiv.org/abs/2512.03563) | Mamba vs Transformers comparison for bioacoustics |

## Tokenization & Codecs

| Paper | Year | ArXiv | Why It Matters |
|-------|------|-------|----------------|
| **WavTokenizer: Efficient Acoustic Discrete Codec Tokenizer** | 2024 | [2408.16532](https://arxiv.org/abs/2408.16532) | Single codebook at 40-75 tokens/sec, high quality |
| **BEST-RQ: BERT-based Speech pre-Training with Random-projection Quantizer** | 2022 | [2202.01855](https://arxiv.org/abs/2202.01855) | Random projection tokenization, simpler but less semantic |
| **Open Implementation of BEST-RQ** | 2024 | [2405.04296](https://arxiv.org/abs/2405.04296) | Open reproduction with training details |
| **Codec2Vec: Self-Supervised Speech Representation Learning** | 2024 | [2511.16639](https://arxiv.org/abs/2511.16639) | Discrete codec input, 2.3× training speedup |
| **EnCodec: High Fidelity Neural Audio Compression** | 2022 | [2210.13438](https://arxiv.org/abs/2210.13438) | Foundational RVQ codec, optimized for reconstruction |
| **SpeechTokenizer: Unified Speech Tokenizer for Speech LMs** | 2023 | [2308.16692](https://arxiv.org/abs/2308.16692) | Semantic + acoustic token separation |
| **SoundStream: An End-to-End Neural Audio Codec** | 2021 | [2107.03312](https://arxiv.org/abs/2107.03312) | Google's foundational codec, influenced EnCodec |
| **DAC: Descript Audio Codec** | 2023 | [2306.06546](https://arxiv.org/abs/2306.06546) | High-quality codec used in Codec2Vec |
| **DinoSR: Self-Distillation and Online Clustering for Speech** | 2023 | [2305.10005](https://arxiv.org/abs/2305.10005) | Online clustering alternative to HuBERT |

## Efficient Architectures & Distillation

| Paper | Year | ArXiv | Why It Matters |
|-------|------|-------|----------------|
| **MLP-Mixer: An all-MLP Architecture for Vision** | 2021 | [2105.01601](https://arxiv.org/abs/2105.01601) | O(n) complexity alternative to transformers |
| **ASM: Audio Spectrogram Mixer** | 2024 | [2401.11102](https://arxiv.org/abs/2401.11102) | MLP-Mixer applied to audio spectrograms |
| **ScaleKD: Strong Vision Transformers as Teachers** | 2024 | [2411.06786](https://arxiv.org/abs/2411.06786) | ViT→MLP-Mixer distillation, student beats larger models |
| **Spatial-Channel Token Distillation for Vision MLPs** | 2022 | [ICML](https://proceedings.mlr.press/v162/li22c.html) | Token-level distillation for Mixer architectures |
| **SSAMBA: Self-Supervised Audio with Mamba State Space Model** | 2024 | [2405.11831](https://arxiv.org/abs/2405.11831) | Linear complexity, 92% faster inference |
| **Samba-ASR: Speech Recognition with State-Space Models** | 2025 | [2501.02832](https://arxiv.org/abs/2501.02832) | Mamba for ASR, cross-attention mechanism |
| **EfficientNet: Rethinking Model Scaling** | 2019 | [1905.11946](https://arxiv.org/abs/1905.11946) | Compound scaling, backbone for BirdNET/Perch |
| **model2vec: Distill a Small Fast Model from any Sentence Transformer** | 2024 | [GitHub](https://github.com/MinishLab/model2vec) | Static embeddings via distillation, 500× speedup |

## BirdCLEF Competition

| Paper | Year | ArXiv/Link | Why It Matters |
|-------|------|------------|----------------|
| **Overview of BirdCLEF 2024** | 2024 | [CEUR-WS](https://ceur-ws.org/Vol-3740/paper-198.pdf) | Competition structure, evaluation metrics |
| **Mind the Domain Gap: Bioacoustic Sound Event Detection** | 2024 | [2403.18638](https://arxiv.org/abs/2403.18638) | Focal→soundscape domain shift analysis |
| **Improving Bird Classification with Unsupervised Sound Separation** | 2021 | [2110.03209](https://arxiv.org/abs/2110.03209) | Bird MixIT: TDCN++ separation, 10.5 dB SI-SNR, max-channel classification |
| **Unsupervised Sound Separation Using MixIT** | 2020 | [2006.12701](https://arxiv.org/abs/2006.12701) | Original MixIT: mixture-of-mixtures unsupervised training |
| **RemixIT: Continual Self-Training with Bootstrapped Remixing** | 2022 | [2202.08862](https://arxiv.org/abs/2202.08862) | Self-supervised domain adaptation for separation |
| **Mixture of Mixups for Multi-Label Bioacoustics** | 2024 | [2403.09598](https://arxiv.org/abs/2403.09598) | Mix2 for anuran sounds, co-occurrence-aware mixing proposed |
| **Transfer Learning with Pseudo Multi-Label Birdcall Classification** | 2024 | [2407.06291](https://arxiv.org/abs/2407.06291) | BirdCLEF 2024 approach with pseudo-labels |

## Production Systems

| Paper | Year | ArXiv/Link | Why It Matters |
|-------|------|------------|----------------|
| **Perch 2.0: The Bittern Lesson for Bioacoustics** | 2025 | [2508.04665](https://arxiv.org/abs/2508.04665) | Google's 15k species classifier, EfficientNet-B3 |
| **BirdNET: A deep learning solution for avian diversity monitoring** | 2021 | [BirdNET](https://birdnet.cornell.edu/publications/) | Cornell's production classifier, 6500+ species |
| **Google USM: Scaling ASR Beyond 100 Languages** | 2023 | [2303.01037](https://arxiv.org/abs/2303.01037) | Massive multilingual speech model, uses BEST-RQ |

## Benchmarks & Evaluation

| Paper | Year | ArXiv | Why It Matters |
|-------|------|-------|----------------|
| **SUPERB: Speech Processing Universal PERformance Benchmark** | 2021 | [2105.01051](https://arxiv.org/abs/2105.01051) | Standard benchmark for speech SSL models |
| **Open ASR Leaderboard** | 2025 | [2510.06961](https://arxiv.org/abs/2510.06961) | Reproducible multilingual ASR evaluation |
| **$100K or 100 Days: Trade-offs in Pre-Training** | 2024 | [2410.23261](https://arxiv.org/abs/2410.23261) | Academic compute budget analysis for SSL |

## Additional References

| Paper | Year | ArXiv | Why It Matters |
|-------|------|-------|----------------|
| **AudioLM: A Language Modeling Approach to Audio Generation** | 2022 | [2209.03143](https://arxiv.org/abs/2209.03143) | Semantic vs acoustic token hierarchy |
| **Moshi: Speech-Text Foundation Model for Real-Time Dialogue** | 2024 | [2410.00037](https://arxiv.org/abs/2410.00037) | Full-duplex audio LM with Mimi codec |
| **CLAP: Learning Audio Concepts from Natural Language** | 2022 | [2206.04769](https://arxiv.org/abs/2206.04769) | Contrastive language-audio pretraining |
| **AST: Audio Spectrogram Transformer** | 2021 | [2104.01778](https://arxiv.org/abs/2104.01778) | ViT adapted for audio, ImageNet transfer |
| **PANNs: Large-Scale Pretrained Audio Neural Networks** | 2020 | [1912.10211](https://arxiv.org/abs/1912.10211) | CNN-based AudioSet pretraining |
| **OMAR-RQ: Open Music Audio Representation Model** | 2025 | [2507.03482](https://arxiv.org/abs/2507.03482) | Multi-feature masked prediction for music |

---

## Reading Priority

### Must Read (Current Direction)
1. [EAT paper](https://arxiv.org/abs/2401.03497) - Efficiency gains over BEATs
2. [OpenBEATs paper](https://arxiv.org/abs/2507.14129) - Best open bioacoustics encoder
3. [Foundation Models for Bioacoustics](https://arxiv.org/abs/2508.01277) - Comprehensive comparison

### Should Read
4. [NatureLM-audio](https://arxiv.org/abs/2411.07186) - Bioacoustics LLM approach
5. [ScaleKD](https://arxiv.org/abs/2411.06786) - Distillation methodology
6. [ProtoCLR](https://arxiv.org/abs/2409.08589) - Domain invariance
7. [ASM: Audio Spectrogram Mixer](https://arxiv.org/abs/2401.11102) - MLP-Mixer for audio

### Background (Foundational)
8. [BEATs](https://arxiv.org/abs/2212.09058) - Already covered in 06_beats.md
9. [HuBERT](https://arxiv.org/abs/2106.07447) - Iterative clustering foundation
10. [wav2vec 2.0](https://arxiv.org/abs/2006.11477) - Contrastive SSL foundation
11. [WavTokenizer](https://arxiv.org/abs/2408.16532) - Tokenization options
12. [MLP-Mixer](https://arxiv.org/abs/2105.01601) - Efficient backbone

### Competition-Specific
13. [Mind the Domain Gap](https://arxiv.org/abs/2403.18638) - Domain shift strategies
14. [BirdSet](https://arxiv.org/abs/2403.10380) - Evaluation benchmark
15. [Perch 2.0](https://arxiv.org/abs/2508.04665) - Production baseline

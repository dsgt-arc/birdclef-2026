# Key References

Essential papers and resources for the BirdCLEF 2026 research. Organized by relevance to current research direction.

> **Last updated**: December 2025

---

## Core Architecture Papers

| Paper | Year | ArXiv | Why It Matters |
|-------|------|-------|----------------|
| **BEATs: Audio Pre-Training with Acoustic Tokenizers** | 2023 | [2212.09058](https://arxiv.org/abs/2212.09058) | Iterative tokenizer training, semantic audio vocabulary. Deep dive in [06_beats.md](06_beats.md) |
| **EAT: Self-Supervised Pre-Training with Efficient Audio Transformer** | 2024 | [2401.03497](https://arxiv.org/abs/2401.03497) | 15× more efficient than BEATs, Utterance-Frame Objective |
| **OpenBEATs: A Fully Open-Source General-Purpose Audio Encoder** | 2025 | [2507.14129](https://arxiv.org/abs/2507.14129) | Scaled BEATs (300M), SOTA on bioacoustics, fully open |
| **AudioMAE: Masked Autoencoders that Listen** | 2022 | [2207.06405](https://arxiv.org/abs/2207.06405) | Reconstruction-based SSL, simpler but less semantic |

## Bioacoustics-Specific

| Paper | Year | ArXiv | Why It Matters |
|-------|------|-------|----------------|
| **Foundation Models for Bioacoustics – A Comparative Review** | 2025 | [2508.01277](https://arxiv.org/abs/2508.01277) | Comprehensive benchmark of all audio encoders on BEANS/BirdSet |
| **NatureLM-audio: Audio-Language Foundation Model for Bioacoustics** | 2025 | [2411.07186](https://arxiv.org/abs/2411.07186) | First bioacoustics LLM, SOTA zero-shot, uses BEATs encoder |
| **BioLingual: Transferable Models for Bioacoustics with Human Language** | 2023 | [2308.04978](https://arxiv.org/abs/2308.04978) | CLIP-style language-audio alignment for zero-shot classification |
| **ProtoCLR: Prototypical Contrastive Learning for Bird Sound** | 2024 | [2409.08589](https://arxiv.org/abs/2409.08589) | Domain-invariant representations, focal→soundscape transfer |
| **BirdSet: A Multi-Task Benchmark for Avian Bioacoustics** | 2024 | [2403.10380](https://arxiv.org/abs/2403.10380) | Standard evaluation benchmark, XCL/XCM datasets |

## Tokenization & Codecs

| Paper | Year | ArXiv | Why It Matters |
|-------|------|-------|----------------|
| **WavTokenizer: Efficient Acoustic Discrete Codec Tokenizer** | 2024 | [2408.16532](https://arxiv.org/abs/2408.16532) | Single codebook at 40-75 tokens/sec, high quality |
| **BEST-RQ: BERT-based Speech pre-Training with Random-projection Quantizer** | 2022 | [2202.01855](https://arxiv.org/abs/2202.01855) | Random projection tokenization, simpler but less semantic |
| **Codec2Vec: Self-Supervised Speech Representation Learning** | 2024 | [2511.16639](https://arxiv.org/abs/2511.16639) | Discrete codec input, 2.3× training speedup |
| **EnCodec: High Fidelity Neural Audio Compression** | 2022 | [2210.13438](https://arxiv.org/abs/2210.13438) | Foundational RVQ codec, optimized for reconstruction |

## Efficient Architectures

| Paper | Year | ArXiv | Why It Matters |
|-------|------|-------|----------------|
| **MLP-Mixer: An all-MLP Architecture for Vision** | 2021 | [2105.01601](https://arxiv.org/abs/2105.01601) | O(n) complexity alternative to transformers |
| **ScaleKD: Strong Vision Transformers as Teachers** | 2024 | [2411.06786](https://arxiv.org/abs/2411.06786) | ViT→MLP-Mixer distillation, student beats larger models |
| **SSAMBA: Self-Supervised Audio with Mamba State Space Model** | 2024 | [2405.11831](https://arxiv.org/abs/2405.11831) | Linear complexity, 92% faster inference |
| **model2vec: Distill a Small Fast Model from any Sentence Transformer** | 2024 | [model2vec](https://github.com/MinishLab/model2vec) | Static embeddings via distillation, 500× speedup |

## BirdCLEF Competition

| Paper | Year | ArXiv/Link | Why It Matters |
|-------|------|------------|----------------|
| **Overview of BirdCLEF 2024** | 2024 | [CEUR-WS](https://ceur-ws.org/Vol-3740/paper-198.pdf) | Competition structure, evaluation metrics |
| **Mind the Domain Gap: Bioacoustic Sound Event Detection** | 2024 | [2403.18638](https://arxiv.org/abs/2403.18638) | Focal→soundscape domain shift analysis |
| **Improving Bird Classification with Unsupervised Sound Separation** | 2021 | [2110.03209](https://arxiv.org/abs/2110.03209) | MixIT for bird audio denoising |

## Production Systems

| Paper | Year | ArXiv/Link | Why It Matters |
|-------|------|------------|----------------|
| **Perch 2.0: The Bittern Lesson for Bioacoustics** | 2025 | [2508.04665](https://arxiv.org/abs/2508.04665) | Google's 15k species classifier, EfficientNet-B3 |
| **BirdNET: A deep learning solution for avian diversity monitoring** | 2021 | [BirdNET](https://birdnet.cornell.edu/publications/) | Cornell's production classifier, 6500+ species |

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

### Background
7. [BEATs](https://arxiv.org/abs/2212.09058) - Already covered in 06_beats.md
8. [WavTokenizer](https://arxiv.org/abs/2408.16532) - Tokenization options
9. [MLP-Mixer](https://arxiv.org/abs/2105.01601) - Efficient backbone

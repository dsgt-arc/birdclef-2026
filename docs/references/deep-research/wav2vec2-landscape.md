# **The Post-Wav2vec 2.0 Era: The Evolution of Discrete Audio Representation and Generative Foundation Models**

## **1\. Introduction: The Discretization of the Auditory World**

The trajectory of self-supervised learning (SSL) in audio processing has fundamentally shifted from the continuous domain to the discrete. For years, the field was defined by the **wav2vec 2.0** paradigm, a seminal framework that demonstrated the viability of learning robust speech representations from raw audio waveforms without labeled data.1 Wav2vec 2.0 utilized a contrastive task over quantized latent representations, effectively proving that a model could "learn to listen" by distinguishing true quantized futures from distractors. However, as the field moves through 2024 and into 2025, the architectural consensus has fractured and reformed around a new set of principles that address the inherent inefficiencies of that original design.
The modern "take" on wav2vec 2.0 is no longer about refining the contrastive loss or tweaking the convolutional feature extractor. Instead, it is about a radical decoupling of **acoustic modeling** (signal reconstruction) from **semantic modeling** (language understanding), facilitated by the rise of high-fidelity **Neural Audio Codecs (NACs)** and the shift toward **Generative Audio Language Models (ALMs)**.
This report provides an exhaustive analysis of this new landscape. We explore how the monolithic end-to-end pre-training of wav2vec 2.0 has been superseded by modular pipelines like **Codec2Vec**, which utilize offline tokenization to reduce storage and compute requirements by orders of magnitude.3 We analyze the transition from learned, unstable codebooks (Gumbel-Softmax) to fixed, random projections (**BEST-RQ**) and sophisticated single-quantizer codecs (**WavTokenizer**) that allow Large Language Models (LLMs) to ingest audio as natively as text.4 Furthermore, we examine the architectural rebellion against the Transformer itself, with **State Space Models (Mamba)** offering linear-complexity alternatives for long-form audio understanding.6
The following analysis dissects these technologies not merely as isolated papers, but as cohesive components of a new "Audio Foundation Model" stack that is currently redefining the state of the art.

## ---

**2\. Theoretical Framework: Deconstructing the Wav2vec 2.0 Legacy**

To understand the innovation of 2025, one must first rigorously deconstruct the limitations of the wav2vec 2.0 architecture that necessitated these changes. Wav2vec 2.0 was built upon three pillars: a Convolutional Neural Network (CNN) feature extractor that processed raw waveforms; a Transformer context network; and a Quantization module that discretized the CNN outputs to serve as targets for a contrastive objective.7

### **2.1 The Continuous Input Bottleneck**

Wav2vec 2.0 and its immediate successors (like WavLM) operate on **continuous raw waveforms**. This design choice imposes a massive computational tax. The model must allocate a significant portion of its parameter budget to the local feature extractor (the CNN frontend) to convert high-frequency samples (e.g., 16,000 samples per second) into manageable frame embeddings (e.g., 50 frames per second).

* **Redundancy:** In the era of "Foundation Models," retraining this feature extractor for every new model is computationally wasteful.
* **Storage:** Training on massive datasets (like 60,000+ hours of Libri-Light or 12 million hours of YouTube audio) requires storing and streaming distinct float32 waveforms, creating massive I/O bottlenecks.8

**Modern Solution:** The move to **Discrete Codec Inputs**. By tokenizing the audio *offline* using a frozen, high-quality neural codec (like EnCodec or DAC), the input data is compressed into integers. This reduces storage requirements by approximately **16.5x** and allows the entire pre-training dataset to potentially fit in RAM, accelerating training throughput by **2.3x**.3

### **2.2 The Quantization Instability**

Wav2vec 2.0 utilized a **Gumbel-Softmax** distribution to differentiably select discrete codebook entries during training. This "online" quantization is notoriously unstable.

* **Codebook Collapse:** The model often converges to using only a tiny subset of the available codebook vectors, requiring complex heuristics (like diversity penalties or entropy losses) to force the model to explore the full latent space.10
* **Entanglement:** The resulting discrete units were a messy mix of acoustic information (speaker identity, channel noise) and linguistic content (phonemes), making them suboptimal for pure generation or pure understanding tasks.

**Modern Solution:** The adoption of **Pre-trained Neural Audio Codecs (NACs)** or **Random Projections (BEST-RQ)**. By separating the training of the tokenizer from the training of the foundation model, researchers ensure high codebook utilization and stable targets without auxiliary loss functions.10

### **2.3 The Contrastive vs. Generative Shift**

Wav2vec 2.0 used a **Contrastive Loss** ($L\_c$), requiring the model to distinguish the true quantized representation of a masked time step from a set of distractors sampled from the same utterance.

* **Limitation:** Contrastive learning is computationally expensive (requiring large negative sample batches) and does not inherently teach the model to *reconstruct* the data, limiting its generative capabilities (TTS).

**Modern Solution:** A shift toward **Masked Prediction (BERT-style)** and **Causal Language Modeling (GPT-style)**. Models like **Codec2Vec** and **Moshi** treat audio tokens as discrete vocabulary items (0 to $N$) and simply maximize the likelihood of the correct token, unifying the objectives of text and audio modeling.8

## ---

**3\. The Codec-as-Input Paradigm: Codec2Vec and the Offline Shift**

The most direct "modern take" on the wav2vec 2.0 philosophy is **Codec2Vec**.3 It accepts the premise of masked prediction but fundamentally alters the input modality.

### **3.1 Architectural Decoupling**

Codec2Vec proposes a pipeline where the "Feature Extractor" is no longer part of the learning loop. Instead, it utilizes an off-the-shelf neural audio codec (specifically the **Descript Audio Codec (DAC)** or **EnCodec**) to process the audio offline.

| Component | Wav2vec 2.0 / WavLM | Codec2Vec |
| :---- | :---- | :---- |
| **Input** | Raw Waveform (Float32) | Discrete Codec Units (Integers) |
| **Feature Extraction** | 7-layer CNN (Learned Online) | Frozen Neural Codec (Offline) |
| **Tokenizer** | Gumbel-Softmax (Learned) | DAC / EnCodec (Pre-trained) |
| **Target** | Quantized Latent Vector | Codebook Indices / Cluster IDs |
| **Learning Objective** | Contrastive ($L\_c$) | Masked Prediction (Cross-Entropy) |

This decoupling has profound implications for scalability. In wav2vec 2.0, the CNN context window is fixed. In Codec2Vec, the "frame rate" is determined by the codec (e.g., 50 Hz). Since the input is just a sequence of integers, the model architecture becomes identical to a standard text-based BERT or RoBERTa, allowing for the direct transfer of optimizations from the NLP domain (e.g., FlashAttention, aggressive quantization).14

### **3.2 Target Derivation Strategies**

A critical finding in the Codec2Vec research is that while the *input* should be discrete codec units, the *prediction target* does not necessarily have to be the same unit. The framework explores three distinct strategies to optimize the semantic richness of the representation 9:

1. Reconstruction-based Target:
   The model is tasked with predicting the exact codebook index of the masked frames.
   * *Pros:* Simple implementation; requires no additional clustering steps.
   * *Cons:* Codec units often contain high-frequency acoustic noise irrelevant to speech understanding. Forcing the model to predict exact acoustic details can waste capacity on non-semantic features.
2. Iterative Clustering (HuBERT-style):
   Inspired by HuBERT 1, this strategy involves training a generation of the model, extracting latent representations, performing K-means clustering on these latents to create new "pseudo-labels," and then retraining the model to predict these new cluster IDs.
   * *Insight:* This process "distills" the representation. As the model trains, its latent space organizes phonetically. Clustering these latents creates targets that are more aligned with phonemes than the raw codec units, improving downstream ASR performance significantly.9
3. Online Clustering (DinoSR-style):
   This method utilizes a Teacher-Student architecture. A "Teacher" network (an exponential moving average of the Student) produces the targets online.
   * *Mechanism:* The teacher produces a centered, sharpened distribution over prototypes. The student tries to match this distribution for the masked tokens.
   * *Performance:* Codec2Vec with online clustering achieves results competitive with state-of-the-art continuous models on the SUPERB benchmark, proving that the raw waveform is not strictly necessary for learning high-quality speech representations.8

### **3.3 Efficiency Gains**

The empirical results of Codec2Vec highlight the efficiency advantage of discrete inputs. On the 960-hour LibriSpeech dataset, storing the raw waveforms requires approximately **60 GB**. Storing the compressed codec units requires only **3.6 GB**. This **16.5x reduction** allows the entire dataset to reside in GPU memory or system RAM, eliminating the disk I/O bottleneck that often throttles large-scale audio training.3 Furthermore, the removal of the CNN forward/backward pass accelerates the pre-training wall-clock time by **2.3x**, enabling rapid iteration cycles for researchers.13

## ---

**4\. Advanced Tokenization: The Search for the "Perfect" Audio Token**

If the input to the modern audio transformer is a token, the quality of the model is inextricably linked to the quality of the tokenizer. Wav2vec 2.0's learned quantizer was a "weak" tokenizer. The 2025 landscape is defined by "strong" tokenizers that balance three competing objectives: **Compression Rate**, **Reconstruction Quality**, and **Semantic Disentanglement**.

### **4.1 WavTokenizer: The Single-Quantizer Breakthrough**

Most neural audio codecs (like EnCodec) rely on **Residual Vector Quantization (RVQ)**. To achieve high fidelity, they use multiple codebooks (e.g., 8 layers), where each subsequent layer corrects the errors of the previous one. This creates a "multi-stream" input (e.g., 8 tokens per time step), which is cumbersome for Transformer modeling.
**WavTokenizer** 4 represents a major leap forward by achieving high-fidelity reconstruction with a **single codebook** (single quantizer).

#### **4.1.1 The Architecture of Extreme Compression**

WavTokenizer compresses 24kHz audio into just **40 to 75 tokens per second** using a single stream.

* **Expanded VQ Space:** Instead of cascading small codebooks (e.g., size 1024), it utilizes a broader, semantically richer singular vector quantization space (size 4096 or larger).
* **Inverse Fourier Transform Decoder:** The critical innovation is in the decoder. Standard codecs use convolutional upsampling, which struggles with high-frequency phase coherence. WavTokenizer integrates an **inverse Short-Time Fourier Transform (iSTFT)** mechanism directly into the decoder network (inspired by the **Vocos** vocoder).
* **Mechanism:** The decoder predicts complex spectral coefficients, which are then transformed into the waveform via iSTFT. This inductive bias allows the model to reconstruct audio with fewer discrete codes because the "physics" of the signal generation is handled by the DSP transformation, not learned entirely from scratch.15

#### **4.1.2 Performance vs. EnCodec**

Comparative analysis shows that WavTokenizer at **75 tokens/s** achieves a UTMOS (subjective quality score) of **\~4.05**, outperforming EnCodec at significantly higher bitrates.15 For self-supervised learning, this is transformative: it allows "Audio Language Models" to model audio as a simple, flat sequence of integers (just like text), drastically simplifying the architecture compared to the hierarchical modeling required for 8-layer RVQ codes.

### **4.2 SpeechTokenizer: Hierarchical Information Disentanglement**

While WavTokenizer focuses on compression, **SpeechTokenizer** 17 tackles the problem of "Information Entanglement." In a standard codec, the codes contain a mix of linguistic content (what is said) and paralinguistic content (timbre, noise). For tasks like ASR, the paralinguistic information is noise. For TTS, it is essential.

#### **4.2.1 The Distillation Mechanism**

SpeechTokenizer employs a **Residual Vector Quantization (RVQ)** structure but with a twist:

* **Layer 1 (Semantic):** The first quantizer is trained with a **distillation loss** from a pre-trained semantic teacher (like HuBERT or WavLM). It is forced to align its codebook vectors with the semantic representations of the teacher.19
* **Layers 2-N (Acoustic):** The subsequent layers are trained with standard reconstruction losses. They capture the "residuals"—the acoustic details that the semantic teacher ignored (speaker identity, background noise).

#### **4.2.2 Unified Modeling Implications**

This architecture provides a "best of both worlds" scenario.

* **For ASR:** The model can attend only to the tokens from Layer 1, effectively filtering out noise and speaker variance.
* **For TTS/Generation:** The model generates Layer 1 (content) first, then conditionally generates Layers 2-N (style), enabling zero-shot voice conversion and highly controllable synthesis.17

### **4.3 Google’s BEST-RQ: The "No-Tokenizer" Tokenizer**

At the other end of the spectrum lies Google's **BEST-RQ** (BERT-based Speech pre-Training with Random-projection Quantizer).10 Used in the massive **Universal Speech Model (USM)**, this approach rejects the idea of "learning" a tokenizer entirely.

#### **4.3.1 The Random Projection Theory**

BEST-RQ operates on a simple premise: a fixed, random projection of the input feature space preserves the geometric relationships of the data without the risk of collapse associated with learned codebooks.

1. **Project:** Input Mel-filterbanks are multiplied by a frozen, randomly initialized matrix $A$.
2. **Quantize:** The resulting vector is matched to the nearest neighbor in a frozen, randomly initialized codebook $C$.
3. **Target:** The index of this neighbor becomes the target for the masked prediction task.

#### **4.3.2 Why It Scales**

While counter-intuitive, this method works exceptionally well at scale (e.g., 2 billion parameters, 300 languages). Because the tokenizer is frozen and random, it provides a consistent, unbiased target. The encoder is forced to learn robust internal representations to solve the projection task. Unlike wav2vec 2.0, where the quantizer and encoder evolve together (often chasing a moving target), BEST-RQ provides a stationary target, leading to greater training stability and simpler hyperparameter tuning.11

## ---

**5\. Architectural Innovations: Beyond the Quadratic Transformer**

Wav2vec 2.0 utilized a standard Transformer encoder. However, audio sequences are extremely long compared to text. A 10-second clip at 50 Hz results in 500 tokens. A 30-second clip results in 1500\. The Transformer's self-attention mechanism scales quadratically ($O(N^2)$), making the processing of minute-long audio segments prohibitively expensive. Two major solutions have emerged in 2025: **Efficient Audio Transformers (EAT)** and **State Space Models (Mamba)**.

### **5.1 Efficient Audio Transformer (EAT): Sparse Processing**

**EAT** 21 adapts the **Masked Autoencoder (MAE)** philosophy to audio. The core insight is **sparsity**.

* **Sparse Encoding:** In standard wav2vec 2.0, the Transformer processes *all* tokens, even the masked ones (replaced by a mask token). EAT, like MAE for images, *discards* the masked tokens at the encoder input. The encoder only processes the visible patches (roughly 20-30% of the sequence).
* **Efficiency:** This reduces the sequence length $N$ significantly, and since attention is $O(N^2)$, the computational savings are massive (up to **15x faster** training).21
* **Utterance-Frame Objective (UFO):** EAT introduces a specialized loss function. It combines a **Frame-level loss** (reconstructing the missing patches) with an **Utterance-level loss**, where a \`\` token predicts the mean-pooled representation of the entire target sequence. This ensures the model captures both local acoustic textures and global semantic events.21

### **5.2 The Linear Revolution: Mamba and SSAMBA**

The most radical architectural shift is the adoption of **State Space Models (SSMs)**, specifically the **Mamba** architecture.6

#### **5.2.1 The Mathematics of Mamba**

Mamba utilizes a **Selective State Space** mechanism. Unlike Transformers, which keep the entire history in the attention matrix, Mamba compresses context into a recurrent hidden state $h\_t$.

* **Selection Mechanism:** Crucially, the parameters of the state transition are *input-dependent*. This allows the model to "selectively" remember or forget information based on the current input, solving the bottleneck of traditional RNNs while maintaining **linear complexity ($O(N)$)**.
* **Hardware Awareness:** Mamba is designed with a "hardware-aware" parallel scan algorithm, allowing it to train efficiently on GPUs despite its recurrent nature.23

#### **5.2.2 SSAMBA: Performance and Speed**

**SSAMBA (Self-Supervised Audio Mamba)** replaces the Transformer in the standard Masked Spectrogram Modeling task.6

* **Speedup:** For an input sequence of 22k patches (long audio), SSAMBA is **92.7% faster** in inference and uses **95.4% less memory** than a comparable Transformer (SSAST).6
* **Accuracy:** Despite lacking global attention, SSAMBA matches or exceeds Transformer baselines on standard benchmarks like AudioSet and ESC-50, proving that the selective state space is sufficient for capturing long-range audio dependencies.

#### **5.2.3 Samba-ASR and Hybrid Models**

In the ASR domain, **Samba-ASR** 23 demonstrates that Mamba can function as both encoder and decoder. It introduces a **Mamba-cross-connection**, a novel mechanism to allow the decoder Mamba blocks to attend to the encoder outputs (simulating Cross-Attention). This model achieves state-of-the-art results on Librispeech, particularly in long-form transcription where Transformers struggle with context window limits.

## ---

**6\. Generative Foundation Models: The Convergence of Audio and Text**

The ultimate evolution of the wav2vec 2.0 lineage is the "Generative Foundation Model." These models do not just output embeddings; they output discrete audio and text tokens, enabling full-duplex interaction.

### **6.1 Moshi: The Full-Duplex Dialogue Agent**

**Moshi** 12 represents the cutting edge of late 2024/2025. It is an **Audio Language Model (ALM)** that integrates the **Mimi** codec with a LLaMA-style backbone (Helium).

#### **6.1.1 The Inner Monologue**

A key limitation of generating audio tokens directly is that audio contains pauses, non-speech sounds, and prosody that are not strictly linguistic. A model trained only on audio tokens often produces "babble"—speech that sounds realistic but lacks semantic coherence.

* **Mechanism:** Moshi employs an **Inner Monologue**. It predicts time-aligned **text tokens** (representing the semantic thought) *in parallel* with the audio tokens.
* **Effect:** This acts as a "Chain of Thought" for audio. The model "thinks" in text (determining *what* to say) while simultaneously generating audio tokens (determining *how* to say it).12

#### **6.1.2 Parallel Stream Architecture**

Unlike traditional pipelines (ASR \-\> LLM \-\> TTS) which introduce latency, Moshi models the user's input stream and the system's output stream simultaneously.

* **Full Duplex:** It can listen and speak at the same time. This allows for natural conversational behaviors like **backchanneling** (saying "uh-huh" while the user speaks) and **interruption handling**, which are impossible in turn-based wav2vec 2.0 pipelines.27

### **6.2 Google USM: Scaling to the World**

Google's **Universal Speech Model (USM)** 5 focuses on linguistic breadth.

* **Data Scale:** 12 million hours of audio.
* **Language Coverage:** 300+ languages.
* **Text Injection:** USM is not purely self-supervised on audio. It introduces a mandatory **Text Injection** pre-training stage, where the model is trained on unpaired text data (converted to phonemes or tokens). This aligns the audio latent space with the text latent space, significantly improving performance on low-resource languages where paired audio-text data is scarce.20

## ---

**7\. Comparative Analysis and Benchmarks**

To quantify these advancements, we examine performance on the **SUPERB** benchmark and the **Open ASR Leaderboard**.

### **7.1 Accuracy vs. Efficiency Trade-offs**

| Model Architecture | Representative Model | ASR Accuracy (WER) | Inference Speed | Long-Form Capability |
| :---- | :---- | :---- | :---- | :---- |
| **Conformer \+ LLM Decoder** | NVIDIA Canary, USM | **Highest** (SOTA) | Low (Slow Decoding) | High (with chunking) |
| **Discrete Input Transformer** | Codec2Vec, WavLM | High | Medium | Medium |
| **Mamba (SSM)** | SSAMBA, Samba-ASR | Competitive | **Very High** (Linear) | **Excellent** |
| **CTC-based** | NVIDIA Parakeet | Moderate | **Highest** (Non-AR) | Low |

* **Observation:** There is currently a bifurcation. For maximum accuracy (especially in English), **Conformer encoders** paired with massive **LLM decoders** are dominant on the Open ASR Leaderboard.29 However, for efficiency and low-latency applications, **Mamba-based** and **Discrete-input** models are rapidly closing the gap.

### **7.2 Tokenizer Performance Comparison**

Evaluating the tokenizers themselves (Code-SUPERB benchmark) reveals critical differences 30:

* **WavTokenizer (75Hz)** achieves a UTMOS score of **4.05**, significantly higher than **EnCodec (24kHz)** which scores **\~3.8**. This confirms that the single-quantizer architecture with iSTFT decoding is superior for generative modeling tasks.
* **SpeechTokenizer** demonstrates superior performance in **zero-shot TTS**, confirming that its disentangled semantic tokens provide a better control signal than raw acoustic tokens.17

### **7.3 Generalization (SUPERB)**

On the SUPERB benchmark (which evaluates frozen representations):

* **WavLM** remains the king of **paralinguistic tasks** (Emotion Recognition, Speaker ID) due to its noisy/denoising pre-training strategy.
* **HuBERT** and **Codec2Vec** (with iterative clustering) excel at **semantic tasks** (ASR, Intent Classification).9
* **Implication:** If the goal is ASR, discrete clustering (HuBERT/Codec2Vec) is optimal. If the goal is emotion/speaker analysis, continuous noisy pre-training (WavLM) is still superior.

## ---

**8\. Training Recipes and Practical Implementation**

For researchers and practitioners implementing these models in 2025, the training recipes have evolved significantly from the original wav2vec 2.0 setup.

### **8.1 Data Scale and Augmentation**

* **Scale:** The "standard" pre-training set has moved from LibriSpeech (960h) to **Libri-Light (60k hours)** or **YouTube-based datasets (1M+ hours)**. The simplified objectives (BEST-RQ, Masked Prediction) allow for stable training at this scale.
* **Noisy Training:** Following WavLM's success, modern recipes almost always include **utterance mixing**, noise injection, and overlapped speech during pre-training to force the model to perform source separation implicitly.7

### **8.2 Masking Strategies**

* **Block Masking:** EAT and Codec2Vec utilize aggressive block masking (masking 70-80% of the input in large contiguous blocks). This is distinct from the random token masking of BERT. Audio contains high local redundancy; masking individual tokens is too easy. Large blocks force the model to use global context.21
* **Span Masking:** For generative models (Moshi), causal masking (predicting the next token) is used instead of bidirectional masking.

### **8.3 Hardware Requirements**

* **Discrete Inputs:** Allow for training on consumer-grade GPUs (e.g., 4x A100s or even 4090s for smaller models) because the heavy lifting (tokenization) is done once on CPU/GPU before training begins.8
* **Mamba:** Requires kernels optimized for the specific GPU architecture (e.g., Triton kernels) to realize the theoretical speedups. Standard PyTorch implementations of SSMs can be slow without these fused kernels.23

## ---

**9\. Future Outlook and Conclusion**

The state of the art in self-supervised audio transformers has matured into a diverse ecosystem. The monolithic "wav2vec 2.0" model has evolved into a modular stack:

1. **Tokenizer:** A high-compression, single-stream neural codec (WavTokenizer, Mimi).
2. **Backbone:** A linear-complexity sequence model (Mamba) or a sparse Transformer (EAT).
3. **Objective:** A generative, multimodal loss (Joint Audio-Text Modeling).

**Key Trends for 2025-2026:**

* **The End of the "Tokenizer"?** While discrete tokens are currently dominant, end-to-end continuous generative models (Flow Matching) are beginning to emerge, challenging the need for discrete codes entirely in favor of continuous latent diffusion.33
* **Multimodal Native:** Models will increasingly be "Omni," processing audio, text, and video in a shared embedding space (like Gemini 2.0 or GPT-4o), making "audio-only" SSL a specialized niche.
* **On-Device Foundation Models:** The efficiency gains from Mamba and Codec2Vec are paving the way for high-performance foundation models that run locally on mobile devices, enabling private, low-latency voice intelligence.34

Recommendations:
For practitioners seeking to build upon wav2vec 2.0:

* **Adopt Discrete Inputs:** Use Codec2Vec or WavTokenizer to immediately gain 2-3x training speedups.
* **Investigate Mamba:** For long-form audio applications (meetings, podcasts), Mamba-based backbones offer a decisive advantage over Transformers.
* **Prioritize Data Scale:** The move to fixed targets (BEST-RQ) or offline targets (Codec2Vec) makes it easier than ever to scale to millions of hours of data. Prioritize dataset size over architectural tweaking.

The era of "learning to listen" via simple contrastive loss is over; the era of "listening to speak" via generative, discrete, and efficient modeling has begun.

#### **Works cited**

1. State Space Models for Bioacoustics: A comparative Evaluation with Transformers \- arXiv, accessed December 5, 2025, [https://arxiv.org/html/2512.03563v1](https://arxiv.org/html/2512.03563v1)
2. SUPERB: Speech processing Universal PERformance Benchmark \- alphaXiv, accessed December 5, 2025, [https://www.alphaxiv.org/overview/2105.01051v4](https://www.alphaxiv.org/overview/2105.01051v4)
3. Codec2Vec: Self-Supervised Speech Representation Learning Using Neural Speech Codecs \- arXiv, accessed December 5, 2025, [https://arxiv.org/html/2511.16639v1](https://arxiv.org/html/2511.16639v1)
4. WAVTOKENIZER: AN EFFICIENT ACOUSTIC DISCRETE CODEC TOKENIZER FOR AUDIO LANGUAGE MODELING \- ICLR Proceedings, accessed December 5, 2025, [https://proceedings.iclr.cc/paper\_files/paper/2025/file/ea1f5f0878d43ff4fb8bf64ef4a2326c-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/ea1f5f0878d43ff4fb8bf64ef4a2326c-Paper-Conference.pdf)
5. Universal Speech Model (USM): State-of-the-art speech AI for 100+ languages, accessed December 5, 2025, [https://research.google/blog/universal-speech-model-usm-state-of-the-art-speech-ai-for-100-languages/](https://research.google/blog/universal-speech-model-usm-state-of-the-art-speech-ai-for-100-languages/)
6. SSAMBA: Self-Supervised Audio Representation Learning with Mamba State Space Model, accessed December 5, 2025, [https://arxiv.org/html/2405.11831v2](https://arxiv.org/html/2405.11831v2)
7. Self-Supervised Audio Models \- Emergent Mind, accessed December 5, 2025, [https://www.emergentmind.com/topics/self-supervised-audio-models](https://www.emergentmind.com/topics/self-supervised-audio-models)
8. (PDF) Codec2Vec: Self-Supervised Speech Representation Learning Using Neural Speech Codecs \- ResearchGate, accessed December 5, 2025, [https://www.researchgate.net/publication/397824312\_Codec2Vec\_Self-Supervised\_Speech\_Representation\_Learning\_Using\_Neural\_Speech\_Codecs](https://www.researchgate.net/publication/397824312_Codec2Vec_Self-Supervised_Speech_Representation_Learning_Using_Neural_Speech_Codecs)
9. \[Literature Review\] Codec2Vec: Self-Supervised Speech Representation Learning Using Neural Speech Codecs \- Moonlight, accessed December 5, 2025, [https://www.themoonlight.io/en/review/codec2vec-self-supervised-speech-representation-learning-using-neural-speech-codecs](https://www.themoonlight.io/en/review/codec2vec-self-supervised-speech-representation-learning-using-neural-speech-codecs)
10. Open Implementation and Study of BEST-RQ for Speech Processing \- arXiv, accessed December 5, 2025, [https://arxiv.org/html/2405.04296v1](https://arxiv.org/html/2405.04296v1)
11. Optimized Self-supervised Training with BEST-RQ for Speech Recognition \- arXiv, accessed December 5, 2025, [https://arxiv.org/html/2501.16131v1](https://arxiv.org/html/2501.16131v1)
12. Moshi: Unified Speech-Text Dialogue Model, accessed December 5, 2025, [https://www.emergentmind.com/topics/moshi-a-speech-text-foundation-model](https://www.emergentmind.com/topics/moshi-a-speech-text-foundation-model)
13. \[2511.16639\] Codec2Vec: Self-Supervised Speech Representation Learning Using Neural Speech Codecs \- arXiv, accessed December 5, 2025, [https://arxiv.org/abs/2511.16639](https://arxiv.org/abs/2511.16639)
14. Codec2Vec: Self-Supervised Speech Representation Learning Using Neural Speech Codecs \- arXiv, accessed December 5, 2025, [https://arxiv.org/pdf/2511.16639](https://arxiv.org/pdf/2511.16639)
15. WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling | alphaXiv, accessed December 5, 2025, [https://www.alphaxiv.org/overview/2408.16532v1](https://www.alphaxiv.org/overview/2408.16532v1)
16. wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling, accessed December 5, 2025, [https://arxiv.org/html/2408.16532v1](https://arxiv.org/html/2408.16532v1)
17. SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models \- arXiv, accessed December 5, 2025, [https://arxiv.org/html/2308.16692v2](https://arxiv.org/html/2308.16692v2)
18. SpeechTokenizer \- Dong Zhang, accessed December 5, 2025, [https://0nutation.github.io/SpeechTokenizer.github.io/](https://0nutation.github.io/SpeechTokenizer.github.io/)
19. SPEECHTOKENIZER: UNIFIED SPEECH TOKENIZER FOR SPEECH LANGUAGE MODELS \- OpenReview, accessed December 5, 2025, [https://openreview.net/pdf?id=AF9Q8Vip84](https://openreview.net/pdf?id=AF9Q8Vip84)
20. Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages \- arXiv, accessed December 5, 2025, [https://arxiv.org/pdf/2303.01037](https://arxiv.org/pdf/2303.01037)
21. EAT: Self-Supervised Pre-Training with Efficient Audio Transformer, accessed December 5, 2025, [https://arxiv.org/abs/2401.03497](https://arxiv.org/abs/2401.03497)
22. cwx-worst-one/EAT: \[IJCAI 2024\] EAT: Self-Supervised Pre-Training with Efficient Audio Transformer \- GitHub, accessed December 5, 2025, [https://github.com/cwx-worst-one/EAT](https://github.com/cwx-worst-one/EAT)
23. Samba-ASR: State-Of-The-Art Speech Recognition Leveraging Structured State-Space Models \- arXiv, accessed December 5, 2025, [https://arxiv.org/html/2501.02832v2](https://arxiv.org/html/2501.02832v2)
24. \[2501.02832\] Samba-ASR: State-Of-The-Art Speech Recognition Leveraging Structured State-Space Models \- arXiv, accessed December 5, 2025, [https://arxiv.org/abs/2501.02832](https://arxiv.org/abs/2501.02832)
25. SSAMBA: Self-Supervised Audio Representation Learning With Mamba State Space Model \- IEEE Xplore, accessed December 5, 2025, [https://ieeexplore.ieee.org/iel8/10830790/10830793/10832304.pdf](https://ieeexplore.ieee.org/iel8/10830790/10830793/10832304.pdf)
26. Moshi \- Hugging Face, accessed December 5, 2025, [https://huggingface.co/docs/transformers/model\_doc/moshi](https://huggingface.co/docs/transformers/model_doc/moshi)
27. Managed Inference for Moshi \- Scaleway Labs, accessed December 5, 2025, [https://labs.scaleway.com/en/moshi/](https://labs.scaleway.com/en/moshi/)
28. Chirp 3 Transcription: Enhanced multilingual accuracy | Cloud Speech-to-Text, accessed December 5, 2025, [https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3](https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3)
29. Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation \- arXiv, accessed December 5, 2025, [https://arxiv.org/html/2510.06961v2](https://arxiv.org/html/2510.06961v2)
30. WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling | OpenReview, accessed December 5, 2025, [https://openreview.net/forum?id=yBlVlS2Fd9](https://openreview.net/forum?id=yBlVlS2Fd9)
31. Layer-wise benchmarking of wav2vec 2.0 Large, HuBERT Large and Data2vec... | Download Scientific Diagram \- ResearchGate, accessed December 5, 2025, [https://www.researchgate.net/figure/Layer-wise-benchmarking-of-wav2vec-20-Large-HuBERT-Large-and-Data2vec-Large-on-5-tasks\_fig3\_379882278](https://www.researchgate.net/figure/Layer-wise-benchmarking-of-wav2vec-20-Large-HuBERT-Large-and-Data2vec-Large-on-5-tasks_fig3_379882278)
32. vec2wav 2.0: Advancing Voice Conversion via Discrete Token Vocoders \- ResearchGate, accessed December 5, 2025, [https://www.researchgate.net/publication/383745256\_vec2wav\_20\_Advancing\_Voice\_Conversion\_via\_Discrete\_Token\_Vocoders](https://www.researchgate.net/publication/383745256_vec2wav_20_Advancing_Voice_Conversion_via_Discrete_Token_Vocoders)
33. UniWav: Towards Unified Pre-training for Speech Representation Learning and Generation, accessed December 5, 2025, [https://openreview.net/forum?id=yj9lLwMjnE](https://openreview.net/forum?id=yj9lLwMjnE)
34. ARE MAMBA-BASED AUDIO FOUNDATION MODELS THE BEST FIT FOR NON-VERBAL EMOTION RECOGNITION? \- Eusipco 2025, accessed December 5, 2025, [https://eusipco2025.org/wp-content/uploads/pdfs/0000501.pdf](https://eusipco2025.org/wp-content/uploads/pdfs/0000501.pdf)

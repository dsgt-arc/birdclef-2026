# **BEATs: Mechanisms of Acoustic Tokenization and Iterative Self-Supervised Learning in Audio Representation**

## **Executive Summary**

The advancement of Self-Supervised Learning (SSL) in the audio domain has long been impeded by the fundamental "continuous-discrete gap." Unlike Natural Language Processing (NLP), where the atomic units of information—words or sub-words—are inherently discrete and semantic, audio signals are continuous, high-dimensional, and laden with stochastic noise. This discrepancy has historically forced researchers to rely on reconstruction-based objectives, such as Masked Autoencoders, or contrastive learning paradigms, both of which suffer from significant inefficiencies. Reconstruction models waste capacity modeling imperceptible acoustic details, while contrastive models require massive negative sampling and large batch sizes to prevent collapse.  
The Bidirectional Encoder representation from Audio Transformers (BEATs) framework represents a paradigm shift in this landscape. By introducing an iterative, self-supervised learning protocol that enforces discrete tokenization on continuous audio, BEATs bridges the gap between audio analysis and the masked language modeling objectives that have revolutionized NLP.1 This report provides an exhaustive technical analysis of the BEATs framework, specifically dissecting the iterative tokenizer-transformer training loop that enables its state-of-the-art (SOTA) performance on benchmarks like AudioSet-2M and ESC-50.  
The analysis explores the architectural decisions behind the acoustic tokenizer, the mathematical formulation of the masked prediction objective, and the computational requirements necessary to replicate such high-performance models. Furthermore, we investigate the scalability of BEATs through the lens of recent "OpenBEATs" initiatives, evaluating its efficacy in specialized downstream domains such as bioacoustics, where it outperforms models with vastly larger parameter counts.3 A critical component of this investigation is a comparative analysis against alternative quantization strategies, specifically the Google BEST-RQ method and frozen neural codecs like EnCodec, to isolate the specific contributions of the BEATs iterative protocol to semantic abstraction.

## **1\. The Continuous-Discrete Dilemma in Audio Self-Supervised Learning**

To understand the design choices of BEATs, one must first contextualize the limitations of the paradigms it seeks to replace. The central problem in audio representation learning is the absence of a natural, pre-existing vocabulary. In computer vision, visual concepts can be approximated through clustering or patch-based tokenization; in NLP, tokenizers are deterministic algorithms (e.g., BPE, WordPiece). In audio, however, a "phoneme" equivalent does not exist for general environmental sounds—a category that encompasses everything from a car horn to a bird call or the hum of machinery.1

### **1.1 The Limitations of Reconstruction Architectures**

Prior to the introduction of semantic tokenizers, the dominant approach for vision-like audio processing was the Masked Autoencoder (MAE), exemplified by Audio-MAE.5 These models operate by masking patches of a spectrogram and attempting to reconstruct the raw pixel values of the missing regions. While effective for learning low-level textures, this objective forces the model to allocate significant parameter capacity to modeling stochastic details—the "grain" of the noise floor or the exact phase of a signal—which are often irrelevant to high-level semantic classification.  
The reconstruction loss (Mean Squared Error) creates a misalignment between the pre-training objective and downstream tasks. A model that perfectly reconstructs the wind noise in a recording of a dog barking has achieved a low pre-training loss but has not necessarily learned the concept of "dog." It has merely functioned as a sophisticated signal denoiser. BEATs addresses this by discarding the reconstruction objective entirely in favor of discrete label prediction, hypothesizing that the discretization process itself acts as an information bottleneck that filters out noise and retains only semantic structure.4

### **1.2 The Contrastive Bottleneck**

The alternative to reconstruction has been contrastive learning, popularized by Wav2Vec 2.0 and its successors.6 These models learn to distinguish the true latent representation of a masked time-step from a set of "negative" distractors sampled from other parts of the audio. While successful in speech, contrastive learning in general audio is notoriously unstable. It requires large batch sizes to ensure a diverse set of negatives and is highly sensitive to the sampling strategy. Furthermore, contrastive objectives essentially learn a continuous embedding space without explicit clustering, which can lead to "smeared" representations where distinct acoustic events overlap ambiguously. BEATs circumvents the need for negative sampling by using a fixed (within an iteration) codebook of discrete targets, stabilizing the training dynamics and allowing for a direct cross-entropy loss formulation.7

## **2\. The Iterative Tokenizer-Transformer Training Protocol**

The defining characteristic of BEATs, and its primary contribution to the field, is the Iterative Self-Distillation Protocol. Unlike models that use a fixed, pre-trained codec (like EnCodec) or a static clustering method (like K-means on MFCCs), BEATs evolves its target labels over time. The framework simultaneously optimizes two distinct components: the **Acoustic Tokenizer** and the **Audio SSL Model** (the backbone encoder). This creates a symbiotic feedback loop where improvements in the encoder lead to a better tokenizer, which in turn generates better targets for the next generation of the encoder.1

### **2.1 The Cold Start: Random Projection as a Proto-Tokenizer**

A fundamental challenge in training a tokenizer from scratch is the "cold start" problem. If the encoder has not yet learned good representations, it cannot guide the training of the tokenizer. Conversely, if the tokenizer is untrained, it cannot generate meaningful labels for the encoder.  
BEATs solves this by initializing the first iteration with a **Random Projection Tokenizer**, a technique inspired by the BEST-RQ (BERT-based Speech pre-Training with Random-projection Quantizer) framework.8

* **Mechanism:** For the first iteration, the continuous audio features (Mel-filterbanks) are projected into a lower-dimensional space using a fixed, randomly initialized linear projection matrix $W\_{rand}$.  
* **Vector Quantization:** These projected features are then matched to the nearest neighbor in a set of randomly initialized codebook vectors $V\_{rand}$.  
* **Theoretical Justification:** This approach relies on the Johnson-Lindenstrauss lemma, which posits that random projections can preserve the relative distances (topology) of high-dimensional data in a lower-dimensional space. Even though the tokenizer has learned no semantics, the resulting tokens preserve local acoustic consistency. Similar spectral patches will map to the same random vector, providing sufficiently structured targets for the SSL model to begin learning low-level acoustic features such as edges, formants, and basic textures.8

### **2.2 Iterative Semantic Distillation**

From the second iteration onward, the tokenizer is no longer random. It is actively trained to distill knowledge from the SSL model trained in the previous iteration. This process effectively converts the implicit knowledge of the encoder into an explicit vocabulary of discrete tokens.  
The cycle proceeds as follows:

1. **Teacher Generation:** The Audio SSL model from Iteration $N-1$ is frozen and acts as the "teacher." It processes unmasked audio to produce a sequence of continuous, high-level context representations.  
2. **Tokenizer Training:** A new tokenizer network is trained to approximate these teacher representations. Crucially, this tokenizer includes a quantization bottleneck (codebook). It minimizes the distance between its quantized output and the teacher's continuous output.  
3. **Label Refinement:** Once trained, this new tokenizer processes the entire training dataset, re-labeling every audio patch with a new integer ID.  
4. **Student Training:** A new Audio SSL model (Iteration $N$) is initialized. It is trained to predict these new discrete labels from *masked* input audio.4

This cycle typically repeats for three iterations. Empirical evidence from the BEATs ablation studies suggests that semantic quality saturates after the third iteration, with diminishing returns for subsequent loops. By Iteration 3, the tokens have evolved from representing simple frequency patterns (Iter 1\) to representing complex acoustic events (e.g., "engine idling," "dog barking").1

### **2.3 Convergence and Codebook Evolution**

The convergence of this iterative process relies on the principle of non-decreasing likelihood. In each step, the tokenizer improves by clustering the improved representations of the SSL model, creating cleaner and more semantically grouped targets. The SSL model then improves by training on these cleaner targets.

* **Qualitative Evolution:** In Iteration 1, tokens represent random partitions of the acoustic space (e.g., "high frequency energy"). By Iteration 3, tokens cluster highly semantic concepts. For instance, different recordings of rain (light rain, heavy storm) might initially map to different random tokens, but as the encoder learns the concept of "rain," the distilled tokenizer will map them to the same semantic token.7  
* **Stability Mechanisms:** The use of exponential moving average (EMA) updates for the codebook embeddings during tokenizer training is critical. This prevents the codebook from oscillating rapidly between iterations and ensures that the discrete space stabilizes into a consistent semantic map.2

## **3\. Architecture of the Acoustic Tokenizer**

The acoustic tokenizer in BEATs is not merely a lookup table; it is a learned neural network designed to quantize continuous audio into a finite set of integers. Its architecture is carefully balanced to be expressive enough to capture semantics but constrained enough to force abstraction.

### **3.1 Tokenizer Components**

The tokenizer consists of three primary modules, distinct from the main Audio SSL backbone:

1. **Tokenizer Encoder:** A Transformer-based encoder (typically shallower than the main SSL model) that processes the input spectrogram patches. It converts the input $X$ into latent vectors $Z$.  
2. **Quantizer (Vector Quantization Bottleneck):** A codebook $V \\in \\mathbb{R}^{K \\times D}$ containing $K$ learnable vectors of dimension $D$. This module maps continuous latents $Z$ to discrete indices.  
3. **Tokenizer Estimator (Decoder):** A small Transformer that attempts to reconstruct the *teacher's* output representations given the quantized tokens. This estimator ensures that the tokens retain enough information to reconstruct the semantic content of the teacher's embeddings.2

### **3.2 Codebook Design: The "Goldilocks" Vocabulary**

The design of the codebook is a critical hyperparameter that determines the granularity of the audio model's "language."

* **Vocabulary Size ($K$):** BEATs utilizes a codebook size of **$K=1024$**.4 This is significantly smaller than NLP vocabularies (which range from 30,000 to 50,000 tokens) or recent image generation codebooks (often 16,384).  
* **Theoretical Implication:** The choice of 1024 is deliberate. A smaller codebook forces higher abstraction. If $K$ were too large (e.g., 65,536), the model might assign different tokens to the same sound event recorded in slightly different environments (e.g., "dog bark indoors" vs. "dog bark outdoors"). This would fragment the semantic space. A size of 1024 compels the model to merge acoustically similar but distinct instances into shared semantic categories, functioning as an Information Bottleneck that filters out environmental noise while retaining event identity.4

### **3.3 Quantization Logic and Gradient Estimation**

For an encoded vector $z$, the tokenizer finds the nearest codebook entry $e\_k$ via Euclidean distance:

$$z\_q \= e\_k \\quad \\text{where} \\quad k \= \\arg \\min\_j \\| z \- e\_j \\|\_2$$

Since the argmin operation is non-differentiable, BEATs employs the Straight-Through Estimator (STE) mechanism. During the forward pass, the quantization is applied rigidly. During the backward pass, the gradients flow directly from the estimator (decoder) to the encoder, bypassing the quantization step. This allows the encoder to be updated to produce vectors that are "easy to quantize" (i.e., cluster tightly around the codebook vectors).2

## **4\. The Masked Prediction Objective vs. Alternatives**

The training objective of the main Audio SSL model in BEATs is **Masked Audio Modeling (MAM)** with discrete targets. This aligns the audio pre-training task with the BERT objective in NLP, but the nature of the targets distinguishes it from other audio approaches.

### **4.1 Input Representation: Patch-Based Processing**

Unlike Wav2Vec 2.0, which processes raw waveforms, or AST, which processes standard spectrograms, BEATs treats audio as 2D images.

* **Input Features:** 16kHz audio is converted to 128-dimensional Mel-filterbank (Fbank) features.  
* **Patching:** The spectrogram is sliced into **$16 \\times 16$ patches**.10  
* **Significance:** This 2D patching captures temporal-spectral correlations simultaneously. A single token represents a block of time (approx. 160ms) and frequency band. This differs from frame-level models (like HuBERT) which slice audio only in time. The 2D patch approach allows the model to learn local spectral shapes (like the harmonic stack of a vowel or the sweep of a chirp) as atomic units.12

### **4.2 Aggressive Masking Strategy**

BEATs employs an exceptionally high masking ratio of **75%**.4 This means 3 out of every 4 patches are hidden from the model.

* **Block Masking:** Rather than masking random individual patches, BEATs masks contiguous blocks of time and frequency.  
* **Why 75%?** In reconstruction-based models (Audio-MAE), a high masking ratio is needed to prevent interpolation. In BEATs, it serves a semantic purpose: it forces the model to infer the content of the masked region based on global context. For example, inferring the presence of a "siren" in a masked block based on the Doppler shift visible in the remaining 25% of the spectrogram requires high-level reasoning, not just local texture matching.

### **4.3 Loss Function Comparison**

The objective function is the cross-entropy loss between the predicted token probabilities and the target tokens provided by the acoustic tokenizer.

$$\\mathcal{L} \= \-\\sum\_{t \\in \\mathcal{M}} \\log P(y\_t | \\tilde{x})$$

Where $\\mathcal{M}$ is the set of masked indices, $y\_t$ is the discrete token ID from the tokenizer, and $\\tilde{x}$ is the masked input.

| Feature | BEATs | HuBERT | Wav2Vec 2.0 |
| :---- | :---- | :---- | :---- |
| **Objective** | Discrete Label Prediction | Discrete Label Prediction | Contrastive Learning |
| **Target Origin** | **Learned Tokenizer (Distilled)** | **K-Means Clustering** | Quantized Latent Representation |
| **Iteration Target** | **Tokenizer Network Updates** | **Cluster Center Updates** | End-to-End Updates |
| **Granularity** | Patch (Time-Frequency) | Frame (Time) | Frame (Time) |
| **Semantic Depth** | High (Semantic Abstraction) | Medium (Acoustic Clustering) | Low (Discriminative) |

**Comparative Insight:** The crucial difference between BEATs and HuBERT lies in *what* evolves. In HuBERT, the targets evolve because the clustering is re-run on the hidden states of the improved transformer. In BEATs, a separate *tokenizer network* is trained to distill the teacher. This decoupling allows the BEATs tokenizer to be optimized specifically for discreteness and semantic purity, whereas HuBERT's clusters are a byproduct of the encoder's internal state which may contain redundancies.7

## **5\. Computational Economics and Scalability**

Replicating SOTA audio models requires significant computational resources. The iterative nature of BEATs implies a multiplier on the training cost, as the pre-training process must be run multiple times (typically 3 iterations).

### **5.1 Training Infrastructure and Compute Cost**

Based on data from the OpenBEATs reproduction 10 and standard SSL benchmarks, we can estimate the computational requirements.

* **Model Scale:** The standard BEATs model is approximately **90M parameters** (Base), comparable to ViT-Base. The OpenBEATs initiative scaled this to **300M parameters** (Large).  
* **Data Scale:** The original BEATs was trained on AudioSet-2M (approx. 5.8k hours). OpenBEATs extended this to **20k hours** by including environmental and bioacoustic datasets.  
* **Batch Size:** Large batch sizes are crucial for transformer stability. OpenBEATs reports a batch size equivalent to **10,700 seconds of audio per update**.10 This effectively means the model sees nearly 3 hours of audio in a single forward/backward pass.  
* **Training Duration:** The standard training schedule involves **400,000 updates**.  
* **GPU Hour Estimation:**  
  * Processing a batch of 10.7k seconds of audio requires massive parallelism. Training the 90M parameter model on 20k hours of data typically requires a cluster of 32 to 64 NVIDIA A100 GPUs running for several days.  
  * Estimates suggest approx. **20,000 GPU-hours** for a complete 3-iteration cycle of the Base model.13 This is computationally expensive compared to a single-pass model but efficient compared to LLMs (which require millions of GPU hours).

### **5.2 Scalability Findings (OpenBEATs)**

The OpenBEATs project provided critical insights into the scaling laws of the BEATs objective.10

* **Parameter Scaling:** Scaling from 90M to 300M parameters yielded consistent improvements in downstream tasks, suggesting the masked prediction objective does not saturate quickly.  
* **Data Scaling:** Increasing pre-training data from 5.8k hours to 20k hours improved performance, particularly on out-of-distribution tasks (bioacoustics), validating the "foundation model" hypothesis for audio.

## **6\. Comparison to Alternatives: BEST-RQ and Frozen Codecs**

To fully appreciate the BEATs architecture, one must contrast it with two alternative "tokenization" strategies: the random projection of BEST-RQ and the learned reconstruction of Neural Codecs (EnCodec).

### **6.1 BEATs vs. BEST-RQ**

**BEST-RQ** (BERT-based Speech pre-Training with Random-projection Quantizer) is essentially the "Iter 1" of BEATs frozen forever.9

* **BEST-RQ Method:** Use a fixed random projection and random codebook. Train the encoder to predict these random codes.  
* **Comparison:** BEST-RQ is incredibly efficient because it skips the tokenizer training step. However, it lacks semantic depth. The tokens are merely locality hashes. BEATs demonstrates that while BEST-RQ is a good *initialization*, the **learning** of the tokenizer in Iterations 2 and 3 is what drives the model to SOTA performance (50.6% mAP on AudioSet vs. lower scores for random projection methods). The iterative learning bridges the gap between "acoustic consistency" and "semantic meaning."

### **6.2 BEATs vs. EnCodec (Frozen Codecs)**

Recent generative models (like VALL-E or MusicGen) use tokens from **EnCodec**, a pre-trained neural audio codec.11

* **EnCodec Method:** Optimize a tokenizer for *reconstruction* of the waveform. Use Residual Vector Quantization (RVQ) with multiple codebooks (e.g., 32 codebooks).  
* **The Mismatch:** EnCodec tokens are designed to preserve everything, including noise, phase, and speaker identity. A single second of audio might be represented by 2,250 tokens (75Hz $\\times$ 32 codebooks). This is too granular for semantic classification.  
* **BEATs Advantage:** By using a **single codebook** of size 1024, BEATs acts as a semantic compressor. It intentionally loses the information required to reconstruct the waveform perfectly, retaining only the class-discriminative information. This explains why BEATs excels at AudioSet classification while EnCodec excels at Text-to-Speech generation.

## **7\. Downstream Transfer: The Bioacoustics Case Study**

A critical test of any foundation model is its ability to transfer to domains unseen during pre-training. Bioacoustics (animal vocalizations) presents a rigorous challenge due to the high variability of calls, distinct frequency patterns (often ultra-high frequency), and sparse data regimes.

### **7.1 Performance in Bioacoustics**

The OpenBEATs study explicitly investigated this domain, benchmarking the model on 6 bioacoustic datasets.3

* **SOTA Results:** BEATs representations transfer exceptionally well to bioacoustic tasks (e.g., detecting bird calls, marine mammal sounds, or insect chirps). OpenBEATs achieves state-of-the-art performance, performing better than models exceeding a billion parameters at one-fourth their parameter size.14  
* **Few-Shot Efficiency:** In few-shot scenarios (e.g., training a classifier on only 5 or 10 examples of a rare bird call), BEATs significantly outperforms supervised baselines. A supervised model trained on limited data tends to overfit to the background noise. BEATs, having learned stable acoustic tokens during pre-training, can isolate the "call" structure from the "rainforest" background noise effectively.

### **7.2 Mechanism of Transfer**

Why does a model trained on YouTube (AudioSet) work for whales?

* **Shift Invariance of Patches:** The patch-based ViT architecture allows the model to recognize acoustic shapes (e.g., a "chirp" or "trill") regardless of their absolute frequency. A bird call at 8kHz looks geometrically similar to a whistle at 2kHz in a spectrogram.  
* **Token Abstraction:** The iterative tokenizer learns "proto-features"—basic acoustic units like pulses, harmonics, and sweeps—that are universal across sound generation physics, whether biological or mechanical. When transferred, the model re-combines these universal tokens to describe new biological classes.

## **8\. Conclusion**

BEATs represents the maturation of audio self-supervised learning, moving the field beyond the naive assumption that "reconstruction equals understanding." By implementing an **Iterative Self-Distillation** loop, BEATs bootstraps its own semantic vocabulary from continuous audio, solving the lack of discrete units that has historically plagued the domain.  
The technical success of BEATs rests on three pillars:

1. **The Iterative Protocol:** Transforming random projections (Iter 1\) into semantic clusters (Iter 3\) via distillation.  
2. **The Information Bottleneck:** Using a small, single codebook ($K=1024$) to force the model to discard noise and retain semantics, distinguishing it from reconstruction-focused codecs like EnCodec.  
3. **The Patch-Based Objective:** Utilizing 2D spectrogram patches to capture time-frequency structures, enabling robust transfer to domains like bioacoustics.

With the release of **OpenBEATs**, the framework has been proven scalable to larger parameters (300M) and diverse datasets (20k hours), solidifying the discrete masked prediction objective as a leading candidate for general-purpose audio foundation models. The framework demonstrates that in the absence of language, a model can learn to speak its own language of sound, provided it is given the iterative freedom to evolve its own vocabulary.

#### **Works cited**

1. BEATs: Audio Pre-Training with Acoustic Tokenizers | Request PDF \- ResearchGate, accessed December 13, 2025, [https://www.researchgate.net/publication/366423257\_BEATs\_Audio\_Pre-Training\_with\_Acoustic\_Tokenizers](https://www.researchgate.net/publication/366423257_BEATs_Audio_Pre-Training_with_Acoustic_Tokenizers)  
2. BEATs: Audio Pre-Training with Acoustic Tokenizers, accessed December 13, 2025, [https://proceedings.mlr.press/v202/chen23ag.html](https://proceedings.mlr.press/v202/chen23ag.html)  
3. OpenBEATs: A Fully Open-Source General-Purpose Audio Encoder \- ResearchGate, accessed December 13, 2025, [https://www.researchgate.net/publication/393852145\_OpenBEATs\_A\_Fully\_Open-Source\_General-Purpose\_Audio\_Encoder](https://www.researchgate.net/publication/393852145_OpenBEATs_A_Fully_Open-Source_General-Purpose_Audio_Encoder)  
4. BEATS : Audio Pre-Training with Acoustic Tokenizers \- Proceedings of Machine Learning Research, accessed December 13, 2025, [https://proceedings.mlr.press/v202/chen23ag/chen23ag.pdf](https://proceedings.mlr.press/v202/chen23ag/chen23ag.pdf)  
5. \[PDF\] BEATs: Audio Pre-Training with Acoustic Tokenizers \- Semantic Scholar, accessed December 13, 2025, [https://www.semanticscholar.org/paper/BEATs%3A-Audio-Pre-Training-with-Acoustic-Tokenizers-Chen-Wu/89a1dbbfd4c96d90b769f5d3427bd970b082898e](https://www.semanticscholar.org/paper/BEATs%3A-Audio-Pre-Training-with-Acoustic-Tokenizers-Chen-Wu/89a1dbbfd4c96d90b769f5d3427bd970b082898e)  
6. The Effect of Batch Size on Contrastive Self-Supervised Speech Representation Learning, accessed December 13, 2025, [https://arxiv.org/html/2402.13723v1](https://arxiv.org/html/2402.13723v1)  
7. BEATs : Audio Pre-Training with Acoustic Tokenizers, accessed December 13, 2025, [https://icml.cc/media/icml-2023/Slides/25555.pdf](https://icml.cc/media/icml-2023/Slides/25555.pdf)  
8. \[2212.09058\] BEATs: Audio Pre-Training with Acoustic Tokenizers \- ar5iv \- arXiv, accessed December 13, 2025, [https://ar5iv.labs.arxiv.org/html/2212.09058](https://ar5iv.labs.arxiv.org/html/2212.09058)  
9. OMAR-RQ: Open Music Audio Representation Model Trained with Multi-Feature Masked Token Prediction \- arXiv, accessed December 13, 2025, [https://arxiv.org/html/2507.03482v1](https://arxiv.org/html/2507.03482v1)  
10. OpenBEATs: A Fully Open-Source General-Purpose Audio Encoder \- arXiv, accessed December 13, 2025, [https://arxiv.org/html/2507.14129v1](https://arxiv.org/html/2507.14129v1)  
11. Discrete Audio Representations for Automated Audio Captioning \- arXiv, accessed December 13, 2025, [https://arxiv.org/html/2505.14989v1](https://arxiv.org/html/2505.14989v1)  
12. Effective Techniques for Scaling Audio Encoder Pretraining \- Amazon Science, accessed December 13, 2025, [https://assets.amazon.science/44/1b/bb7b0f2b4b02af36c3b81c08aa23/effective-techniques-for-scaling-audio-encoder-pretraining.pdf](https://assets.amazon.science/44/1b/bb7b0f2b4b02af36c3b81c08aa23/effective-techniques-for-scaling-audio-encoder-pretraining.pdf)  
13. $100K or 100 Days: Trade-offs when Pre-Training with Academic Resources \- arXiv, accessed December 13, 2025, [https://arxiv.org/html/2410.23261v2](https://arxiv.org/html/2410.23261v2)  
14. OpenBEATs: A Fully Open-Source General-Purpose Audio Encoder \- Shikhar, accessed December 13, 2025, [https://shikhar-s.github.io/OpenBEATs/](https://shikhar-s.github.io/OpenBEATs/)

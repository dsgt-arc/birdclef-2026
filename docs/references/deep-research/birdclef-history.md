# **A Decade of Computational Ornithology: An Exhaustive Analysis of Algorithmic Evolution in BirdCLEF Competitions (2014-2025)**

## **1\. Introduction: The Bioacoustic Imperative and the BirdCLEF Paradigm**

The accelerating decline of global biodiversity has precipitated an urgent need for scalable, reliable, and cost-effective monitoring systems. Birds, as highly mobile and vocally active organisms, serve as premier bio-indicators for ecosystem health. Their specific assemblages, population densities, and behavioral patterns offer critical insights into habitat quality, climate change impacts, and the efficacy of restoration interventions.1 Traditionally, avian biodiversity surveys relied on human observers conducting point counts—a methodology constrained by logistical costs, observer bias, and the inability to monitor vast, remote areas continuously.
In response to these limitations, Passive Acoustic Monitoring (PAM) has emerged as a transformative methodology. By deploying Autonomous Recording Units (ARUs) such as AudioMoths, researchers can collect continuous audio data over weeks or months, capturing the soundscapes of ecosystems with high temporal resolution.3 However, this technological shift has generated a new bottleneck: the "data deluge." A single PAM project can generate terabytes of audio, far exceeding the capacity of human analysts to review. Consequently, the automated extraction of ecological signals from complex acoustic environments has become a paramount challenge in computational ecology.3
Since its inception in 2014, the BirdCLEF (Bird Identification Task at the Conference and Labs of the Evaluation Forum) competition has served as the primary testbed for advancing these automated technologies. Hosted annually, largely on the Kaggle platform in recent years, BirdCLEF challenges the global machine learning community to develop algorithms capable of identifying bird species in noisy, overlapping, and real-world soundscapes.6 This report provides an exhaustive, diachronic analysis of the techniques employed across all editions of BirdCLEF. By synthesizing overview papers, winning solution write-ups, and open-source code repositories from 2014 through 2025, we trace the technological evolution from classical template matching to the current era of semi-supervised foundation models. The analysis reveals a field that has matured from simple signal processing to complex, data-centric AI pipelines, offering a roadmap for the future of bioacoustic research.

## **2\. The Pre-Deep Learning Era (2014-2017): Template Matching and Feature Engineering**

The early phase of BirdCLEF is characterized by the dominance of classical signal processing and machine learning techniques. Understanding these foundational methods is essential, as they defined the core challenges—feature extraction, segmentation, and noise resilience—that modern deep learning models still grapple with today.

### **2.1 The Dominance of Template Matching**

Before the widespread adoption of Convolutional Neural Networks (CNNs), the state-of-the-art in bird sound recognition was defined by template matching and cross-correlation. This approach was exemplified by the winning solutions of Mario Lasseck, whose systems consistently outperformed others in the 2016 and 2017 editions.8
The fundamental premise of this methodology was that bird vocalizations consist of repetitive, distinct spectro-temporal patterns (syllables). The recognition pipeline typically followed these steps:

1. **Spectrogram Segmentation:** The continuous audio stream was converted into a spectrogram. Algorithms then attempted to segment this image into "regions of interest" (ROIs) based on energy thresholds, separating potential bird calls from background noise.9
2. **Template Library Creation:** For each target species, a library of representative "templates" was curated from the training data. These templates captured the idealized frequency contours of the bird's calls.
3. **Cross-Correlation:** During inference, these templates were "slid" across the test spectrograms. A normalized cross-correlation coefficient was calculated at each time step. High correlation peaks indicated a potential match.10

While effective for distinct calls in low-noise environments, this approach suffered from significant limitations. It was computationally expensive, scaling linearly with the number of species and templates. More critically, it lacked robustness to **intra-species variation**. Birds of the same species often have regional dialects or individual variations in call structure that a rigid template could not accommodate. Furthermore, the "sliding window" approach struggled with the complex, overlapping soundscapes introduced in later competitions, where biotic and abiotic noise distorted the correlation signals.11

### **2.2 Feature Engineering and Random Forests**

To augment template matching, early competitors employed extensive feature engineering. Instead of raw spectrogram pixels, systems extracted parametric acoustic features. The **openSMILE** toolkit was a standard utility, used to extract thousands of features including:

* **Mel-Frequency Cepstral Coefficients (MFCCs):** Capturing the timbral envelope of the sound.
* **Spectral Centroids and Flux:** Describing the "brightness" and rate of change of the spectrum.
* **Zero-Crossing Rates:** Useful for distinguishing noisy, unvoiced calls from harmonic ones.

These features were fed into ensemble classifiers, most notably **Random Forests** and **Decision Trees**.9 In the 2016 and 2017 editions, Lasseck’s approach involved selecting the most discriminative features for each species via a decision tree ensemble, reducing the dimensionality of the problem before classification. This feature selection step was critical, as the "curse of dimensionality" was a major hurdle given the limited training data available for rare species.9

### **2.3 The Introduction of Soundscapes**

The 2017 competition marked a pivotal shift in the dataset structure. Previous years focused on "focal" recordings—high-quality clips where the target bird was centered and dominant. In 2017, organizers introduced "soundscape" recordings: long-duration, omnidirectional recordings from the field containing multiple overlapping birds and significant environmental noise.11
This introduction exposed the brittleness of classical methods. Template matching systems, which assumed a clean signal, generated high false-positive rates when confronted with rain, wind, or insect noise that mimicked the frequency bands of bird calls. The performance drop observed in 2017 motivated the community to explore more robust, non-linear feature extractors, setting the stage for the deep learning revolution.11

## **3\. The Convolutional Revolution (2018-2020): Audio as an Image**

The years 2018 through 2020 witnessed a paradigm shift that mirrored the broader computer vision community. Participants began treating audio classification not as a signal processing task, but as an image classification task. The raw audio was converted into visual representations (spectrograms), and Convolutional Neural Networks (CNNs) became the standard architecture.13

### **3.1 The Spectrogram Paradigm**

The conversion of 1D audio waveforms into 2D time-frequency representations became the universal preprocessing step. While various formats were tested (e.g., constant-Q transforms), the **Log-Mel Spectrogram** emerged as the industry standard.

* **Mel Scale:** The frequency axis is warped to the Mel scale, which mimics human (and avian) auditory perception by allocating more resolution to lower frequencies and compressing higher frequencies.
* **Logarithmic Amplitude:** The amplitude is scaled logarithmically (decibels) to compress the high dynamic range of audio signals, allowing the model to "see" faint distant calls alongside loud, nearby ones.14

Specific parameters became standardized through trial and error across competitions. A sample rate of 32 kHz (capturing frequencies up to 16 kHz) became common, with window sizes around 2048 samples and hop lengths of 512 samples. These settings balanced temporal resolution (critical for rapid trills) with frequency resolution.15

### **3.2 Architecture Adaptation: From Inception to ResNet**

Initial deep learning attempts utilized architectures pre-trained on ImageNet (a massive dataset of object photos). The hypothesis was that the edge and texture detectors learned from visual images would transfer to the "edges" (frequency contours) and "textures" (harmonics) of spectrograms.

* **Inception-v3 and v4:** Mario Lasseck’s 2018 and 2019 winning solutions transitioned from template matching to Inception-based CNNs. The multi-scale nature of Inception modules (processing the input with filters of different sizes simultaneously) proved effective at capturing bird calls that varied in duration and bandwidth.8
* **ResNets:** Residual Networks (ResNet-50, ResNet-101) gained popularity for their ability to train deeper networks without vanishing gradient problems. Stefan Kahl’s work, which culminated in the **BirdNET** system, demonstrated that ResNet architectures could achieve state-of-the-art performance on large-scale datasets like Xeno-canto.13

### **3.3 The Focal vs. Soundscape Domain Gap**

A defining characteristic of the 2019 and 2020 competitions was the explicit focus on the "domain gap."

* **Source Domain (Training):** Xeno-canto recordings. These are "focal," meaning a human recordist pointed a directional microphone at a specific bird. The signal-to-noise ratio (SNR) is high, and the bird is usually centered.
* **Target Domain (Test):** Soundscapes. These are passive recordings. The birds are often distant, off-axis, and buried in noise.

Models trained purely on Xeno-canto data often failed on soundscapes due to "overfitting to clean audio." They would classify high-frequency insect noise as bird calls or miss faint bird calls entirely. This necessitated the development of **Data Augmentation** as a primary regularization strategy. Techniques such as adding Gaussian noise, mixing background tracks, and random cropping were introduced to simulate the noisy test conditions during training.19

## **4\. The Sound Event Detection (SED) Era (2021-2023)**

As the competitions progressed, the complexity of the task increased. The challenge moved from "what bird is in this file?" to "exactly *when* does this bird call?" This necessitated the adoption of Weakly Supervised Sound Event Detection (SED) frameworks.

### **4.1 The Weak Supervision Challenge**

The training data from Xeno-canto is "weakly labeled." A 3-minute recording might be labeled "Black-capped Chickadee," but the bird might only vocalize for 5 seconds at the 1:12 mark. Standard classifiers treating the whole file as a positive sample would learn the background noise (wind, silence) as features of the class.
To address this, competitors adopted **Multiple Instance Learning (MIL)** and SED architectures. The model is designed to output a prediction for every "frame" or small time segment (e.g., every 32ms).

* **Pooling Layers:** To compute a loss against the file-level label, the frame-level predictions are aggregated. Global Max Pooling (GMP) was a common choice: if *any* frame predicts the bird with high confidence, the file is classified as positive.
* **Attention Pooling:** More sophisticated models used attention mechanisms to learn weighted averages of the frames, allowing the model to focus on the relevant segments while ignoring the silence.21

### **4.2 Architectural Dominance: EfficientNet and NFNet**

By 2021, the **EfficientNet** family (B0 through B7) had displaced ResNets as the backbone of choice. Their compound scaling method provided a better trade-off between accuracy and floating-point operations (FLOPs), which was critical given Kaggle's inference runtime limits (typically 2 hours CPU or 9-hour GPU constraints).15
In 2022 and 2023, **NFNet (Normalizer-Free Networks)** also saw significant adoption, particularly by top teams like the 1st place winner in 2022\. NFNet architectures remove Batch Normalization layers, which can be problematic when batch statistics vary significantly between training (focal) and testing (soundscape) domains. By removing this dependency, NFNets often generalized better to the shifted distribution of the test set.24

### **4.3 Advanced Data Augmentation Pipelines**

To bridge the domain gap, data augmentation became highly sophisticated. The following table summarizes the key augmentation techniques employed during this era:

| Augmentation Technique | Implementation Mechanism | Purpose & Impact | Source |
| :---- | :---- | :---- | :---- |
| **MixUp** | Linearly interpolates two audio clips and their labels ($x' \= \\lambda x\_1 \+ (1-\\lambda)x\_2$). | Simulates overlapping calls, a pervasive feature of soundscapes. Prevents the model from becoming over-confident. | 15 |
| **Pink/Gaussian Noise** | Injects colored noise into the raw waveform. | Mimics thermal noise of microphones and environmental wind/rain noise. | 24 |
| **SpecAugment** | Masks blocks of time or frequency bands in the spectrogram. | Forces the model to learn robust features rather than relying on specific frequency transients. Prevents overfitting to specific syllables. | 21 |
| **Background Mixing** | Mixes "nocall" segments (rain, forest silence) with bird calls. | Directly addresses the focal-to-soundscape shift by embedding clean calls into noisy backgrounds. | 21 |
| **Pitch Shift / Time Stretch** | Alters the pitch or speed of the audio without changing the other. | Simulates intra-species variation (e.g., dialects, individual differences) and recording artifacts. | 25 |

The **"Background Mixing"** strategy deserves special mention. Competitors often curated specific datasets of "nocall" audio—recordings from the competition site that contained no birds. Mixing these with Xeno-canto training data was the single most effective technique for reducing false positives in the 2021 and 2022 competitions.21

## **5\. The Semi-Supervised Frontier (2024-2025): Pseudo-Labeling and Self-Distillation**

The most recent editions of BirdCLEF (2024 and 2025\) introduced a structural change that fundamentally altered the competition landscape: the provision of large, **unlabeled soundscape datasets** from the target recording locations (e.g., the Western Ghats in India for 2024, Colombia for 2025). This shifted the problem from pure supervised learning to **Semi-Supervised Learning (SSL)**.

### **5.1 The "Noisy Student" Paradigm**

The winning solutions in 2024 and 2025 converged on iterative pseudo-labeling strategies, often referred to as "Noisy Student" training. This approach leverages the massive unlabeled test data to adapt the model to the target distribution.28
**The Iterative Workflow:**

1. **Teacher Training:** An initial ensemble (the "Teacher") is trained on the labeled Xeno-canto data using standard supervised learning.
2. **Pseudo-Label Generation:** The Teacher model is run on the massive *unlabeled* soundscape dataset. It predicts probabilities for every segment.
3. **Filtering and Soft-Labeling:** High-confidence predictions are selected. Crucially, the 2025 winner (Nikita Babych) introduced a **power-scaling** technique ($p' \= p^x$, with $x \> 1$) to sharpen the probability distribution, suppressing noise while retaining information from lower-confidence predictions.28
4. **Student Training:** A new model (the "Student") is trained on a combination of the original labeled data and the newly pseudo-labeled soundscape data. The Student is typically trained with stronger augmentation (MixUp, Dropout) to force it to learn robust features.
5. **Iteration:** The Student becomes the new Teacher, and the process is repeated 3-4 times.

**Impact:** In 2024, the 2nd place team (adsr) showed that this method improved their public leaderboard score from \~0.65 to over 0.71—a massive leap that architectural tuning alone could not achieve.15

### **5.2 Self-Distillation for Secondary Labels**

While "Noisy Student" targets the unlabeled *test* data, the 2025 5th place solution by MYSO introduced **"Self-Distillation"** to refine the *training* data.
**The Problem:** Xeno-canto recordings are labeled with a "primary" species. However, these field recordings often capture other "secondary" birds in the background that are not labeled by the uploader. When a model predicts these background birds correctly, it is penalized by the loss function because the ground truth label is missing.16
**The Solution:**

1. Train a model on the provided labels.
2. Run this model on the *training set* itself.
3. Where the model confidently predicts a bird that is not labeled, update the training label to include this "discovered" species.
4. Retrain the model on the enriched labels.

This technique effectively cleans the noisy labels of the training set, allowing the model to learn from the background signals rather than treating them as noise.16

### **5.3 Case Study: BirdCLEF 2025 1st Place Solution (Nikita Babych)**

The 2025 winning solution represents the current pinnacle of this semi-supervised approach. Key technical details include:

* **Backbone:** EfficientNet variants (B0 through V2-S).
* **Inference Duration:** 20-second audio chunks were found to be optimal, significantly longer than the standard 5-second window. This allowed the model to capture longer-term temporal context.
* **Loss Function:** A shift back to **Cross-Entropy** from Focal Loss. The author argued that Focal Loss could lead to overfitting on noisy labels, while Cross-Entropy provided more stable gradients when combined with the Noisy Student process.
* **Power-Law Pseudo-Labels:** The defining innovation was applying a power function to the pseudo-labels. This allowed the use of *all* unlabeled data (soft labels) rather than hard thresholding, preserving the subtle distribution of the test set.28

## **6\. Technical Deep Dive: Engineering the Modern Pipeline**

Beyond the high-level algorithms, the success of top solutions relies on precise engineering choices in data handling, inference, and hardware utilization.

### **6.1 Hardware and Software Ecosystem**

An analysis of GitHub repositories from top competitors reveals a standardized software stack optimized for speed and reproducibility.

* **Frameworks:** **PyTorch** is the overwhelming favorite, utilizing libraries like timm (PyTorch Image Models) for accessing pre-trained backbones.30 **Torchaudio** and **Librosa** are used for spectrogram generation.
* **Dependency Management:** Modern solutions increasingly use **Poetry** (as seen in the 2025 2nd place solution) or Docker containers to ensure reproducible environments.31
* **Accelerators:** For inference, converting models to **ONNX** or **OpenVINO** is a standard practice. This is critical for meeting the 2-hour submission limit on Kaggle's CPU-only inference environments. OpenVINO, in particular, allows for FP16 quantization, doubling inference speed with negligible accuracy loss.16

### **6.2 Inference Strategies and Test-Time Augmentation (TTA)**

Inference is rarely a single forward pass. Top teams employ sophisticated aggregation strategies:

* **Sliding Window Smoothing:** Predictions are made on overlapping windows (e.g., sliding a 5-second window every 2.5 seconds). A weighted average (e.g., \[0.1, 0.2, 0.4, 0.2, 0.1\]) is applied to smooth the probabilities over time, enforcing temporal consistency (a bird call is unlikely to disappear and reappear in 30ms).16
* **Delta-Shift TTA:** Running the model on the original clip and a time-shifted version (e.g., shifted by 1 second) and averaging the results helps handle edge cases where a call might be split by the window boundary.28
* **NoCall Suppression:** Explicitly predicting a "nocall" class allows the model to calibrate its output. If the nocall probability is high, all species probabilities are dampened, reducing false positives in silent sections.33

### **6.3 Data Augmentation Libraries**

Specific libraries have become staples in the competition codebases:

* **Audiomentations:** A Python library for audio data augmentation. Competitors use it for AddGaussianNoise, TimeStretch, and PitchShift.25
* **Torchlibrosa:** A PyTorch implementation of Librosa's spectrogram functions. This allows the spectrogram generation to be part of the GPU computation graph, significantly speeding up training compared to CPU-based preprocessing.34

## **7\. Future Directions: Foundation Models and Source-Free Adaptation**

As the field moves beyond 2025, the literature and recent trends point toward three distinct frontiers for future research: Foundation Models, Source-Free Domain Adaptation, and Edge Intelligence.

### **7.1 The Rise of Foundation Models (Bioacoustic LLMs)**

The era of training CNNs from scratch on competition data is ending. The future lies in **Foundation Models**—large-scale models pre-trained on millions of hours of bioacoustic data.

* **BirdNET and Perch:** Google's Perch and Cornell's BirdNET are already being used as feature extractors. Competitors freeze the backbone and train only a lightweight "head" on the competition data. This "Transfer Learning" approach is particularly effective for rare species (Few-Shot Learning).18
* **AudioMAE (Masked Autoencoders):** Recent research highlights **AudioMAE**, a Transformer-based model trained via self-supervision (masking parts of the spectrogram and forcing the model to reconstruct them). AudioMAE has shown superior transferability compared to supervised CNNs because it learns general acoustic features (edges, harmonics, formants) without being biased by human labels.36
* **BioLingual and Cross-Modal Models:** Emerging models like **BioLingual** align audio embeddings with text descriptions or taxonomic trees. This could allow future models to identify a bird species it has never heard before, simply by reading a textual description of its call (Zero-Shot Learning).37

### **7.2 Source-Free Domain Adaptation (SFDA)**

A critical limitation of current "Noisy Student" methods is the requirement to access both the source data (Xeno-canto) and target data (soundscapes) simultaneously. In real-world deployment, privacy or storage constraints often prevent access to the original training data.
Future research will focus on **Source-Free Domain Adaptation (SFDA)**. This involves adapting a pre-trained model to a new environment using *only* the unlabeled data from that environment. Techniques like **Contrastive Adversarial Domain (CAD)** bottlenecks are emerging to learn features that are invariant to the recording device or background noise, allowing models to "snap" to a new location without retraining on the source data.3

### **7.3 Edge Intelligence and Open-Set Recognition**

The ultimate goal of BirdCLEF is to enable conservationists to deploy models on low-power devices in the wild.

* **Edge Deployment:** The focus on **EfficientVit** and **MobileNet** in recent competitions signals a move toward models that can run on microcontrollers (e.g., Raspberry Pi, AudioMoth).14
* **Open-Set Recognition:** Models in the wild encounter sounds they were never trained on (e.g., a chainsaw, a new invasive species). Current models confidently classify these as known birds (overconfidence). Future research must develop **Open-Set Recognition** capabilities, allowing models to output an "Unknown" token rather than a false positive, flagging the sound for human review.36

## **8\. Conclusion**

The evolution of BirdCLEF methodologies from 2014 to 2025 represents a microcosm of the broader AI revolution. The field has transitioned from hand-crafted features and template matching to deep CNNs, and now to sophisticated semi-supervised pipelines that leverage unlabeled data.
The "solved" problems in this domain are the input representation (Log-Mel Spectrograms) and the backbone architecture (EfficientNet). The active frontier—and the key to winning future competitions—lies in **Data-Centric AI**. The ability to clean noisy labels (Self-Distillation), adapt to new acoustic environments (Noisy Student/SFDA), and leverage general-purpose acoustic knowledge (Foundation Models) differentiates the state-of-the-art from the baseline.
For the conservation community, these advancements promise a new era of biodiversity monitoring. We are moving toward systems that can be deployed anywhere, adapt to the local soundscape without human intervention, and provide real-time, reliable data on the health of our planet's avian populations. The algorithms are no longer just recognizing patterns; through self-supervision and adaptation, they are beginning to "listen" with a level of nuance that rivals expert ornithologists.

## **Appendix: Summary of Key Techniques by Era**

| Era | Dominant Algorithms | Input Representation | Key Innovations | Representative Solutions |
| :---- | :---- | :---- | :---- | :---- |
| **2014-2017** | Random Forest, SVM, Template Matching | MFCCs, Spectrogram Segments, OpenSMILE features | Cross-correlation templates, Decision tree feature selection | Lasseck (2016, 2017\) 8 |
| **2018-2020** | Inception-v3, ResNet-50/101 | Log-Mel Spectrograms (PCEN) | Transfer Learning from ImageNet, Mixup Augmentation, Soundscape generalization | Kahl (BirdNET) 13, Lasseck (2019) 8 |
| **2021-2023** | EfficientNet B0-B4, NFNet, SED Architectures | Mel Spectrograms with SpecAugment | Weakly Supervised Learning (SED), Background Mixing, Pink Noise Injection | Kami (2021) 33, Team KDL (2022) 24 |
| **2024-2025** | EfficientNetV2, EfficientVit, AudioMAE | Mel Spectrograms \+ Unlabeled Data | **Pseudo-labeling**, Multi-Iterative Noisy Student, Self-Distillation, Power-law probabilities | Nikita Babych (2025) 28, MYSO (2025) 16 |

#### **Works cited**

1. BirdCLEF 2024 | ImageCLEF / LifeCLEF \- Multimedia Retrieval in CLEF, accessed December 13, 2025, [https://www.imageclef.org/node/316](https://www.imageclef.org/node/316)
2. BirdCLEF 2023 | Kaggle, accessed December 13, 2025, [https://www.kaggle.com/competitions/birdclef-2023](https://www.kaggle.com/competitions/birdclef-2023)
3. Overview of BirdCLEF 2024: Acoustic Identification ... \- Agritrop \- Cirad, accessed December 13, 2025, [https://agritrop.cirad.fr/613024/1/613024.pdf](https://agritrop.cirad.fr/613024/1/613024.pdf)
4. Overview of BirdCLEF 2022: Endangered bird species recognition in soundscape recordings \- CEUR-WS, accessed December 13, 2025, [https://ceur-ws.org/Vol-3180/paper-154.pdf](https://ceur-ws.org/Vol-3180/paper-154.pdf)
5. Notebook for the BirdCLEF Lab at CLEF 2024 \- CEUR-WS.org, accessed December 13, 2025, [https://ceur-ws.org/Vol-3740/paper-198.pdf](https://ceur-ws.org/Vol-3740/paper-198.pdf)
6. BirdCLEF 2023 | ImageCLEF / LifeCLEF \- Multimedia Retrieval in CLEF, accessed December 13, 2025, [https://www.imageclef.org/BirdCLEF2023](https://www.imageclef.org/BirdCLEF2023)
7. Overview of BirdCLEF 2020: Bird Sound Recognition in Complex Acoustic Environments, accessed December 13, 2025, [https://www.semanticscholar.org/paper/Overview-of-BirdCLEF-2020%3A-Bird-Sound-Recognition-Kahl-Clapp/f0b5cbfaa51373c9db816f834cb2dd10f2877151](https://www.semanticscholar.org/paper/Overview-of-BirdCLEF-2020%3A-Bird-Sound-Recognition-Kahl-Clapp/f0b5cbfaa51373c9db816f834cb2dd10f2877151)
8. Bird Species Identification in Soundscapes \- CEUR-WS.org, accessed December 13, 2025, [https://ceur-ws.org/Vol-2380/paper\_86.pdf](https://ceur-ws.org/Vol-2380/paper_86.pdf)
9. Improving Bird Identification using Multiresolution Template Matching and Feature Selection during Training \- CEUR-WS, accessed December 13, 2025, [https://ceur-ws.org/Vol-1609/16090490.pdf](https://ceur-ws.org/Vol-1609/16090490.pdf)
10. visualizes the official scores of the LifeCLEF 2016 Bird Identification... \- ResearchGate, accessed December 13, 2025, [https://www.researchgate.net/figure/sualizes-the-official-scores-of-the-LifeCLEF-2016-Bird-Identification-Task-The-here\_fig3\_338423035](https://www.researchgate.net/figure/sualizes-the-official-scores-of-the-LifeCLEF-2016-Bird-Identification-Task-The-here_fig3_338423035)
11. BirdCLEF 2017 results overview-Mean Average Precision \- ResearchGate, accessed December 13, 2025, [https://www.researchgate.net/figure/BirdCLEF-2017-results-overview-Mean-Average-Precision\_fig2\_318899617](https://www.researchgate.net/figure/BirdCLEF-2017-results-overview-Mean-Average-Precision_fig2_318899617)
12. Improved Automatic Bird Identification through Decision Tree based Feature Selection and Bagging \- Semantic Scholar, accessed December 13, 2025, [https://www.semanticscholar.org/paper/Improved-Automatic-Bird-Identification-through-Tree-Lasseck/26291fd568abd2e5f5d59177490a97538f025be7](https://www.semanticscholar.org/paper/Improved-Automatic-Bird-Identification-through-Tree-Lasseck/26291fd568abd2e5f5d59177490a97538f025be7)
13. Overview of BirdCLEF 2018: Monospecies vs. Sundscape Bird Identification, accessed December 13, 2025, [https://www.semanticscholar.org/paper/Overview-of-BirdCLEF-2018%3A-Monospecies-vs.-Bird-Go%C3%ABau-Kahl/17dc4e8cbe45bb26daaffddd2c0b15993c48aa3b](https://www.semanticscholar.org/paper/Overview-of-BirdCLEF-2018%3A-Monospecies-vs.-Bird-Go%C3%ABau-Kahl/17dc4e8cbe45bb26daaffddd2c0b15993c48aa3b)
14. 1st place solution | Kaggle, accessed December 13, 2025, [https://www.kaggle.com/competitions/birdclef-2024/writeups/team-kefir-1st-place-solution](https://www.kaggle.com/competitions/birdclef-2024/writeups/team-kefir-1st-place-solution)
15. 2nd place solution | Kaggle, accessed December 13, 2025, [https://www.kaggle.com/competitions/birdclef-2024/writeups/adsr-2nd-place-solution](https://www.kaggle.com/competitions/birdclef-2024/writeups/adsr-2nd-place-solution)
16. 5th place solution: Self-Distillation is All You Need \- Kaggle, accessed December 13, 2025, [https://www.kaggle.com/competitions/birdclef-2025/writeups/noir-5th-place-solution-self-distillation-is-all-y](https://www.kaggle.com/competitions/birdclef-2025/writeups/noir-5th-place-solution-self-distillation-is-all-y)
17. \[PDF\] Audio-based Bird Species Identification with Deep Convolutional Neural Networks | Semantic Scholar, accessed December 13, 2025, [https://www.semanticscholar.org/paper/Audio-based-Bird-Species-Identification-with-Deep-Lasseck/011a4c362759a7f5c5e9a642417561b7073b1306](https://www.semanticscholar.org/paper/Audio-based-Bird-Species-Identification-with-Deep-Lasseck/011a4c362759a7f5c5e9a642417561b7073b1306)
18. Publications \- BirdNET, accessed December 13, 2025, [https://birdnet.cornell.edu/publications/](https://birdnet.cornell.edu/publications/)
19. Overview of BirdCLEF 2019: Large-Scale Bird Recognition in Soundscapes \- CEUR-WS.org, accessed December 13, 2025, [https://ceur-ws.org/Vol-2380/paper\_256.pdf](https://ceur-ws.org/Vol-2380/paper_256.pdf)
20. Overview of BirdCLEF 2019: Large-Scale Bird Recognition in Soundscapes, accessed December 13, 2025, [https://www.semanticscholar.org/paper/Overview-of-BirdCLEF-2019%3A-Large-Scale-Bird-in-Kahl-St%C3%B6ter/c90b8fe46eb289b31e0f8d9d84e5f800ea4bd2d8](https://www.semanticscholar.org/paper/Overview-of-BirdCLEF-2019%3A-Large-Scale-Bird-in-Kahl-St%C3%B6ter/c90b8fe46eb289b31e0f8d9d84e5f800ea4bd2d8)
21. 1st place solution: Correct Data is All You Need \- Kaggle, accessed December 13, 2025, [https://www.kaggle.com/competitions/birdclef-2023/writeups/volodymyr-1st-place-solution-correct-data-is-all-y](https://www.kaggle.com/competitions/birdclef-2023/writeups/volodymyr-1st-place-solution-correct-data-is-all-y)
22. Bird Species Recognition using Convolutional Neural Networks with Attention on Frequency Bands \- ResearchGate, accessed December 13, 2025, [https://www.researchgate.net/publication/389264675\_Bird\_Species\_Recognition\_using\_Convolutional\_Neural\_Networks\_with\_Attention\_on\_Frequency\_Bands](https://www.researchgate.net/publication/389264675_Bird_Species_Recognition_using_Convolutional_Neural_Networks_with_Attention_on_Frequency_Bands)
23. 4th place solution | Kaggle, accessed December 13, 2025, [https://www.kaggle.com/competitions/birdclef-2025/writeups/dylan-liu-4th-place-solution](https://www.kaggle.com/competitions/birdclef-2025/writeups/dylan-liu-4th-place-solution)
24. 1st place solution models (it's not all BirdNet) \- Kaggle, accessed December 13, 2025, [https://www.kaggle.com/competitions/birdclef-2022/writeups/kdl-1st-place-solution-models-it-s-not-all-birdnet](https://www.kaggle.com/competitions/birdclef-2022/writeups/kdl-1st-place-solution-models-it-s-not-all-birdnet)
25. Bird Species Classification Using CNNBased Deep Learning and Advanced Data Augmentation Techniques \- ResearchGate, accessed December 13, 2025, [https://www.researchgate.net/publication/385152191\_Bird\_Species\_Classification\_Using\_CNNBased\_Deep\_Learning\_and\_Advanced\_Data\_Augmentation\_Techniques](https://www.researchgate.net/publication/385152191_Bird_Species_Classification_Using_CNNBased_Deep_Learning_and_Advanced_Data_Augmentation_Techniques)
26. Improving Learning-Based Birdsong Classification by Utilizing Combined Audio Augmentation Strategies \- ResearchGate, accessed December 13, 2025, [https://www.researchgate.net/publication/381758861\_Improving\_Learning-Based\_Birdsong\_Classification\_by\_Utilizing\_Combined\_Audio\_Augmentation\_Strategies](https://www.researchgate.net/publication/381758861_Improving_Learning-Based_Birdsong_Classification_by_Utilizing_Combined_Audio_Augmentation_Strategies)
27. Data augmentation approaches for improving animal audio classification | Request PDF, accessed December 13, 2025, [https://www.researchgate.net/publication/340044591\_Data\_augmentation\_approaches\_for\_improving\_animal\_audio\_classification](https://www.researchgate.net/publication/340044591_Data_augmentation_approaches_for_improving_animal_audio_classification)
28. 1st Place Solution: Multi-Iterative Noisy Student Is All You Need | Kaggle, accessed December 13, 2025, [https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n](https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n)
29. Improving Bird Recognition using Pseudo-Labeled Recordings from the Target Location \- CEUR-WS.org, accessed December 13, 2025, [https://ceur-ws.org/Vol-3740/paper-199.pdf](https://ceur-ws.org/Vol-3740/paper-199.pdf)
30. 3rd Place Solution to the BirdCLEF 2024 Competition \- Theo's part \- GitHub, accessed December 13, 2025, [https://github.com/TheoViel/kaggle\_birdclef2024](https://github.com/TheoViel/kaggle_birdclef2024)
31. myso1987/BirdCLEF-2025-5th-place-solution \- GitHub, accessed December 13, 2025, [https://github.com/myso1987/BirdCLEF-2025-5th-place-solution](https://github.com/myso1987/BirdCLEF-2025-5th-place-solution)
32. VSydorskyy/BirdCLEF\_2025\_2nd\_place \- GitHub, accessed December 13, 2025, [https://github.com/VSydorskyy/BirdCLEF\_2025\_2nd\_place](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place)
33. \[1st Place\] Quick Solution | Kaggle, accessed December 13, 2025, [https://www.kaggle.com/competitions/birdclef-2021/writeups/dr-1st-place-quick-solution](https://www.kaggle.com/competitions/birdclef-2021/writeups/dr-1st-place-quick-solution)
34. Kaggle Winning Solutions: AI Trends & Insights, accessed December 13, 2025, [https://www.kaggle.com/code/tahaalselwii/kaggle-winning-solutions-ai-trends-insights](https://www.kaggle.com/code/tahaalselwii/kaggle-winning-solutions-ai-trends-insights)
35. Transfer Learning with Pseudo Multi-Label Birdcall Classification for DS@GT BirdCLEF 2024 \- arXiv, accessed December 13, 2025, [https://arxiv.org/html/2407.06291v1](https://arxiv.org/html/2407.06291v1)
36. Improving Bird Classification with Unsupervised Sound Separation | Request PDF, accessed December 13, 2025, [https://www.researchgate.net/publication/360793140\_Improving\_Bird\_Classification\_with\_Unsupervised\_Sound\_Separation](https://www.researchgate.net/publication/360793140_Improving_Bird_Classification_with_Unsupervised_Sound_Separation)
37. BEANS: The Benchmark of Animal Sounds \- ResearchGate, accessed December 13, 2025, [https://www.researchgate.net/publication/371287440\_BEANS\_The\_Benchmark\_of\_Animal\_Sounds](https://www.researchgate.net/publication/371287440_BEANS_The_Benchmark_of_Animal_Sounds)
38. (PDF) Clustering and novel class recognition: evaluating bioacoustic deep learning feature extractors \- ResearchGate, accessed December 13, 2025, [https://www.researchgate.net/publication/390638894\_Clustering\_and\_novel\_class\_recognition\_evaluating\_bioacoustic\_deep\_learning\_feature\_extractors](https://www.researchgate.net/publication/390638894_Clustering_and_novel_class_recognition_evaluating_bioacoustic_deep_learning_feature_extractors)

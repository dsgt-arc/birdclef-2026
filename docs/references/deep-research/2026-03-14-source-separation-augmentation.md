# Reference: Audio Source Separation for Data Augmentation in Bird Audio Classification

**Source**: Deep web research across multiple papers, repos, and competition write-ups
**Date Captured**: 2026-03-14
**Type**: deep-research

## Summary

Audio source separation (MixIT, Demucs, Biodenoising) can serve as a data augmentation
strategy for bird audio classification, particularly for bridging the domain gap between
focal recordings and soundscapes. The key paper (Denton et al., arXiv 2021 / ICASSP 2022) demonstrated that
classifying separated sources via max-pooling over channels improves precision on real-world
soundscape datasets. However, no top BirdCLEF competition solution has actually used source
separation as augmentation -- pseudo-labeling and standard noise mixing dominate instead.

## Key Points

- Bird-MixIT (Google, arXiv 2021 / ICASSP 2022) is the canonical approach: separate into 4-8 sources, classify each, take max activation across channels. Improves precision on 3 independent soundscape datasets.
- No top BirdCLEF solution (2021-2025) has used source separation for augmentation. Top solutions rely on pseudo-labeling, noise mixing, and domain adaptation via model distillation.
- Biodenoising (Earth Species Project, 2024) adapts Demucs dns48 for animal vocalizations without clean data, using a teacher-student two-step training approach.
- Co-occurrence-aware mixing (Mix2, 2024) is an active research direction but not yet deployed at competition scale.
- Mamba/SSM architectures show strong promise for source separation (linear complexity, efficient long-range modeling) and bioacoustics classification (BioMamba matches transformer AVES with less VRAM).

## Relevant Sections

### 1. Source-Separation-Based Augmentation Strategies

**Bird-MixIT (Denton, Wisdom, Hershey 2021) -- the foundational paper**
- Paper: [arXiv:2110.03209](https://arxiv.org/abs/2110.03209) / [Project page](https://bird-mixit.github.io/)
- Architecture: TDCN++ masking network with learned filterbank (1D convolution, not STFT). The bird-specific model uses N=256 coefficients, 1 ms window, 0.5 ms hop at 22.05 kHz. Predicts M masks via sigmoid, applies mixture consistency projection.
- Two model variants: **4-output sources** and **8-output sources** (checkpoints available at [google-research/sound-separation](https://github.com/google-research/sound-separation/tree/master/models/bird_mixit)).
- Trained on birdsong from Xeno-Canto and Macaulay Library using MixIT (unsupervised -- no clean sources needed). Outperforms general AudioSet-trained MixIT by >5 dB SI-SNR.
- **Classification pipeline**: Run separator on each audio clip, producing 4 (or 8) source waveforms. Run classifier on each source AND the original. Take **element-wise maximum** of model activations (logits/probabilities) across all channels. This "max-over-channels" approach captures quiet/background birds that are drowned out in the original.
- Additional improvements: taxonomic classification, random low-pass filter augmentation, channel normalization.
- Key insight: separation helps most for **quiet and background birds** and **overlapping vocalizations**.

**Augmentation strategy variants** (not all from Bird-MixIT paper, synthesized from literature):

| Strategy | Description | Complexity |
|----------|-------------|------------|
| Separate-and-reclassify | Run separator, classify each source, max-pool | Inference only |
| Background swap | Separate fg/bg from soundscape, replace bg with soundscape bg on focal recording | Pre-compute sources |
| SNR augmentation | Separate fg/bg, remix at varied SNR levels (simulate distance) | Pre-compute sources |
| Smart mixup | Mix separated foregrounds from multiple species with co-occurrence constraints | Pre-compute + co-occurrence data |
| Distance simulation | Attenuate separated foreground signal, add reverb/environmental noise | Pre-compute sources |

**Biodenoising (Earth Species Project, 2024)**
- Paper: [arXiv:2410.03427](https://arxiv.org/abs/2410.03427) / [GitHub](https://github.com/earthspecies/biodenoising)
- Uses pre-trained speech enhancement model (demucs dns48 or CleanUNet) as starting point.
- Two-step training: (1) Apply speech enhancer to get pseudo-clean targets, (2) Retrain on synthetic mixtures of noise + pseudo-clean targets.
- Can be fine-tuned on your own recordings for domain adaptation.
- Pip installable: `pip install biodenoising` / separate [inference repo](https://github.com/earthspecies/biodenoising-inference).
- Model is small (based on Demucs dns48) -- designed to be "fast to train."
- Outputs a single denoised signal (not multiple sources like MixIT).
- Best for: cleaning focal recordings, not for separating overlapping species.

**RemixIT (Tzinis et al., 2022)**
- Paper: [arXiv:2202.08862](https://arxiv.org/abs/2202.08862)
- Self-supervised domain adaptation for separation: teacher model on out-of-domain data generates pseudo-targets, then permute/remix estimated clean+noise to create bootstrapped training data for student.
- Directly applicable to focal-to-soundscape domain adaptation: train teacher on focal (cleaner), adapt to soundscape domain without labels.
- Can be combined with any separation model architecture.

### 2. Co-Occurrence-Aware Mixing

**Why it matters**: Random mixup creates unrealistic combinations (e.g., mixing a tropical bird with an arctic species). Co-occurrence-aware mixing creates synthetic soundscapes that match the distribution of real soundscapes.

**Mixture of Mixups (Mix2) for Anuran Sounds (2024)**
- Paper: [arXiv:2403.09598](https://arxiv.org/abs/2403.09598)
- Combines Mixup, Manifold Mixup, and MultiMix for multi-label bioacoustics.
- Tested on AnuraSet (anuran species classification with class imbalance).
- **Future work explicitly proposes**: integrating ecological info on diel temporal activity patterns and temporal acoustic niche segregation to only mix species co-occurring spatially and temporally.
- Shows Mix2 is effective for rare classes and multi-label scenarios.

**How to get co-occurrence data**:

1. **BirdCLEF metadata**: Training CSV includes `latitude`, `longitude`, `date` per recording. Group by location/date to find which species co-occur in training data.
2. **eBird Basic Dataset (EBD)**: Tab-separated file with all validated sightings. Each row = one species in one checklist. Group by `SAMPLING EVENT IDENTIFIER` to build co-occurrence matrix. Request access at [ebird.org/science/use-ebird-data](https://ebird.org/science/use-ebird-data/download-ebird-data-products).
3. **eBird API**: Limited to recent/summary data. Not ideal for building full co-occurrence matrices.
4. **Kaggle notebook**: [Species Co-Occurrence from eBird Checklists](https://www.kaggle.com/code/lobrien/species-co-occurrence-from-ebird-checklists) -- directly builds co-occurrence matrices from eBird data.
5. **eBird Status Data Products**: Species distributions across 3km grid, 52 weeks/year, 2980 species. Can derive temporal co-occurrence from overlapping ranges.

**Implementation sketch for co-occurrence-aware mixing**:
```
1. Build NxN co-occurrence matrix C from eBird checklists for target region
2. For each training sample of species A:
   a. Sample species B with probability proportional to C[A, B]
   b. Optionally constrain by month/time-of-day
   c. Mix separated foreground of A with separated foreground of B
   d. Add background noise from target soundscape domain
   e. Label as multi-label {A, B}
```

### 3. BirdCLEF Competition Solutions Using Source Separation

**Short answer: None of the top solutions used source separation for augmentation.**

| Year | Top Technique | Source Separation Used? |
|------|--------------|----------------------|
| 2021 | Pseudo-labeling, noise augmentation | No |
| 2022 | Ensemble, noise mixing | No |
| 2023 | Pre-trained models, pseudo-labels | No |
| 2024 | Pseudo-labeling + distillation (1st), additive mixup (3rd) | No |
| 2025 | Iterative pseudo-labeling (3-4 rounds), power scaling, model soup | No |

**What top solutions actually used for domain adaptation** (BirdCLEF 2024-2025):
- **Pseudo-labeling on unlabeled soundscapes** (dominant technique): Train on focal, predict on soundscape clips, add high-confidence pseudo-labeled clips to training set. Iterate 3-4 rounds.
- **Additive mixup**: Mix two training clips, max their labels. Simple and effective.
- **Background noise mixing**: Mix training samples with background from target domain.
- **Time shifting** (1-second window).
- **Human voice augmentation** (BirdCLEF 2025 top-5).
- **Power adjustment / temporal smoothing** at inference.
- **Model distillation** for domain shift mitigation.
- **Silero VAD** for detecting/removing human speech segments.

**Why hasn't source separation been used in competitions?**
- Computational overhead: running separator on every training clip is expensive.
- Pseudo-labeling achieves similar domain adaptation with much less complexity.
- 4-8x data expansion from separated sources is hard to manage in Kaggle's resource constraints.
- Separation quality is imperfect -- artifacts may hurt training.
- The "max-over-channels" approach is primarily an inference-time technique, not a training augmentation.

### 4. Computational Considerations

**Bird-MixIT**:
- Framework: TensorFlow 1.x checkpoint format (requires TF >=2.14 for CUDA 12.x compatibility, running in v1 compat mode).
- Model checkpoints: `output_sources4/model.ckpt-3223090` and `output_sources8/model.ckpt-2178900`.
- No published inference speed benchmarks, but TDCN++ is relatively lightweight. Rough estimate: ~0.01-0.05s per 3s clip on GPU (RTX 6000), so ~13-50 GPU-hours for the full focal dataset.
- Bird-specific model operates at **22.05 kHz** (the generic AudioSet model uses 16 kHz). BirdCLEF data (32 kHz) must be resampled before separation.
- The "939K clips" figure comes from BirdNET's 3s/1s windowing of focal recordings, not the raw recording count. Processing with 4 outputs produces ~3.76M source clips.
- **Recommendation**: Pre-compute and store separated sources on disk. On-the-fly separation during training is impractical.

**Demucs / Biodenoising**:
- Demucs has ONNX export support: [demucs.onnx](https://github.com/sevagh/demucs.onnx) (C++ ONNX/ORT inference), also [demucs-onnx for WebGPU](https://github.com/gianlourbano/demucs-onnx).
- Demucs v4 has been successfully converted to fully self-contained ONNX model (STFT/ISTFT rewritten for ONNX compatibility).
- TensorRT FP16 compilation gives ~25-30x speedup over naive ONNX Runtime inference.
- Biodenoising uses Demucs dns48 (small model, "fast to train") -- designed for real-time speech enhancement, so CPU inference should be viable.
- **CPU inference**: No published benchmarks for bioacoustics specifically, but Demucs dns48 was designed for real-time speech enhancement at 16kHz, suggesting <1x real-time on modern CPU is achievable.

**Data expansion management**:
- 4-source MixIT: 4x expansion (4 sources per clip).
- 8-source MixIT: 8x expansion.
- Practical approach: Pre-compute separated sources, store as WAV/FLAC on scratch storage.
- For BirdCLEF 2026 focal data (~939K 3s clips at 32 kHz): raw audio ~180 GB. 4-source separation at 22.05 kHz: ~496 GB; if upsampled back to 32 kHz: ~720 GB.
- Alternative: Only separate soundscape data (smaller, higher value for domain adaptation).
- Selective separation: Run BirdNET/Perch first, only separate clips with detected overlapping species.

### 5. Connection to Sequence Modeling (Mamba/SSM)

**Mamba for source separation** (recent work):
- MSNet (2025): Dual-path Mamba for music source separation. Time-frequency decoupling with alternating Mamba blocks on temporal and frequency paths. Linear complexity O(L) vs O(L^2) for transformers. Published in [Scientific Reports](https://www.nature.com/articles/s41598-025-20179-3).
- U-Mamba-Net: U-Net + Mamba for speech separation, 8.50 dB SI-SNR on Libri2Mix with only 4.4M params and 2.5 GMACs.
- SPMamba: "State-space model is all you need in speech separation" -- validates Mamba's effectiveness for separation tasks.
- Omni-directional Mamba attention for speech separation (2026).

**BioMamba for bioacoustics classification**:
- Paper: [arXiv:2512.03563](https://arxiv.org/abs/2512.03563) (Dec 2025)
- Mamba-based audio LLM pre-trained with self-supervised learning, fine-tuned on BEANS benchmark.
- **Matches transformer-based AVES** while consuming significantly less VRAM.
- More data-efficient: stronger results than transformers on smaller data subsets.
- Validates that Mamba is a viable architecture for bioacoustics.

**How separated sources help temporal modeling**:
- **Cleaner input sequences**: Mamba's selective SSM mechanism works better when each token represents a single source rather than a mixture. Separation reduces the complexity of what the SSM needs to track.
- **Per-source temporal patterns**: Bird vocalizations have characteristic temporal patterns (song structure, call repetition). Separated sources let the model focus on one species' temporal pattern at a time.
- **Co-occurrence as implicit supervision**: If training data includes realistic multi-species mixtures (via co-occurrence-aware mixing), the model learns to handle temporal interleaving of species -- directly relevant for soundscape evaluation.
- **Directionality is task-dependent**: Some audio Mamba work finds unidirectional SSMs sufficient for causal separation tasks, but this project's SSAMBA uses bidirectional Mamba (which has proven beneficial for classification). The optimal directionality likely depends on the downstream task.

### 6. CPU-Viable Approaches

| Model | CPU Viable? | ONNX? | Notes |
|-------|------------|-------|-------|
| Bird-MixIT (TDCN++) | Likely yes | TF1 checkpoint (convertible to ONNX via tf2onnx, requires SavedModel conversion first) | Designed for research, not optimized for production |
| Demucs dns48 | Yes | Yes ([demucs.onnx](https://github.com/sevagh/demucs.onnx)) | Designed for real-time speech; small model |
| Biodenoising | Yes | Via Demucs dns48 | Pip installable, inference repo available |
| Demucs v4 (Hybrid Transformer) | Slower on CPU | Yes (ONNX conversion done for [Mixxx GSOC 2025](https://mixxx.org/news/2025-10-27-gsoc2025-demucs-to-onnx-dhunstack/)) | Larger model, transformer component is expensive |
| CleanUNet | Likely yes | Should be exportable | Alternative to Demucs in biodenoising pipeline |

**Recommended CPU pipeline for BirdCLEF augmentation**:
1. Use Biodenoising (Demucs dns48 backbone) for denoising focal recordings -- pip installable, fast.
2. Use Bird-MixIT 4-source for separating soundscape data -- pre-compute offline as batch job.
3. Store separated sources on scratch storage.
4. During training, load pre-computed sources and remix with co-occurrence constraints.

## Connections

- **Vendor docs**: [Bird MixIT](../../vendor/bird-mixit.md), [Biodenoising](../../vendor/biodenoising.md)
- **Status**: the standalone source-separation concept note was removed during documentation cleanup; this reference remains as background.
- **Related concept**: [Soundscape Domain Shift](../../concepts/soundscape-domain-shift.md), pseudo-labeling as alternative to source separation
- **Related experiments**: exp 010 (teacher labels -- BirdNET/Perch detections could guide selective separation), exp 013 (weak supervision -- separated sources could improve label quality), exp 009 (pretrain pipeline -- separated sources as pretraining data)

## Practical Recommendations for BirdCLEF 2026

### What to try (ordered by expected impact vs. effort):

1. **Inference-time separation + max-pooling** (low effort, proven): Run Bird-MixIT on test soundscapes, classify each source, max-pool predictions. This is the original Bird-MixIT approach and has proven improvements. Does not require any training changes.

2. **Background swap augmentation** (medium effort): Pre-compute Biodenoising on focal recordings to extract clean foregrounds. Mix with soundscape backgrounds at varied SNR levels. This directly addresses the focal-to-soundscape domain gap without needing co-occurrence data.

3. **Pseudo-labeling remains king** (proven in competitions): Source separation is complementary to, not a replacement for, pseudo-labeling. The 2024-2025 winning strategies all use pseudo-labeling as the primary domain adaptation mechanism. Consider source separation as a way to improve pseudo-label quality on soundscapes.

4. **Co-occurrence-aware mixing** (high effort, speculative): Build co-occurrence matrix from BirdCLEF metadata or eBird checklists. Use separated foregrounds to create realistic multi-species mixtures. Potentially high value but unproven at competition scale.

### What to avoid:

- On-the-fly source separation during training (too expensive).
- Using separation as the sole domain adaptation strategy (pseudo-labeling is proven more effective).
- Storing 8-source outputs for the full dataset (disk explosion -- use 4-source or be selective).

## Sources

- [Bird-MixIT project page](https://bird-mixit.github.io/)
- [Denton et al. 2021 - Improving Bird Classification with Unsupervised Sound Separation](https://arxiv.org/abs/2110.03209)
- [Google Research sound-separation repo (Bird-MixIT checkpoints)](https://github.com/google-research/sound-separation/tree/master/models/bird_mixit)
- [Separating Birdsong in the Wild - Google Research Blog](https://research.google/blog/separating-birdsong-in-the-wild-for-classification/)
- [Biodenoising - Earth Species Project](https://github.com/earthspecies/biodenoising)
- [Biodenoising paper (arXiv:2410.03427)](https://arxiv.org/abs/2410.03427)
- [Biodenoising inference repo](https://github.com/earthspecies/biodenoising-inference)
- [RemixIT - Continual self-training via bootstrapped remixing](https://arxiv.org/abs/2202.08862)
- [Demucs ONNX export (C++)](https://github.com/sevagh/demucs.onnx)
- [Demucs ONNX for WebGPU](https://github.com/gianlourbano/demucs-onnx)
- [Demucs v4 to ONNX (Mixxx GSOC 2025)](https://mixxx.org/news/2025-10-27-gsoc2025-demucs-to-onnx-dhunstack/)
- [Mixture of Mixups for Anuran Sounds (arXiv:2403.09598)](https://arxiv.org/abs/2403.09598)
- [Species Co-Occurrence from eBird Checklists (Kaggle notebook)](https://www.kaggle.com/code/lobrien/species-co-occurrence-from-ebird-checklists)
- [eBird Data Products](https://ebird.org/science/use-ebird-data/download-ebird-data-products)
- [BirdCLEF 2024 3rd place solution](https://zenn.dev/yuto_mo/articles/53ed2b27c1f52b)
- [BirdCLEF 2025 2nd place repo](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place)
- [BirdCLEF 2025 top-5 overview](https://tekkix.com/articles/ai/2025/07/birdclef-2025-overview-of-the-competition-a)
- [MSNet - Mamba for music source separation (2025)](https://www.nature.com/articles/s41598-025-20179-3)
- [State Space Models for Bioacoustics (BioMamba, arXiv:2512.03563)](https://arxiv.org/abs/2512.03563)
- [MixIT original paper (NeurIPS 2020)](https://arxiv.org/abs/2006.12701)
- [Tackling Domain Shift in Bird Audio Classification](https://ceur-ws.org/Vol-4038/paper_256.pdf)
- [Improving birdsong classification with combined augmentation](https://www.sciencedirect.com/science/article/pii/S1574954124002413)
- [Dynamic Multi-Species Bird Soundscape Generation](https://arxiv.org/html/2511.19275v1)
- [BioCPPNet - automatic bioacoustic source separation](https://www.nature.com/articles/s41598-021-02790-2)

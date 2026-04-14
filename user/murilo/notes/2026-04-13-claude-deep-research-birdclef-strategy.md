# Claude Deep Research: BirdCLEF 2026 Strategy (0.906 → 0.93+)

- **Source**: Claude Deep Research (unbiased — no context of our experiments)
- **Prompt**: Strategy to go from 0.906 to 0.93+ single model in BirdCLEF 2026
- **Date**: 2026-04-13
- **Context given**: Competition link, discussion thread, current score (0.906)
- **Context NOT given**: Our specific experiments, failures, architecture details

---

# Bridging 0.906 to 0.93+ in BirdCLEF 2026: a complete technical playbook

**Iterative pseudo-labeling with soft targets is the single highest-leverage technique separating your current 0.906 from the 0.93+ tier.** In BirdCLEF 2025, the 1st-place solution (0.933 private) achieved its critical +0.032 jump from 0.898 to 0.930 through four rounds of pseudo-labeling with power scaling. The 2nd-place solution (0.928) documented a +0.017–0.024 gain from three pseudo-labeling iterations alone. Combined with transfer learning (+0.014–0.018), postprocessing (+0.011–0.015), and proper augmentation (MixUp alone worth +0.036 on private), these techniques stack to close the gap. Your SED with EfficientNet-class backbone is already a strong foundation — the remaining gains come from systematic domain adaptation, not architectural overhaul.

---

## The 0.90-to-0.93 gap is about domain adaptation, not architecture

Analysis of BirdCLEF 2025's top solutions reveals a consistent pattern: the architecture matters less than how effectively you bridge the focal-to-soundscape domain gap. The 2nd-place team's detailed ablation tells the story clearly:

| Component added | Private LB | Gain |
|---|---|---|
| Baseline (EfficientNetV2-S, enhanced) | 0.871 | — |
| + Transfer learning (819K recordings) | 0.889 | +0.018 |
| + Pseudo-labels (3 iterations) | 0.909 | +0.020 |
| + Postprocessing (TopN=1) | 0.924 | +0.015 |
| + Ensemble (3 experiments × 5 folds) | **0.928** | +0.004 |

The 1st-place progression was even more dramatic: baseline 0.872 → pseudo-labeling round 1 + MixUp + StochasticDepth → 0.898 → four rounds of pseudo-labeling with power scaling → **0.930** → separate insect/amphibian pipeline → **0.933**. Notice that **~80% of the improvement above baseline came from pseudo-labeling iterations** — not from changing the backbone, adding exotic losses, or inventing new architectures.

Your current SED + EfficientNet at 0.906 likely already captures the architecture and basic training gains. The path forward is clear: iterative pseudo-labeling, transfer learning, and smart postprocessing.

---

## Priority 1: Build a multi-round pseudo-labeling pipeline

This is your single highest-ROI investment. The proven recipe from the BirdCLEF 2025 2nd-place solution (documented with full ablation):

**Round 0 — Teacher ensemble creation.** Train 6–8 diverse models on your labeled data. Use both your EfficientNet SED backbone and a complementary architecture like **eca_nfnet_l0** or **EfficientNetV2-S** (from `timm`). Vary spectrogram parameters across models: one set at n_fft=1024, hop=320, n_mels=256 (fine-grained); another at n_fft=2048, hop=512, n_mels=128 (coarse context). This diversity produces better-calibrated pseudo-labels.

**Round 1 — Generate soft pseudo-labels.** Run the teacher ensemble on all ~600 unlabeled soundscapes, producing per-species probabilities for each 5-second chunk. Apply confidence filtering: **discard chunks where max predicted probability < 0.5**; within kept chunks, **zero out per-class probabilities below 0.1**. Critically, **keep predictions as soft targets** (continuous probabilities), not binarized hard labels. The 2nd-place team selected 4,430 files / 19,405 chunks in round 1.

**Rounds 2–3 — Iterative refinement.** Retrain models on combined labeled + pseudo-labeled data, then re-predict on unlabeled soundscapes with the improved ensemble. Each iteration surfaces additional chunks that now pass the confidence threshold (round 2: 1,483 files / 3,750 chunks; round 3: 1,437 files / 4,108 chunks). The 1st-place solution ran **four total rounds** with diminishing but still meaningful returns.

**Sampling strategy.** When pseudo-labeled data exists for a species, replace the original training sample with a pseudo-labeled one with **40% probability**. This maintains balance between clean focal supervision and noisy-but-in-domain soundscape data.

**Power scaling** (used by the 1st-place solution) adjusts sample weights across rounds: `w_i = (count_i / total)^(-0.5)`, preventing the model from becoming overconfident on head classes across iterations. This was part of the critical 0.898→0.930 jump.

For the 159 of 234 species with zero soundscape training data, pseudo-labeling is especially critical — it is your only mechanism to generate in-domain training signal for these species.

---

## Priority 2: Transfer learning from a large bird audio corpus

The 2nd-place team pretrained on **819,032 recordings spanning 7,489 species** collected from Xeno-Canto and prior BirdCLEF competitions (2020–2024). This single step improved their EfficientNetV2-S from 0.859 to 0.889 private (+0.030). The protocol:

**Dataset construction.** Gather recordings from BirdCLEF 2020–2025 competitions plus Xeno-Canto. **Exclude species overlapping with BirdCLEF 2026's 234 target species** to avoid data leakage into validation. Remove species with fewer than 10 recordings. Cap at **500 recordings per species** (most recent) to manage training time.

**Pretraining.** Use your existing SED architecture with the same spectrogram parameters. Train for 50 epochs on this large corpus without class balancing. Extract the **backbone weights only** — reinitialize the classification head for fine-tuning on competition data.

**Fine-tuning.** Apply your full training recipe (augmentations, class balancing, loss function) on BirdCLEF 2026 data, starting from the pretrained backbone. No special learning rate adjustments needed — the 2nd-place team used the same LR schedule for both stages.

An alternative fast-track: the **Perch v2** embedding model (Google, EfficientNet-B3, 1,536-dim embeddings, ~14,795 species) already achieves 0.908 on BirdCLEF 2026's public LB with just a linear head. You could use Perch embeddings as features in a secondary model, or use Perch as a teacher for generating pseudo-labels on unlabeled soundscapes. Perch embeddings are pre-computed at 5-second granularity and are linearly separable by design.

---

## Priority 3: PostProcessing with the TopN temporal consistency method

The 2nd-place team's **TopN postprocessing** added +0.011–0.015 to private LB — a massive gain for a post-hoc technique. The formula:

```
postproc_prob[file, segment, class] = prob[file, segment, class] × max(prob[file, :, class])
```

For each 5-second segment, multiply the predicted probability for species *c* by the **maximum probability for species *c* anywhere in the same audio file** (TopN with N=1). The intuition: if a species appears strongly in one segment of a recording, adjacent segments with weaker detections get boosted; if a species never appears confidently anywhere in the file, spurious low-confidence predictions get suppressed.

Apply postprocessing **after ensembling** (the team confirmed this ordering outperforms per-model postprocessing). An additional refinement is **neighbor window smoothing**: weight adjacent 5-second windows with factors like [0.1, 0.2, 0.4, 0.2, 0.1], exploiting temporal continuity in bird vocalizations.

---

## Priority 4: Get augmentation right — MixUp is non-negotiable

The ablation data is unequivocal: **removing MixUp caused a -0.036 drop on private LB** (0.871 → 0.835), making it the single most impactful augmentation. The proven augmentation stack:

- **MixUp** (50% probability): Add two raw waveforms together; combine labels via element-wise maximum (not averaging). This simulates multi-species soundscape conditions during training — directly addressing the domain shift. Use alpha from a uniform distribution, not Beta.
- **Background noise mixing** (50% probability): Add real soundscape noise or ESC-50 environmental sounds to training clips. Split equally between competition soundscapes and ESC-50 samples. This teaches the model to extract bird signals from realistic noise.
- **SpecAugment**: Standard time and frequency masking on mel spectrograms. Worth ~+0.010 on private.
- **RandomFiltering**: Simplified random equalizer simulating recording equipment variation. Worth ~+0.010 on private.
- **Gain adjustment**: Random volume variation (-12 to +12 dB) for amplitude robustness.

Avoid over-augmenting. The BirdCLEF 2025 top-2% competitor found that heavy augmentation probability confused the model. **Pitch shift, time stretch, and CutMix all failed** in multiple top solutions. Stick to the proven stack above.

---

## Priority 5: Architecture and loss function refinements

Your SED + EfficientNet backbone is solid. Here are targeted refinements with documented gains:

**Backbone upgrade.** Consider switching to **tf_efficientnetv2_s_in21k** (from timm), which was the top-performing backbone in both the 1st and 2nd place 2025 solutions. Its ImageNet-21K pretraining provides richer initial features. Pair it with **eca_nfnet_l0** as a complementary architecture for ensemble diversity.

**Classification head.** The 2nd-place architecture: GeM pooling across the frequency dimension → 512-dim hidden layer with ReLU → Dropout 0.25 after CNN encoder, 0.5 after hidden layer. This head **omits RNN blocks** from traditional SED — attention-based aggregation alone is sufficient and faster.

**Loss function.** Use a **linear combination of BCE and Focal Loss** with **label smoothing alpha=0.05**. The 1st-place team explored **SoftAUCLoss** (directly optimizing the competition metric via pairwise differences + log-loss), which natively supports soft pseudo-labels. Since the metric is ROC-AUC, a loss that optimizes ranking quality outperforms losses that optimize probability calibration. An important BirdCLEF 2024 finding: CrossEntropyLoss with **sigmoid at inference** (not softmax) dramatically outperformed BCE in that year's competition, suggesting this should be tested.

**Spectrogram parameters.** Converged "best" settings from multiple winning solutions: sample rate **32,000 Hz**, n_mels **128**, n_fft **2,048**, hop_length **512**, f_min **20 Hz**, top_db **80**, normalized and scaled to [0, 1]. For ensemble diversity, add a second spectrogram config at n_fft=1024, hop_length=320, n_mels=256 (higher time resolution).

**Training recipe.** 50 epochs with cosine LR decay to 1e-6, batch size 64. Use **AdamW (lr=1e-4)** for EfficientNetV2-S or **RAdam (lr=1e-3)** for NFNet-L0. Apply **sqrt-balanced sampling** (gamma = -0.5) to handle the long tail: `w_i = (c_i / sum(c_j))^(-0.5)`, plus duplicate classes with <100 samples up to 100.

---

## Priority 6: CPU inference optimization within 90 minutes

The time budget is tight: ~600+ test soundscape files must be processed on CPU in 90 minutes. The proven optimization stack:

**Model export.** Convert PyTorch models to **ONNX** first, then to **OpenVINO IR format with FP16 compression**. The 2nd-place team used this exact pipeline. OpenVINO is optimized for Intel Xeon CPUs (which Kaggle uses) and provides **2-3x speedup** over raw PyTorch.

**Spectrogram caching.** Compute mel spectrograms **once** for each audio file and reuse across all ensemble models. If using multiple spectrogram configurations, cache each configuration separately. Store all spectrograms in RAM (competition data fits comfortably).

**Model selection for inference.** Budget your time carefully. A single EfficientNet-B0 with ONNX processes the full test set in ~15-20 minutes. A 5-fold ensemble takes ~40 minutes. This leaves room for 2-3 diverse model families within 90 minutes. **EfficientViT-b0** is the fastest option (5 folds in 40 minutes via ONNX) while maintaining competitive accuracy. **MNASNet-100** is another fast alternative used by 2024's 3rd-place team.

**Inference strategy.** Process full 1-minute audio files as batches of twelve 5-second windows rather than individual chunks. Use **joblib** for parallel spectrogram computation. Pre-convert all audio to HDF5 format for byte-wise random access. Leave a **10-minute safety margin** below the runtime cap.

**Test-time augmentation (TTA).** The 2nd-place team's TTA with 2.5-second left/right shifts improved from 0.91 to 0.922 — a **+0.012 gain**. This means predicting each 5-second window three times (centered, shifted left 2.5s, shifted right 2.5s) and averaging. Budget ~3x the base inference time. Only use TTA if your time budget allows it after accounting for the full ensemble.

---

## A concrete implementation roadmap

Based on all documented evidence, here is the priority-ordered plan to move from 0.906 to 0.93+:

**Week 1: Foundation improvements (+0.01-0.02 expected).** Verify MixUp at 50% probability with element-wise max labels is active. Add background noise mixing from competition soundscapes + ESC-50 at 50% probability. Implement BCE + Focal Loss with label smoothing alpha=0.05. Apply sqrt-balanced sampling (gamma = -0.5). Add secondary labels with equal weight to primary. These are "table stakes" improvements that compound.

**Week 2: Transfer learning (+0.015-0.020 expected).** Collect recordings from BirdCLEF 2020-2025 and Xeno-Canto, excluding the 234 target species. Pretrain your SED backbone on this corpus. Fine-tune on competition data. Alternatively, integrate Perch v2 embeddings as a secondary feature source or pseudo-label generator.

**Week 3: Pseudo-labeling pipeline (+0.020-0.030 expected).** Build the iterative pipeline: train 6-8 diverse teachers → predict unlabeled soundscapes → filter (max_prob >= 0.5, zero below 0.1) → retrain with 40% pseudo-label sampling → repeat 2-3 times. Use soft targets throughout. Apply power scaling for class balance across rounds.

**Week 4: PostProcessing and inference (+0.012-0.015 expected).** Implement TopN=1 postprocessing. Add neighbor window smoothing. Convert final models to ONNX → OpenVINO FP16. Implement TTA with +/-2.5s shifts if time budget permits. Optimize spectrogram caching and parallel processing.

**Week 5: Ensemble and polish (+0.004-0.005 expected).** Combine your best 2-3 model families (varying backbone, spectrogram params, and pseudo-label vintage). Use simple averaging of predictions. Select the optimal subset via OOF macro ROC-AUC using Optuna. Apply postprocessing after ensembling.

---

## Techniques that consistently fail in BirdCLEF

Multiple top competitors independently confirmed these approaches **did not help** or actively hurt performance, saving you from dead-end experiments:

- **Raw audio models** (Wav2Vec, HuBERT, audio transformers without mel spectrograms): consistently underperformed mel-spectrogram CNNs. The BirdCLEF 2025 top-2% competitor's Wav2Vec + GNN approach scored only 0.60.
- **CutMix on spectrograms**: unlike MixUp on waveforms, CutMix has been neutral or negative across multiple years.
- **Heavy augmentation** (strong pitch shift, time stretch, aggressive noise, blur): over-regularizes and confuses the model.
- **Additional Xeno-Canto data for the target species** hurt 2024's 1st place — quality filtering matters more than quantity.
- **Complex architectures** (multi-stage detectors, separate bird-call detection pre-filters, adversarial domain adaptation): "simple, careful engineering beats complexity" is the consistent lesson.
- **Training beyond 50 epochs** or **SWA/EMA** on this task: minimal or negative returns documented.
- **Softmax at inference** (when training with CE loss): 2024's 1st place discovered that switching to **sigmoid at inference** was critical, even when training with CrossEntropyLoss.

---

## Conclusion: the three-technique stack that closes the gap

The evidence from three years of BirdCLEF winning solutions converges on a clear hierarchy. **Iterative pseudo-labeling** (3-4 rounds with soft targets and confidence filtering) is the dominant technique, consistently providing +0.02-0.03 AUC. **Transfer learning** from a large bird audio corpus adds +0.015-0.020. **TopN postprocessing** contributes another +0.011-0.015. Together, these three techniques account for roughly +0.05-0.065 AUC improvement — more than enough to bridge your 0.906 to the 0.93+ tier.

The critical insight is that these gains are **multiplicative in effect but independent in implementation**: each technique addresses a different aspect of the domain shift problem. Pseudo-labeling provides in-domain training data for the 159 species with zero soundscape examples. Transfer learning provides a richer feature space for rare species. PostProcessing exploits temporal coherence that the model cannot learn from 5-second training clips. Stack all three, add proper augmentation and CPU optimization, and the 0.93 threshold becomes achievable with a single well-engineered model family.

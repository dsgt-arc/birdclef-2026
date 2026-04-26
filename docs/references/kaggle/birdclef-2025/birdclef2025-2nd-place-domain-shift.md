---
slug: "sydorskyi-2nd-place-2025"
type: "writeup"
competition: "birdclef-2025"
date_captured: "2026-04-26"
---

# BirdCLEF+ 2025 2nd Place — Sydorskyi & Gonçalves

**Paper**: [CEUR Vol-4038, paper_256](https://ceur-ws.org/Vol-4038/paper_256.pdf) — *"Tackling Domain Shift in Bird Audio Classification via Transfer Learning and Semi-Supervised Distillation: A Case Study on BirdCLEF+ 2025"*
**Authors**: Volodymyr Sydorskyi (KPI, Kyiv) & Fernando Gonçalves
**GitHub**: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
**Kaggle Discussion**: https://www.kaggle.com/competitions/birdclef-2025/discussion/583699
**Final score**: Public 0.925, **Private 0.928** (2nd place)

> Same lead author as the 2023 1st place solution (`docs/references/kaggle/birdclef-2023/birdclef2023-1st-place-solution.md`). The 2025 pipeline is an evolution of that recipe with two big additions: (a) a much larger in-domain pretraining corpus (819K recordings, 7,489 species) and (b) iterative semi-supervised distillation on unlabeled soundscapes — both motivated as **domain-shift mitigation**.

## Summary

The solution has five pillars:

1. **Strong baseline** — CNN backbone + GeMFreq + AttHead (a refined version of the 2023 recipe).
2. **In-domain pretraining** — pretrain backbones on a 7,489-species, 819,032-recording corpus assembled from prior BirdCLEF competitions + Xeno-Canto (everything except 2025 species, to avoid data leakage).
3. **Iterative pseudo-labeling / semi-supervised distillation** — **the largest single LB lever**. Use an 8-model ensemble to label unlabeled soundscapes, mix soft pseudo-labels into the training distribution at 40% sampling probability, retrain, repeat for 3 iterations. **+0.025 to +0.035 LB.**
4. **Postprocessing** — multiplicative TopN with N=1: `prob_a,s,c *= max(prob_a,:,c)`. **+0.01 to +0.015 LB**.
5. **Ensembling** — 5-fold × 3 selected experiments = 15 models, simple average. Optuna-selected ensemble vs Public-LB-selected ensemble (Public+0.929 Private vs Public+0.925 Private). +0.005 over best single.

Final ensemble = 15 models. Best single fold = 0.917 LB Public. Best ensemble = 0.925 Public / 0.929 Private.

## Architecture (`WaveCNNAttenClasifier`)

Raw waveform → on-the-fly nnAudio mel-spectrogram → 2D CNN backbone → GeMFreq pooling → attention-based SED head.

### Backbones (final ensemble)

- **EfficientNetV2-S** (`tf_efficientnetv2_s_in21k`) — primary
- **ECA-NFNet-L0** (`eca_nfnet_l0`) — diversity

Both pretrained from ImageNet-21k → in-domain pretraining → fine-tuning on 2025 with pseudo-labels. EfficientNetV2-S replaced ConvNeXt from the 2023 recipe; the paper notes both work well but EfficientNetV2-S converged faster after transfer-learning warm starts.

### Mel-spectrogram (nnAudio + AmplitudeToDB)

Identical across pretrain, fine-tune, and inference. Implemented as `TraceableMelspec` for ONNX export.

| Parameter | Value |
|---|---|
| Sample rate | 32,000 Hz |
| n_mels | 128 |
| n_fft | 2,048 |
| hop_length | 512 |
| f_min | 20 Hz |
| f_max | (none — uses full Nyquist) |
| normalized | True |
| top_db | 80 |
| amin | 1e-10 |
| Post-mel norm | AmplitudeToDB → standardize (zero-mean, unit-std) → min-max scale to [0, 1] |

Code: `code_base/models/wave_clasifier.py:WaveCNNAttenClasifier` + `code_base/models/blocks.py:NormalizeMelSpec`.

### Attention head (`AttHead`)

Closely matches the 2023 1st-place head:

```
GeMFreq(p=3, learnable)
→ Dropout(0.25)
→ Linear(in_chans, 512)
→ ReLU
→ Dropout(0.5)
→ [Conv1D attention branch:  out = num_class, kernel=1, +tanh]
→ [Conv1D value branch:      out = num_class, kernel=1]
```

**Clipwise prediction** (training and inference, when `train_period == infer_period`):

```python
clipwise_pred  = sum( sigmoid(value)  * softmax(tanh(att), dim=time), dim=time )
clipwise_logit = sum(         value   * softmax(tanh(att), dim=time), dim=time )
```

Clipwise pred is in **probability space** — sigmoid is applied to value before attention-weighted sum. Logit version exists for loss computation.

When `train_period > infer_period`, an additional center-cropped pathway computes `framewise_clipwise_pred_short = framewise_max * 0.5 + clipwise * 0.5` (the 75/25 blend from the 2023 paper has been changed to **50/50** here; `infer_framewise_max_coef=0.5` in the head). The selected experiments use `train_period = infer_period = 5.0`, so this branch is unused in the final models.

Code: `code_base/models/blocks.py:AttHead`.

### Hidden channels

- Fine-tuning configs: 512 (default `hidden_chans=512`)
- Pretraining configs: **1024** (`hidden_chans=1024` for the 7,591-class head — they widen to absorb the extra label space)

The classification head is **discarded** between pretrain and fine-tune; only the CNN backbone is carried over (`scripts/create_pretrain_backbone.py`).

## Loss function (`FocalLossBCE`)

```python
loss = bce_weight * BCEWithLogitsLoss + focal_weight * sigmoid_focal_loss
     = 1.0 * BCE + 1.0 * Focal(alpha=0.25, gamma=2.0)
```

Two losses **added together**, equal weight. Computed on `clipwise_logits_long` only (no framewise-max term in the selected configs — `loss_type="baseline"`). Applied with reduction="mean".

Code: `code_base/losses/focal_loss.py:FocalLossBCE`, `code_base/forwards/cls_forward.py:MultilabelClsForwardLongShort`.

### Multi-hot targets with secondary labels

Targets are built as multi-hot vectors over **primary + all secondary** labels (each at value 1.0, summed and clipped to [0,1]). XC recordings carry secondary labels (4,958 of 28,564 in 2025); iNat / CSA do not.

### Label smoothing (mass-preserving variant)

```python
all_tgt = all_tgt * (1 - 0.05) + 0.05 * all_tgt.sum() / num_classes
```

Note this is a **mass-preserving** form: smoothing is proportional to the number of active classes per sample, not a flat 1/K. So a sample with one positive gets `0.05/K` smoothing on negatives and `0.95 + 0.05/K` on the positive; a sample with two positives gets `0.10/K` on negatives.

Code: `code_base/datasets/wave_dataset.py:WaveDataset._prepare_target` (line ~599).

## Augmentation cascade

Applied **per sample in the dataloader** (CPU, before batch collate), with a few exceptions:

### 1. Random 5s crop

Random 5-second segment from full audio. **No "first 7s" or "last 7s" trick** — they tested it, found mild improvement, then tried surgical curation (cutting alien speech), found it hurt LB, abandoned it. Final pipeline: pure random 5s.

### 2. Audio-domain MixUp (p=0.5)

- Pick another sample uniformly (or via weighted sampler if `weights_path` set, but selected configs don't use this).
- Mix at **equal weight** (`alpha=None` → `lam=0.5`): `wave = 0.5 * wave1 + 0.5 * wave2`.
- Targets summed and clipped to [0, 1] (`target_aggregation="sum"`).
- Then renormalize the wave via `librosa.util.normalize` (peak-normalize to [-1, 1]).

This is **simple averaging-with-summed-labels MixUp on raw waveforms**, not Beta-distribution interpolation. The paper notes (Table 8): removing MixUp drops Public 0.835→0.835 (no change) but Private 0.871→0.835 (**−0.036**). MixUp is a Private-set robustness lever.

Code: `code_base/datasets/wave_dataset.py:WaveDataset.__getitem__` (lines 762-820).

### 3. Background noise (p=0.5, OneOf two sources)

Wrapped in `OneOf([source_A, source_B])` — picks one source uniformly each call:

**Source A — nocall soundscapes from BirdCLEF 2023** (`data/soundscapes_nocall/train_audio` + `data/v1_no_call_meta.csv`).

**Source B — ESC-50** filtered to 20 categories: `dog, rain, insects, hen, engine, hand_saw, pig, rooster, sea_waves, cat, crackling_fire, thunderstorm, chainsaw, train, sheep, wind, footsteps, frog, cow, crickets`.

Each source has its own `p=0.5`, but inside `OneOf` only one is picked, so the effective per-sample BG-noise application rate is `0.5 × 0.5 + 0.5 × 0.5 = 0.5`.

**Mixing math**:
```python
back_amp = uniform(0.25, 0.75)             # random amplitude per call
augmented = y * (1 - back_amp) + back_sample * back_amp
augmented = librosa.util.normalize(augmented)
```

This is **convex linear mixing**, not additive — the original signal is reduced as the noise is added. Equivalent to a random SNR draw. Paper Table 8: removing **RandomFiltering** alone changes Public +0.019, Private −0.010 (BG noise removal isn't ablated separately in the paper; the 2023-1st recipe used it, so 2025 inherited it).

Code: `code_base/augmentations/transforms.py:BackgroundNoise`.

### 4. SpecAugment (in-model, GPU)

Applied to the mel-spectrogram inside the model's forward pass when training:

- **FreqMasking**: max length 10, max masks 3, p=0.3
- **TimeMasking**: max length 20, max masks 3, p=0.3

Per the ablation (Table 8), removing SpecAugment moves Public **+0.010** but Private **−0.010** — Private-side robustness lever again.

Code: `code_base/augmentations/spec_augment.py:CustomFreqMasking, CustomTimeMasking`.

### 5. RandomFiltering (batch-level GPU augmentation)

Applied as `batch_aug` inside `MultilabelClsForwardLongShort`. Operates on the waveform via STFT → multiplicative filtering → iSTFT.

- 4 control points uniformly placed across frequency bins, each drawn from `uniform(min_db, 0) = uniform(-20, 0)` dB.
- Linearly interpolated to a per-frequency-bin gain, applied to the magnitude spectrogram.
- Inverse STFT recovers a re-equalized waveform.

The 2023 paper called this a "simplified random equalizer to simulate channel distortions." Paper ablation (Table 8): removing it moves Public +0.019, Private −0.010 — same Private-vs-Public trade as SpecAugment.

Code: `code_base/models/augmentations.py:RandomFiltering`.

### Augmentation pipeline summary

```
pick random 5s crop
→ MixUp(p=0.5, audio domain, equal-weight, label-sum)
→ BG noise (p=0.5, OneOf nocall-soundscape | ESC-50, convex blend at amp ~ U[0.25, 0.75])
→ peak-normalize waveform
→ [batch collate, move to GPU]
→ RandomFiltering (4 bands × U[-20, 0] dB)
→ nnAudio mel + AmplitudeToDB + standardize + min-max
→ SpecAugment (freq mask, time mask)
→ Backbone forward
```

## Training: two-stage transfer learning

### Stage 1 — Pretrain on 819K recordings, 7,489 species

- Collected taxonomy of all species from prior BirdCLEF competitions: 16,607 unique.
- Downloaded audio + metadata via Xeno-Canto, iNaturalist, CSA, prior BirdCLEF train sets.
- Pruned: removed all 2025 species (no leakage), corrupted files, invalid eBird codes, species with < 10 recordings → **819,032 recordings, 7,489 species**.
- 5% holdout for evaluation; only species with ≥ 100 recordings used for holdout metric (1,627 species in eval).
- Trained with the **same baseline recipe**, just no class balancing (the corpus is too large; balancing not needed at this scale).
- Best checkpoint by holdout macro ROC AUC. **Discard classification head; keep CNN backbone only** (`scripts/create_pretrain_backbone.py`).

Pretraining configs: `train_configs/pretrain_ebs_2025_extended.py`, `train_configs/pretrain_eca_2025_extended.py`. AdamW lr=2.5e-4, 50 epochs, BS=64, CosineAnnealingWarmRestarts to 1e-6.

> **Important note**: the authors did NOT include the pretraining audio in the released artifacts (~1TB). They provide metadata CSVs and download scripts (`scripts/download_all_xeno_canto.py`, `scripts/download_inaturalist.py`) so you can rebuild the corpus yourself. They also publish the pretrained checkpoints directly on Kaggle: [`bird-clef-2025-all-pretrained-models`](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models).

### Stage 2 — Fine-tune on 2025 + pseudo-labels

- **Data**: `train_audio_2025` (28,564 recordings, 206 species) + ~10K extra recordings from XC dump (28.03.2025) and previous-comp scraping (only the **two** highlighted "+impactful" sources from Table 3 are kept in `selected_ebs.py`; the all-data variant was *worse*: Public 0.835 → 0.832 in Table 7).
- **Same model** as pretrain (backbone weights loaded), but **head re-initialized** for 206 classes, hidden_chans dropped to 512.
- AdamW (EfficientNetV2-S) at **lr=1e-4** / RAdam (NFNet-L0) at **lr=1e-3**, both with `eta_min=1e-6` and `betas=(0.9, 0.999)` for AdamW. CosineAnnealingWarmRestarts, T_0 = 50 epochs × steps-per-epoch (so essentially one cosine cycle).
- **50 epochs**, batch size 64, **no AMP** (`precision="32-true"`).
- **No early stopping**; final checkpoint is `last.ckpt` (best-by-val also saved with `save_top_k=3`, but the inference configs use **`last.ckpt`** by default).

### Class balancing (sampler)

Per the paper Equation 1, weights are `(c_i / Σc_j) ^ γ`:

- **EfficientNetV2-S**: `class_weights_path="balanced"` → γ = −1 (fully balanced sampling).
- **ECA-NFNet-L0**: `class_weights_path="sqrt"` → γ = −0.5 (square-root balanced, gentler) **plus** explicit minor-class oversampling (the `OVERSAMPLE_CONFIG` dict in `selected_eca.py` duplicates rows for 58 specific rare species up to between 10 and 96 extra rows each). Different sampling strategies per backbone — the paper notes this is one source of ensemble diversity.

Implemented via PyTorch `WeightedRandomSampler`.

### CV split

- 5-fold StratifiedGroupKFold by `primary_label`, **grouped by recordist** (to prevent recordist leakage from XC's habit of uploading bursts of recordings in the same session).
- Two variants for handling rare species:
  1. Add them to validation (assess them, lose train data).
  2. Add them only to train (no eval, more data, less label noise).
- The selected configs use the **train-only** variant (`split_path` filenames contain `allbirds` or `allrarebirds_noleak`).

The paper Section 5.1 notes: **CV-LB correlation breaks down once Public > 0.9** (Pearson −0.13, Spearman −0.12 in that regime). They could not find a CV strategy that tracks LB once pseudo-labeling is engaged. **Optuna-on-OOF still produced the highest-Private ensemble** despite low individual correlation, suggesting OOF carries diversity signal even when it doesn't track absolute scores.

## Pseudo-labeling / semi-supervised distillation

This is **the largest single LB lever** (Table 5: Public 0.881 → 0.910–0.917, Private 0.889 → 0.910).

### Pipeline

1. Start with 8 baseline models (EffNetV2-S × 4 + NFNet-L0 × 4) trained from in-domain pretrained checkpoints.
2. Predict on **all 9,726 unlabeled soundscape files** (= 116,712 chunks of 5s each).
3. Mean-ensemble predictions across the 8 models.
4. **Pseudo selection logic** per chunk:
   - Compute `max(prob_c)` across all 206 classes.
   - **Discard** chunks where `max < 0.5`.
   - For retained chunks, **zero out** all class probabilities `< 0.1`.
   - Keep the rest as **soft labels** (not binarized).
5. Save the surviving (chunk → soft-label vector) pairs as the pseudo dataset.
6. Retrain models with pseudo-data fused into the training distribution (next subsection).
7. Generate next iteration of predictions, repeating with stronger models.

Iteration sample counts (Table 4):

| Iteration | Files selected | Chunks selected |
|---|---:|---:|
| 1 | 4,430 | 19,405 |
| 2 | 1,483 | 3,750 |
| 3 | 1,437 | 4,108 |

Each iteration's pseudo dataset is **cumulative** with prior iterations (with the "remove already-used files before re-predicting" logic in Figure 3).

### Two strategies (run independently then merged)

- **Full mode**: every iteration predicts on all soundscapes (excluding files already in the pseudo set), accumulates new ones. Each chunk gets predicted by 10 models (5 folds × 2 architectures).
- **OOF mode**: every iteration *re-predicts* every soundscape, with each fold's predictions coming **only from models not trained on that fold**. Each chunk gets only 2 models' predictions, but pseudo-labels refresh each round.

The two strategies were used independently (paper Table 5) and also **merged** into a single dataset using "remove from previous iteration" logic. Merging adds ensemble diversity. Best results (Table 5): **OOF I2 + Full I2** ensemble (Public 0.917, Private 0.910).

### Training-side fusion: 40% sampling probability + soft-label MixUp

This is the cleverest piece of the pipeline (`code_base/datasets/wave_dataset.py:WaveDataset._prepare_sample_target_from_idx`, lines ~604-735).

For every training sample drawn from the labeled set:

```python
if soundscape_df is not None and bernoulli(soundscape_pseudo_config["sampling_prob"] = 0.4):
    initial_class = original_sample.primary_label
    if initial_class in pseudo_birds:
        # Replace this sample with a pseudo-labeled chunk
        # whose primary class matches the originally-drawn class
        idx = random_choice(np.where(pseudo_df.primary_label == initial_class))
        sample = pseudo_chunk_at(idx)
        target = pseudo_chunk.soft_label_vector
        is_pseudo_sample = True
```

So **40% of training batches** carry a soundscape chunk with a soft-label vector (matched by primary class to preserve the class-balanced sampler's intent). The other 60% are normal hard-labeled focal samples.

**Then MixUp runs on top** at p=0.5 — and the MixUp here is the *interesting* fusion mechanism the paper emphasizes:

- A focal hard-labeled sample (target ∈ {0, 1}^206) and a soundscape soft-labeled sample (target ∈ [0, 1]^206) get audio-mixed at 50/50.
- Their targets are **summed and clipped to [0, 1]**, so the resulting target vector simultaneously carries "hard, primary species at 1.0" alongside "soft, multi-species probabilities from the teacher ensemble" in the same vector.
- This implicitly fuses **distillation** (matching the teacher's soft predictions) with **classification** (matching the ground-truth label) without needing a separate KD loss term.

Pseudo-label thresholds (`soundscape_pseudo_config`):

```python
{
    "primary_label_min_prob": 0.5,   # discard chunks with no high-confidence class
    "trim_min_prob": 0.1,            # zero out per-class probs below this
    "sampling_prob": 0.4,            # 40% chance to swap in pseudo per batch slot
}
```

Soft pseudo-labels were chosen explicitly **against** binarized pseudo-labels — paper Section 2 critiques [DS@GT 2024 (Miyaguchi/Cheung/**Gustineli**/Kim)](https://arxiv.org/abs/2407.06291) for using binarized pseudo-labels that produce overconfident predictions.

> **Direct callout to our prior work**: Sydorskyi cites Miyaguchi/Cheung/Gustineli/Kim 2024 (BirdCLEF 2024 DS@GT) as an example of "classical semi-supervised learning with binarized pseudo-labels" that overconfidently predicts. The 2025 2nd-place pipeline is explicitly engineered to fix that failure mode. Worth knowing as we revisit pseudo-labeling for 2026.

## Postprocessing — TopN, N=1

The 2nd-place team tested four PP methods (Section 5.6):

1. **Mean** — `prob_a,s,c *= mean_s'(prob_a,s',c)`
2. **TopN** — `prob_a,s,c *= mean over top-N segments(prob_a,:,c)`
3. **Convolution** — temporal smoothing across neighboring segments
4. **L2 model** — train a second-stage CNN on pseudo-labeled chunks to model species-specific rhythmic patterns

**Final winner: TopN with N=1**, applied at the **ensemble level only** (applying it per-model regressed). PP gain: **+0.01 to +0.015 LB** (Table 5: Full I2 → Full I2 + PP: Public 0.910 → 0.918, Private 0.909 → 0.924).

The L2 model didn't beat TopN N=1, even though it explicitly modeled per-species call rhythms.

Pipeline order: **PP → rank transform → arithmetic average** (when ensembling). Applying PP after rank-transform was tried but degraded results.

## Ensembling

- Per experiment: 5 folds × 1 architecture = 5 models, simple arithmetic mean.
- Final ensemble: **3 selected experiments × 5 folds = 15 models**.
- Selection strategies:
  - **Public-LB selection** → Public 0.925 / Private **0.928**.
  - **Optuna optimizing OOF AUROC** → Public 0.921 / Private **0.929**. Despite weaker CV-LB correlation, Optuna-on-OOF won Private. The paper interpretation: OOF still captures diversity / model-correlation signal even when it doesn't track absolute scores.

Single-model gain from ensembling: **only +0.005**. The single models already score Public 0.917, so the ensemble is mostly Private-set robustness.

Inference precision: **FP16, OpenVINO** (`scripts/main_inference_and_compile.py`, `code_base/utils/onnx_utils.py:ONNXEnsemble`). Mel extraction is done **once and reused across all ensemble models** for inference speed (90-min Kaggle CPU budget).

## Ablation findings (paper Tables 5, 7, 8)

### Stepwise contribution (Table 5)

Reading the EfficientNetV2-S column from baseline to final:

| Stage | Public | Private |
|---|---:|---:|
| Baseline (no enhancements) | 0.837 | 0.859 |
| + enhancements (label smoothing, balanced sampling) | 0.835 | 0.871 |
| + transfer learning (pretrain on 7,489-species corpus) | 0.881 | 0.889 |
| + pseudo-labels Full I2 | **0.910** | 0.909 |
| + postprocessing (TopN N=1) | **0.918** | **0.924** |
| Final ensemble (15 models) | 0.925 | 0.928 |

So per-component approximate contributions:

- Baseline → Enhancements: **~+0.012 Private** (label smoothing + sampling tweaks).
- Enhancements → Transfer learning: **+0.046 Public, +0.018 Private**.
- Transfer learning → Pseudo-labels: **+0.029 Public, +0.020 Private**.
- Pseudo-labels → PP: **+0.008 Public, +0.015 Private**.
- Single → 15-model ensemble: **+0.007 Public, +0.004 Private**.

**The two big levers are transfer learning and pseudo-labels**, contributing ~+0.07 each in stacked Public.

### Additional data ablation (Table 7)

| Config | Mean CV | Public | Private |
|---|---:|---:|---:|
| Train data only (no extras) | 0.972 | 0.839 | 0.868 |
| + 2 selected extras (XC dump 28.03.2025 + prev-comp samples) | 0.975 | 0.835 | 0.871 |
| + ALL extras (incl. CSA, iNat, more XC) | 0.976 | 0.832 | 0.864 |

**Adding more data hurt LB once past the two highlighted sources.** The paper attributes this to: (a) 2025 train data is already large enough; (b) systematic alien-speech contamination in some extras; (c) overrepresentation of recordists' biases. They did not pursue more aggressive curation.

### Augmentation ablation (Table 8) — measures Public / Private trade

| Setup | Mean CV | Public | Private |
|---|---:|---:|---:|
| Enhanced baseline (all augs) | 0.975 | **0.835** | **0.871** |
| − SpecAugment | 0.974 | 0.845 | 0.861 |
| − RandomFiltering | 0.973 | 0.854 | 0.861 |
| − MixUp | 0.964 | 0.835 | **0.835** |

**Striking**: Public score *increases* when SpecAugment / RandomFiltering are removed, but Private *decreases* every time. The augmentations help the harder, larger Private set at a small cost to Public. **MixUp is essential** — removing it kills CV (−0.011) and Private (−0.036).

The paper argues this is consistent with a smaller density of overlapping species per chunk in the Public split — augmentations help Private's harder distribution.

## Hyperparameter table (final fine-tune configs)

| Component | EfficientNetV2-S | ECA-NFNet-L0 |
|---|---|---|
| Backbone (timm) | `tf_efficientnetv2_s_in21k` | `eca_nfnet_l0` |
| Mel | nnAudio, n_mels=128, n_fft=2048, hop=512, fmin=20, normalized | same |
| Mel norm | AmplitudeToDB(top_db=80) → standardize → min-max | same |
| Head | `AttHead`, hidden=512, dropout=0.5 (p/2 + p) | same |
| Loss | BCE(reduction=mean) + sigmoid_focal(α=0.25, γ=2.0) (1:1) | same |
| Label smoothing | 0.05 (mass-preserving) | same |
| Optimizer | AdamW, lr=1e-4, ε=1e-8, β=(0.9, 0.999) | RAdam, lr=1e-3 |
| Scheduler | CosineAnnealingWarmRestarts(T_0 = 50 × steps), eta_min=1e-6 | same |
| Epochs | 50 | 50 |
| Batch | 64, drop_last, ddp_find_unused_params_true | same |
| Precision | 32-true (no AMP) | same |
| Window | 5s, random crop | same |
| Class sampling | balanced (γ = −1) | sqrt (γ = −0.5) + manual oversampling of 58 minor classes |
| Init | In-domain pretrain backbone (head re-init) | In-domain pretrain backbone (head re-init) |
| Pseudo iter used | I2 (post hoc selected) | I3 + minor oversample variant |
| Final ckpt | `last.ckpt` (NOT best-val) | same |
| Augs (per-sample) | MixUp p=0.5 (audio, 50/50) → BG noise OneOf(nocall-SS, ESC-50) p=0.5 → peak-normalize | same |
| Augs (in-model) | RandomFiltering(4 bands, U[-20, 0] dB) → mel → SpecAugment(freq L=10×3, time L=20×3, both p=0.3) | same |
| Inference | OpenVINO FP16, ensemble extracts spec once, mean across folds, then 3-experiment mean, then TopN N=1 | same |

## Their stated limitations

From Section 7.1:

1. **Validation doesn't reflect deployment.** Soundscapes are biased toward one geographic site; pseudo-labels may not generalize to other regions.
2. **No serious hyperparameter optimization.** Especially pseudo-thresholds (0.5 / 0.1 / 0.4) are educated guesses; the search space wasn't explored.

Future-work directions they list:
- More diverse soundscape evaluation across geographies.
- Model-driven curation to remove non-vocalized fragments (they tried it, hurt LB; revisiting later).
- Use the AttHead's framewise predictions at inference (currently only clipwise is used).
- Better undersampled-class handling.

## What's notably *absent* (compared to other top solutions)

- **No knowledge distillation from BirdNET / Perch / Google Bird Vocalization Classifier.** They explicitly note this is what the 2024 DS@GT paper did and that it underutilizes transfer learning.
- **No BirdNET embeddings / Perch features.** Pure timm CNN backbones, raw waveform input.
- **No CE loss.** Pure BCE+Focal multi-label.
- **No model soup / SWA in the final selected ensemble.** `n_swa_models=1` in the inference configs; they pick `last.ckpt` (which differs from `best-val.ckpt`) — interesting choice, presumably because the cosine schedule lands on a sensible final checkpoint.
- **No attention-architecture innovation.** AttHead is the same one as the 2023 paper, with the framewise blend coefficient nudged from 75/25 → 50/50 — essentially a copy of the 2018 Adavanne–Virtanen weakly-labeled SED head.
- **No multi-context / longer windows.** Pure 5s training and 5s inference.
- **No insect/amphibian-specific pipeline** (Nikita Babych 1st place 2025 had a separate one). Sydorskyi treats all 206 species uniformly.
- **No 2nd-stage temporal model** (the L2 model was tested and abandoned).

The pipeline is deliberately **boring and big**: huge pretraining corpus, lots of pseudo-iterations, simple averaging ensemble. Almost no architectural cleverness.

## Available artifacts

| Artifact | Location |
|---|---|
| Trained inference models (OpenVINO FP16) | [Kaggle: bird-clef-2025-models](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-models) |
| In-domain pretrained backbones | [Kaggle: bird-clef-2025-all-pretrained-models](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models) |
| Pseudo-label CSVs (3 iterations, both Full and OOF) | [Kaggle: bird-clef-2025-pseudo](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-pseudo) |
| Additional 2025 training data (XC dump, prev-comp scrapes) | [Kaggle: bird-clef-2025-add-data](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-add-data) |
| Pretraining metadata CSVs | [Kaggle: bird-clef-2025-pretrained-metadata](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-pretrained-metadata) |
| Inference notebook (final selected) | [Kaggle: bird-clef-2025-ensemble-v2-final-final](https://www.kaggle.com/code/vladimirsydor/bird-clef-2025-ensemble-v2-final-final) |
| Code repo | https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place |

## Key takeaways

1. **The 2025 ceiling without unlabeled data is ~0.89 Public**, just from a careful baseline + in-domain pretrain. Everything past that — for this team — came from **using the unlabeled soundscapes**, exactly as Boredom (1st place) and Salman (13th place 2025, single 0.937) also reported.

2. **Transfer learning at scale is more powerful than competition-year-only training.** The 819K-recording, 7,489-species corpus is the foundation. Without it, the baseline tops out at 0.837 Public.

3. **Soft pseudo-labels, not hard.** The paper explicitly diagnoses binarized PL (the DS@GT 2024 approach) as overconfident. Soft labels with `trim_min_prob=0.1` zero-out preserve uncertainty.

4. **Pseudo-distillation by 40% sampling + sum-MixUp.** The pseudo-target is fused into training implicitly via MixUp-summed targets, not as a separate KD loss. Cleanest semi-supervised recipe published in this competition.

5. **Augmentations help Private more than Public.** Three of four augmentations (SpecAugment, RandomFiltering, MixUp) all reduce Public score slightly while raising Private significantly. The harder distribution is Private; that's where to optimize.

6. **PP TopN N=1 at ensemble level only**: ~+0.015 LB. Per-model PP regressed. Order matters: PP → rank → average is wrong; ensemble average → PP is right.

7. **OOF correlations break down past 0.9 Public, but Optuna-on-OOF still picks the best Private ensemble.** OOF carries diversity signal even when absolute scores diverge.

8. **No KD from frozen embeddings** (BirdNET / Perch / GBVC) was used. The pipeline is pure timm-CNN-on-mel — distillation is from in-domain *student-of-student* ensembles, not external models.

## File-level pointers (for later replication)

- Baseline architecture: `code_base/models/wave_clasifier.py`, `code_base/models/blocks.py`
- Augmentations: `code_base/augmentations/transforms.py`, `code_base/augmentations/spec_augment.py`, `code_base/models/augmentations.py:RandomFiltering`
- Loss: `code_base/losses/focal_loss.py:FocalLossBCE`
- Training forward + MixUp + batch aug: `code_base/forwards/cls_forward.py:MultilabelClsForwardLongShort`
- Pseudo-label fusion: `code_base/datasets/wave_dataset.py:WaveDataset._prepare_sample_target_from_idx` (lines ~604-735)
- Pseudo creation pipeline: `notebooks/create_pseudo.ipynb`
- Pretraining configs: `train_configs/pretrain_*_2025_extended.py`
- Final fine-tune configs: `train_configs/selected_ebs.py`, `train_configs/selected_eca.py`
- Inference + ONNX/OpenVINO compile: `scripts/main_inference_and_compile.py`, `code_base/inefernce/inference_class.py`, `code_base/utils/onnx_utils.py`
- One-button reproduction: `rock_that_bird.sh`

---
experiment: "013-mixit-separation"
status: proposed
date: 2026-03-21
author: Murilo Gustineli
---

# Experiment 013: MixIT Source Separation

## Hypothesis

Source separation via Bird MixIT, applied as offline preprocessing to 10,592
unlabeled soundscapes, will produce cleaner per-channel audio that yields
higher-confidence pseudo-labels from the exp 012 teacher ensemble. Training
on these pseudo-labels will improve LB score by 0.02–0.05 over exp 012 alone.

## Motivation

BirdCLEF 2026 provides 10,592 unlabeled soundscapes — 160× more soundscape data
than the 66 labeled files used in exp 012. Direct pseudo-labeling on raw soundscapes
produces noisy multi-label predictions because:

1. **Dense overlapping vocalizations** — avg 4.2 species per 5s segment
2. **Low-SNR species are missed** — classifiers trained on focal data ignore quiet background species
3. **Co-occurring species confuse the classifier** — nocturnal frog choruses have 300+ co-occurrences

Bird MixIT (ICASSP 2022) was designed exactly for this scenario. It separates bird
mixtures into M=4 channels without requiring clean reference sources. The paper showed
+6.3% CMAP on dense dawn chorus data by classifying separated channels independently
and taking max predictions per species.

## Background

### Bird MixIT

- Pretrained on Xeno-Canto + Macaulay Library (704 species) at 22.05 kHz
- Architecture: TDCN++ with learned filterbank, mask-based separation, mixture consistency
- Domain-matched training gives +5 dB SI-SNR over generic AudioSet model
- Critical finding: max over (M separated channels + original) beats separated-only AND original-only

See `docs/vendor/bird-mixit.md` and `docs/references/bird-mixit-paper.md` for details.

### Teacher Models (from exp 012)

- EAT, ConvNeXt, ProtoCLR — 234-class models fine-tuned on labeled soundscapes
- Architecturally diverse (transformer vs CNN vs conv-transformer)
- Different pretraining domains (AudioSet vs ImageNet vs Xeno-Canto)
- Complementary error profiles make them ideal for ensemble pseudo-labeling

## Method

### Phase 1: TF1 → ONNX Conversion

Convert Google's pretrained Bird MixIT TF1 SavedModel to ONNX for portable inference:
- Use `tf2onnx` to convert the frozen TF1 graph
- Input: mono waveform at 22.05 kHz → Output: 4 separated channels
- Validate ONNX output matches TF1 output on test audio (atol < 1e-4)
- Benchmark CPU inference speed (target: <1s per 1-minute file)

### Phase 2: Offline Separation

Run MixIT on all 10,592 unlabeled soundscapes:
- Resample from 32 kHz → 22.05 kHz for MixIT input
- Save 4 separated channels per file as compressed numpy arrays
- Estimated storage: ~10,592 × 4 channels × ~160 KB ≈ 6.6 GB
- SLURM job: CPU-only, ~2–4 hours with batched processing

**Scope**: Soundscapes only, not focal recordings. Focal recordings have a single
prominent species by design — MixIT would either return the same audio in one channel
(wasted compute) or over-separate a single song across channels, where isolated
fragments might get misclassified. The paper confirmed this: prominent species scores
*decrease* after separation due to lost audio context.

#### Phase 2 Results: Labeled Soundscape Separation Quality

Validated on 66 labeled soundscape files (notebook `004-mixit-separation-quality.ipynb`).

**Channel activity (RMS > 0.01 threshold):**

| Active Channels | Files | % |
|----------------:|------:|----:|
| 0 | 4 | 6% |
| 1 | 21 | 32% |
| 2 | 16 | 24% |
| 3 | 14 | 21% |
| 4 | 11 | 17% |

- Mean RMS per channel: [0.0403, 0.0187, 0.0242, 0.0144]
- Channel 0 dominates at 2.1× higher RMS than channels 1–3
- ~38% of files (0–1 active channels) get no meaningful separation
- ~62% of files (2–4 active channels) show some degree of source isolation

**Qualitative assessment** (spectrograms + listening):
- Dense frog choruses: partial separation visible in spectrograms but quality unclear
- Multi-species bird recordings: some channel isolation of individual species
- Sparse recordings: MixIT correctly puts near-silence in unused channels

**Decision**: Proceed with full unlabeled separation. The max-pool-across-channels
approach means we never do worse than the original — even files with no meaningful
separation fall back to the original prediction. Phase 3 validation will quantitatively
measure whether separation improves pseudo-label quality.

### Phase 3: Teacher Ensemble Pseudo-Labeling

Run exp 012 ONNX models on each separated channel + original:

```
For each unlabeled soundscape (12 × 5s segments per file):
    ├── Original audio → EAT → probs_eat_orig
    │                  → ConvNeXt → probs_cnx_orig
    │                  → ProtoCLR → probs_pcr_orig
    │
    ├── Channel 1 → EAT/ConvNeXt/ProtoCLR → probs per model
    ├── Channel 2 → same
    ├── Channel 3 → same
    └── Channel 4 → same

Per model: max-pool across 5 inputs (4 channels + original)
Final: average across 3 models → soft pseudo-labels per segment
```

- Per file: 1 original + 4 separated = 5 audio streams
- Per stream: 12 segments × 3 models = 36 predictions per file
- Aggregation: max-pool across channels per model, then average across 3 models
- Apply confidence threshold (e.g., 0.3–0.7) to filter low-confidence labels
- Save as CSV: filename, start, end, soft species probabilities

Validate pseudo-label quality on 66 labeled soundscapes (ground truth available).

### Phase 4: Student Training

Fine-tune from exp 012 checkpoint on pseudo-labeled data:
- Soft BCE loss (teacher probabilities as targets, not hard 0/1)
- Mix with focal + labeled soundscape data to prevent drift
- Compare: hard labels vs soft labels, threshold sweep

### Variables

- **Independent**: Separation (yes/no), aggregation strategy, confidence threshold
- **Dependent**: LB score, pseudo-label precision/recall on labeled soundscapes
- **Controlled**: Same models, same training pipeline as exp 012

## Evaluation

### Metrics
- **Pseudo-label quality**: Precision/recall vs expert labels on 66 labeled soundscapes
- **LB score improvement** over exp 012 baseline
- **Per-class improvement**, especially 28 soundscape-only species
- **Per-taxon breakdown**: Bird vs amphibian vs insect pseudo-label quality (see below)

### Baseline
- Exp 012 LB: ProtoCLR **0.767**, ConvNeXt **0.755**, EAT **0.720**

### Success Criteria
- **Confirm if**: LB improves by ≥0.02 over exp 012
- **Reject if**: No LB improvement or degradation on focal species

## Non-Bird Species: Known Limitation

The pretrained Bird MixIT was trained exclusively on **bird audio** (704 bird species
from Xeno-Canto + Macaulay Library). BirdCLEF 2026 includes amphibians (35 species),
insects (28 species), mammals (8 species), and reptiles (1 species).

**Expected behavior by taxon**:
- **Birds** (162 species): Should work well — this is the model's training domain
- **Amphibians** (35 species, incl. 3 soundscape-only): May partially work — frog
  calls have harmonic structure somewhat similar to birds, but the separator was never
  trained on frog choruses. The densest co-occurrences in BirdCLEF 2026 are all
  frog-frog pairs (300+ co-occurrences), which is exactly where separation matters most
- **Insects** (28 species, 25 soundscape-only sonotypes): Likely poor — insect sounds
  (continuous stridulation, broadband buzzing) are spectrally very different from bird
  calls. These are the hardest category regardless of separation
- **Reptiles/Mammals**: Negligible count, impact unclear

**Mitigation**: The max-pool-across-channels approach includes the original unseparated
audio. If separation fails for non-bird species, the max-pool falls back to the original
prediction — we never do *worse* than no separation. During Phase 3 validation, we will
explicitly measure pseudo-label quality broken down by taxonomic class (bird vs amphibian
vs insect) on the 66 labeled soundscapes to quantify this limitation.

**If non-bird separation is poor**: This motivates training a domain-matched MixIT model
on BirdCLEF data (see "Future Work" below).

## Design Decisions

1. **Offline separation only** — inference-time MixIT is infeasible within 90-min Kaggle limit
2. **Soundscapes only, not focal** — focal recordings have single prominent species; separation
   would either be a no-op or actively harmful (over-separation, lost audio context)
3. **Max-pool channels, average models** — if any channel captures a species, max-pool retains it;
   model averaging reduces noise. Original audio is always included as a safety net
4. **Soft pseudo-labels** — preserve teacher confidence calibration (vs hard 0/1)
5. **Validate on labeled soundscapes first** — measure pseudo-label quality (overall and per-taxon)
   before committing to student training on 10,592 files
6. **Pretrained ONNX model first** — use Google's Bird MixIT to get results quickly, assess
   limitations on non-bird taxa, then decide whether domain-matched training is justified

## Limitations

- **Bird-only pretrained model**: Separation quality will degrade for non-bird taxa (frogs,
  insects), which are important in BirdCLEF 2026. See "Non-Bird Species" section above.
- **Sample rate mismatch**: MixIT at 22.05 kHz, BirdCLEF at 32 kHz — loses info above
  11 kHz, which may affect high-frequency bird calls and insect stridulation
- **Pseudo-label noise compounds with separation artifacts**: Over-separated fragments
  may be misclassified, introducing systematic label errors
- **Computational cost**: Separation + multi-model inference on 10,592 files
- **TF1 dependency for Phase 1** (conversion only; Phases 2+ use ONNX runtime)

## Dependencies

- Exp 012 completed (234-class teacher models)
- Bird MixIT TF1 checkpoint downloaded from GCS
- `tf2onnx` for model conversion, `onnxruntime` for inference

## Future Work: Domain-Matched MixIT Training

If the pretrained bird-only model shows poor separation for non-bird taxa, the next step
is training a **BirdCLEF-specific MixIT separator** in PyTorch:

- **Library**: [Asteroid](https://github.com/asteroid-team/asteroid) — provides ConvTasNet/TDCN++
  implementations and separation losses in PyTorch
- **MixIT loss**: Simple to implement — brute-force over 2^M=16 binary assignments for M=4
- **Training data**: 35,549 focal recordings + 10,592 soundscapes provide diverse audio for
  creating Mixtures of Mixtures. Unlike the pretrained model, this would include frog choruses,
  insect stridulation, and mammal vocalizations in the training distribution
- **Expected benefit**: The paper showed +5 dB SI-SNR from domain-matched training (Xeno-Canto
  bird model vs generic AudioSet model). A BirdCLEF-trained model could provide a similar or
  larger gain, especially for non-bird taxa that the pretrained model has never seen
- **Scope**: This would be a separate experiment (e.g., exp 014) if the Phase 3 per-taxon
  validation reveals significant limitations of the pretrained model

## Leads to

- Domain-matched MixIT training (if non-bird separation is poor)
- Student distillation for efficient Kaggle inference (future exp)

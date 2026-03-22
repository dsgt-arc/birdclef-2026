# BirdCLEF Winning Solutions Analysis (2023–2025)

Cross-cutting analysis of BirdCLEF 1st place solutions and top finishes to identify
techniques that consistently drive leaderboard performance.

**Our best**: ProtoCLR-soundscape **0.767 LB** (exp 012, BirdCLEF 2026)
**Current #1 on 2026 LB**: **0.937** — a 0.170 gap to close

## Solutions Analyzed

| Year | Place | Team/Author | Private LB | Key Insight |
|------|-------|-------------|-----------|-------------|
| 2025 | 1st | Nikita Babich | ~0.937* | Iterative pseudo-labeling (5/7 models), SED + attention head |
| 2025 | 9th | i2nfinit3y | N/A | Temporal smoothing, center-crop TTA, ONNX ensemble |
| 2024 | 1st | Team Kefir | 0.691 | Data quality filtering, CE loss >> BCE, pseudo-labeled soundscapes |
| 2023 | 1st | Volodymyr Sydorskyi | N/A | "Correct Data is All You Need" — massive XC curation, 822-class pretrain |

\* Exact score not disclosed; inferred from competition placement.

**Individual solution docs**:
- [2025 1st place](birdclef-2025/birdclef2025-1st-place-inference.md)
- [2025 9th place](birdclef-2025/bird2025-9th-place-solution.md)
- [2024 1st place](birdclef-2024/birdclef2024-1st-place-solution.md)
- [2023 1st place](birdclef-2023/birdclef2023-1st-place-solution.md)

---

## 1. Technique Comparison Matrix

| Technique | 2023 1st | 2024 1st | 2025 1st | 2025 9th | Us (exp 012) |
|-----------|----------|----------|----------|----------|--------------|
| **SED attention head** | GeMFreq + AttHead | No (pure backbone) | GeM + AttHead | AttBlockV2 | Linear/MLP head |
| **Pseudo-labeling unlabeled data** | No* | Yes (+0.025 priv) | Yes (5/7 models, 3-4 iter) | No | Planned (exp 013) |
| **Temporal smoothing** | No | No | Gaussian 5-frame kernel | 20/60/20 weighted avg | No |
| **Overlapping context** | 5s segments | 10s input for 5s pred | 20s windows, 5s step | 10s input for 5s pred | 5s windows only |
| **Ensemble** | 3 ONNX models | 6-9 models (min+mean) | 7 models (weighted avg) | 3 ONNX models | Single model |
| **Loss function** | Focal (BCE-based) | CE (>> BCE) | CE | BCE | BCE |
| **Data filtering** | Massive XC curation | Google classifier + stats | Iterative PL filtering | None | None |
| **Inference acceleration** | ONNX | OpenVINO + joblib | OpenVINO + INT8/FP16 | ONNX | ONNX |
| **Multi-taxonomy training** | 822 → 264 classes | No | 700 → 206 classes | No | 234 classes |
| **Background noise augment** | Nocall soundscapes + ESC-50 | No | Unknown | No | No |
| **Train-infer loss mismatch** | No | Yes (CE train, sigmoid infer) | Yes (CE train, sigmoid infer) | No | No |
| **Framewise max blending** | 75% clip + 25% max | No | Yes (delta-shifted) | Yes (max framewise) | No |

\* 2023 compensated with additional Xeno-Canto data rather than pseudo-labeling unlabeled soundscapes.

---

## 2. Consistent Winning Patterns

### 2.1 Sound Event Detection (SED) Architecture

**Present in**: 2023 1st, 2025 1st, 2025 9th (3/4 solutions)

All top solutions except 2024 use framewise SED rather than clip-level classification:
- **GeM pooling on frequency axis only** — collapses 128 mel bins to 1 while preserving
  all time steps. This maintains temporal resolution that global average pooling destroys.
- **Attention-based temporal aggregation** — learns which time frames contain vocalizations.
  Conv1D produces per-frame attention weights and class logits independently.
- **Inference: framewise max blending** — 75% attention-weighted clipwise + 25% framewise
  max captures both sustained calls (attention) and brief vocalizations (max).

**Why it matters for us**: Our linear/MLP heads apply global average pooling → linear → sigmoid.
This averages the entire 5s clip, diluting brief bird calls (which may occupy <0.5s of the
segment). SED preserves temporal structure and lets the model focus on the most informative
frames.

**Gap estimate**: +0.03–0.05 LB based on the difference between clip-level and SED approaches
in prior competitions.

### 2.2 Pseudo-Labeling Unlabeled Soundscapes

**Present in**: 2024 1st, 2025 1st (2/4 solutions; 2023 used XC data instead)

The most effective pseudo-labeling approach is **iterative self-training** (2025 1st):
1. Train initial model on labeled data
2. Pseudo-label unlabeled soundscapes with temperature scaling (T=0.55–0.6)
3. Retrain on labeled + pseudo-labeled data
4. Repeat 3-4 iterations

The 2024 1st place gained **+0.025 private score** from a single round of pseudo-labeling.
The 2025 1st used 5/7 models trained with iterative pseudo-labels, suggesting even larger gains.

**Key details**:
- Temperature scaling (T < 1.0) sharpens predictions, producing higher-confidence pseudo-labels
- Use an ensemble of models for pseudo-labeling (not a single model)
- Small coefficient (0.05) for pseudo-labels on train data (acts as label smoothing)
- 2024 winner: high-coefficient pseudo-labeling of train data *hurt* performance

**Relevance to exp 013**: Our MixIT-based approach adds source separation as a preprocessing
step. No winning solution has used source separation. The proven approach is direct iterative
pseudo-labeling without separation. MixIT may add value for dense multi-species soundscapes,
but the baseline comparison should be direct pseudo-labeling.

### 2.3 Overlapping Context (10–20s Input for 5s Predictions)

**Present in**: 2024 1st, 2025 1st, 2025 9th (3/4 solutions)

Instead of processing exactly 5 seconds per prediction:
- **2024 1st**: 10s input (5s target + 2.5s before + 2.5s after)
- **2025 1st**: 20s windows with 5s step, then overlap-average reconstruction
- **2025 9th**: 10s input (audio concatenated 3× for edge handling)

**Why it works**:
- Bird calls often span segment boundaries — context captures full chirp periods
- Adjacent-segment smoothing happens naturally via overlapping windows
- The model learns temporal patterns broader than a single 5s window
- 2024 1st: going from 5s to 10s input gained **+0.015 private score**

**Relevance to us**: We use exactly 5s windows with no overlap. This is one of the easiest
changes to implement and has consistent demonstrated gains.

### 2.4 Ensemble of Diverse Models

**Present in**: All 4 solutions (universal)

| Solution | Models | Strategy |
|----------|--------|----------|
| 2023 1st | 3 CNNs (ConvNeXt-S, ECA-NFNet, ConvNeXtV2-T) | Gaussian mean |
| 2024 1st | 6 EffNet-B0 + 3 RegNetY | min() within arch, mean() across |
| 2025 1st | 5 pseudo-labeled + 2 supervised (7 total) | Weighted average [0.13–0.17] |
| 2025 9th | 2 EffNetV2-B3 + 1 SEResNeXt26t | Weighted average [0.4, 0.4, 0.2] |

**Key details**:
- 2024 winner: `min()` reduction within architecture is more effective than `mean()` — it
  lowers noisy sigmoid predictions. `mean()` across different architectures combines diversity.
- Model diversity (different architectures, seeds, training data) matters more than model count
- We already have 3 diverse architectures (EAT, ConvNeXt, ProtoCLR) but can't ensemble due
  to inference time (~120 min for 2 models on ONNX)

**Inference budget**: The bottleneck is CPU time on Kaggle (90 min for 2026). OpenVINO
optimization (2-3× speedup over ONNX) could make 2-3 model ensemble feasible.

### 2.5 CE Loss with Sigmoid Inference

**Present in**: 2024 1st, 2025 1st (2/4 solutions)

The 2024 winner found CE loss "significantly better" than BCE/focal loss:
- **Training**: logits → softmax → CE loss (frames problem as multiclass)
- **Inference**: logits → sigmoid (allows multi-label output)
- **Rationale**: With 182–234 classes and usually 1–4 active per segment, the problem is
  closer to multiclass than multi-label. CE's competitive softmax produces better-calibrated
  predictions than independent BCE sigmoids.
- BCE, focal loss, and positive-weighted BCE all performed worse in ablations.

**Caveat**: This may interact with SED architecture and segment length. The 2023 winner used
focal loss successfully. But the consistency of CE in 2024 and 2025 is notable.

### 2.6 Temporal Post-Processing

**Present in**: 2025 1st, 2025 9th (2/4 solutions)

Zero-cost improvements applied after model inference:

**Gaussian smoothing** (2025 1st):
- Kernel: [0.1, 0.2, 0.4, 0.2, 0.1] applied along temporal axis
- Smooths predictions across consecutive 5s segments within each soundscape
- Reduces isolated false positives without losing sustained true detections

**Weighted neighbor averaging** (2025 9th):
- First segment: 80% current + 20% next
- Middle segments: 20% previous + 60% current + 20% next
- Last segment: 80% current + 20% previous

**Delta-shifted TTA** (2025 1st and 9th):
- Take predictions at frame offset 0, ±2 frames
- Blend: 50% center + 25% left + 25% right
- Acts as temporal TTA without multiple forward passes

### 2.7 Data Quality and Curation

**Present in**: 2023 1st, 2024 1st (2/4 solutions, both winners)

- **2023**: Collected additional Xeno-Canto data for all 264 scored species, incorporated
  secondary labels, used nocall soundscape recordings as realistic background noise
- **2024**: Signal statistics filtering (keep bottom 80th percentile of T = std + var + rms + pwr),
  Google Bird Vocalization Classifier to filter/relabel train chunks, duplicate removal

**Common theme**: The quality of training data matters more than the quantity. Filtering out
noisy/misaligned samples improves model calibration.

---

## 3. What Consistently Does NOT Work

Based on the 2024 1st place ablations and cross-solution patterns:

| Technique | Evidence |
|-----------|----------|
| **Additional Xeno-Canto data** | 2024: hurt both public and private scores |
| **Complex SED architectures** | 2024: "overly complex models do not work better than simple ones" |
| **Vision Transformers** | 2024: "ViTs work significantly worse"; 2023: pure CNN ensemble |
| **BCE / Focal loss** (for sparse labels) | 2024: "significantly worse" than CE |
| **Mixup / heavy augmentation** | 2024: noise, pixdrop, blur, audio 1D augmentations all hurt |
| **Training on 10s, inferring on 5s** | 2024: mismatch degrades performance |
| **Softmax temperature tuning** | 2024: negligible effect |
| **SWA / EMA** | 2024: negligible effect |
| **Multi-stage training** | 2024: train on X then finetune on Y didn't help |
| **Pretraining on prior years** | 2024: no improvement (but 2023 used it successfully) |

**Notes**:
- The "ViTs don't work" finding is specific to small ViTs and the 2024 competition setting.
  Our EAT (ViT with AudioSet pretraining) works well with LoRA, suggesting pretrained ViTs
  are fine — it's training ViTs from scratch that fails.
- Additional XC data hurting in 2024 but helping in 2023 suggests the benefit is
  competition-specific and depends on data quality.

---

## 4. Internal Evidence: Anthony's SED and Weak Supervision Experiments

Anthony (team member) ran four experiments on BirdCLEF 2025 data that directly test some of
the winning-solution techniques. These results constrain our roadmap.

### Exp A-013: FlyingSquid Weak Supervision (label fusion)

Fused BirdNET/Perch/acoustic-indices probes via the triplet method into per-frame soft labels.

| Metric | Focal | Soundscape |
|--------|-------|------------|
| Tokens | 939K | 564K |
| BirdNET probe accuracy | 0.788 | 0.735 |
| Perch probe accuracy | 0.779 | 0.695 |
| Indices probe accuracy | 0.687 | 0.686 |
| LF correlation (BirdNET-Perch) | 0.353 | **0.606** |
| Species with signal (acc > 0.52) | 201/206 | 142/206 |

**Verdict**: Focal labels usable (201/206 species), soundscape labels weak — high LF
correlation (~0.6) violates the conditional independence assumption the triplet method
requires. 64/206 species have no signal on soundscape.

### Exp A-014: EfficientNet Soft Labels (NEGATIVE)

Replaced hard labels with FlyingSquid soft labels for EfficientNet-B1 frozen-backbone training.

| Condition | val/cmap | vs Baseline |
|-----------|----------|-------------|
| Hard labels (exp 012 baseline) | **0.336** | — |
| Soft labels, focal only | 0.238 | −0.098 |
| Soft labels, focal + soundscape | 0.238 | −0.098 |

**Key finding**: Replacing ground truth with noisy teacher estimates is a net loss of
information. Soft labels should be used **additively** (for unlabeled data only), never as
replacements for known-good labels. The 117K soundscape chunks added zero signal.

### Exp A-015: EfficientNet SED with Soft Labels (NEGATIVE)

Added SED attention head (attention train, framewise max infer) on frozen EfficientNet-B1.

| Model | val/cmap | val/auroc |
|-------|----------|-----------|
| Clip + hard labels (baseline) | **0.336** | — |
| SED + soft labels | 0.237 | 0.812 |
| Clip + soft labels (exp 014) | 0.238 | 0.804 |

**Key finding**: The bottleneck was label quality, not architecture. SED on a frozen backbone
with soft labels provides zero gain over clip-level prediction. Anthony concludes SED needs:
1. **Trainable backbone** (LoRA or full fine-tuning) so temporal features can co-adapt
2. **Hard labels** for focal data (don't replace ground truth)
3. **Soft labels only for unlabeled data** where no ground truth exists

### Exp A-016: Contrastive SED (proposed)

Prototypical SED with frame-level cosine similarity on frozen EfficientNet-B1. Well-designed
ablation plan but inherits the frozen-backbone limitation.

### Implications for Our Roadmap

Anthony's experiments provide critical guardrails:

1. **SED architecture needs trainable backbone** — our LoRA setup already provides this,
   which is the key difference from Anthony's frozen-backbone experiments
2. **Never replace hard labels with soft labels** — use soft/pseudo labels only for the
   10,592 unlabeled soundscapes, keep hard labels for focal + labeled soundscape data
3. **FlyingSquid needs 3+ independent sources** — our 3 teacher models share training data
   so aren't truly independent; direct iterative self-training sidesteps this requirement
4. **Error analysis before architecture changes** — understand WHERE we lose 0.170 LB points
   (per-species, per-taxon, frequency bands) before deciding how to close the gap
5. **Non-bird taxa are a blind spot** — insect sonotypes and dense frog choruses occupy
   different spectral ranges than birds; our mel configs (8–16kHz max) may miss them

---

## 5. Gap Analysis: Our Approach vs Winners

### What we do well
- **Three diverse architectures** (EAT/ConvNeXt/ProtoCLR) — strong ensemble candidates
- **LoRA fine-tuning** — parameter-efficient, trainable backbone (addresses Anthony's key SED finding)
- **ONNX export pipeline** — validated end-to-end on Kaggle
- **Soundscape domain adaptation** (exp 012) — 206→234 species expansion
- **MixIT source separation** (exp 013) — unique approach no winner has tried
- **Internal negative results** — Anthony's experiments rule out frozen-backbone SED and
  soft-label replacement, saving us from repeating those dead ends

### Critical gaps (ordered by estimated impact)

| Gap | Winners Have | We Have | Est. LB Impact |
|-----|-------------|---------|----------------|
| SED attention head | GeMFreq + attention pooling | Global avg pool + linear/MLP | +0.03–0.05 |
| Pseudo-labeling (iterative) | 3-4 iterations, temperature scaling | Not yet (exp 013 planned) | +0.02–0.05 |
| Overlapping context | 10–20s windows for 5s predictions | Exactly 5s windows | +0.02–0.03 |
| Ensemble | 3–7 models | Single model (inference budget) | +0.02–0.04 |
| CE loss | CE train + sigmoid infer | BCE throughout | +0.01–0.03 |
| Temporal post-processing | Gaussian smoothing, delta TTA | None | +0.01–0.02 |
| Data quality filtering | Google classifier, stats filtering | None | +0.01–0.02 |
| Inference optimization | OpenVINO INT8/FP16 | ONNX FP32 | Enables ensemble |
| Error analysis | Per-species/taxon tuning | No error breakdown | Informs all above |

**Total estimated gap**: ~0.13–0.24 (consistent with the observed 0.170 gap)

### What might NOT close the gap
- MixIT source separation — no winner uses it, uncertain ROI, only ~62% of files get
  meaningful separation in our Phase 2 quality validation
- Soft labels replacing hard labels — Anthony showed this hurts (exp A-014)
- SED on frozen backbone — Anthony showed this gives zero gain (exp A-015)
- FlyingSquid with correlated sources — LF correlations ~0.6 on soundscape violate
  the conditional independence assumption
- Higher LoRA rank / more LoRA targets — diminishing returns at r=8
- More training data (Xeno-Canto) — 2024 winner found it hurt

---

## 6. Recommended Experiment Roadmap

### Phase 0: Error Analysis (prerequisite — do first)

**Goal**: Understand WHERE the 0.170 LB gap comes from before deciding how to close it.

Using our 3 submitted exp 012 models + 66 labeled soundscape files:

1. **Per-species AUROC breakdown**: Which species do we fail on? Correlate with training
   data volume, taxonomic class, and frequency range.
2. **Per-taxon performance**: Birds (162 sp) vs amphibians (35 sp) vs insects (28 sp, 25
   soundscape-only). Are the 28 new species the main loss driver, or is it calibration
   across all species?
3. **Frequency band analysis**: Our mel configs cap at 8–16kHz. Insect stridulation can
   extend to 20kHz+. Are we missing high-frequency taxa entirely?
4. **False positive analysis on labeled soundscapes**: Where do our models predict species
   that aren't present? Dense frog choruses? Background noise?
5. **Temporal analysis**: Do we miss brief calls that a framewise-max approach would catch?
   Compare per-frame teacher predictions vs our 5s-averaged outputs.

**Output**: Error analysis report identifying the top sources of LB loss, directly informing
which Phase A–D investments have the highest expected ROI.

### Phase A: Architecture Improvements (highest expected impact)

**SED Architecture + CE Loss + Overlapping Context**

This is the biggest architectural gap vs winners. Critically, our LoRA setup provides
the **trainable backbone** that Anthony's frozen-backbone experiments identified as
the missing ingredient for SED to work.

- Replace linear/MLP head with AttHead (GeMFreq + attention pooling + framewise max blending)
- Switch from BCE to CE loss (softmax train, sigmoid infer)
- Use 10–20s input windows with 5s step for temporal context
- Train all 3 encoders (EAT, ConvNeXt, ProtoCLR) with LoRA
- Add temporal post-processing (Gaussian smoothing, delta-shifted TTA)
- **Expected gain**: +0.05–0.10 combined

**Key difference from Anthony's exp A-015**: We use LoRA (trainable backbone), hard labels
for focal data, and BCE/CE loss with ground truth — not soft labels on a frozen backbone.

### Phase B: Pseudo-Labeling (proven winner technique)

**Exp 013** (current, Murilo): MixIT pseudo-labeling — complete as planned, measure gain.
Even if MixIT itself adds little, the pseudo-labeling infrastructure is needed.

**Iterative Self-Training** (following 2025 1st place pattern):
- Use Phase A SED models as teachers (stronger than current exp 012 models)
- Pseudo-label all 10,592 unlabeled soundscapes with temperature scaling (T=0.5–0.6)
- **Hard labels for focal + labeled soundscape data** (per Anthony's exp A-014 finding)
- **Soft pseudo-labels only for unlabeled data** (additive, not replacement)
- Iterate 2-3 times (each iteration uses the improved model as teacher)
- Compare: with vs without MixIT separation, to measure MixIT's true contribution
- **Expected gain**: +0.02–0.05

### Phase C: Ensemble + Inference Optimization

**Ensemble Inference**:
- Convert backbones to OpenVINO (INT8/FP16) for 2-3× speedup
- Fit 2-3 models within 90-min Kaggle budget
- Try 2024's `min()` within-architecture + `mean()` across-architecture strategy
- Temporal smoothing as post-processing (pure numpy, no inference cost)
- **Expected gain**: +0.02–0.04

### Phase D: Data Quality and Non-Bird Taxa (informed by Phase 0)

**Data Curation + Spectral Coverage**:
- Signal statistics filtering (keep quieter/cleaner recordings)
- BirdNET or Google classifier for train data quality filtering
- Secondary label incorporation from Xeno-Canto metadata
- Background noise augmentation using nocall soundscape segments
- **Frequency range expansion**: If Phase 0 reveals insect/high-freq gaps, explore
  mel configs with higher fmax or multi-resolution spectrograms
- **Expected gain**: +0.01–0.02

### Cumulative Target

| Phase | Technique | Cumulative LB (est.) |
|-------|-----------|---------------------|
| Current | Exp 012 (soundscape finetune) | **0.767** |
| 0 | Error analysis (informs priorities) | 0.767 (no score change) |
| A | SED + CE + context + smoothing | **0.82–0.87** |
| B | Iterative pseudo-labeling | **0.85–0.90** |
| C | Ensemble (2-3 models) | **0.87–0.93** |
| D | Data curation + spectral coverage | **0.88–0.94** |

---

## 7. Key Takeaways

1. **Error analysis first**: Understand per-species, per-taxon, and per-frequency-band
   failure modes before investing in architecture changes. The 0.170 gap may be concentrated
   in specific areas (e.g., 28 soundscape-only species, non-bird taxa, dense choruses).

2. **SED needs trainable backbone**: Anthony's frozen-backbone SED experiments (exp A-015)
   showed zero gain. Our LoRA setup provides trainable features — this is the critical
   difference that should make SED work for us.

3. **Never replace hard labels with soft labels**: Anthony's exp A-014 showed a −0.098 cmap
   drop from replacing ground truth with teacher soft labels. Pseudo/soft labels should only
   be used additively for unlabeled data.

4. **Iterative self-training is the proven pseudo-labeling path**: Winners use direct
   iterative pseudo-labeling (teacher → label → retrain → repeat), not source separation
   or weak supervision frameworks. FlyingSquid requires 3+ independent sources that are
   hard to obtain on soundscape data (LF correlations ~0.6).

5. **Simple models beat complex ones**: EfficientNet-B0 won 2024. ConvNeXt-Small/Tiny won
   2023. No winner uses models larger than ~90M params. Our 3 encoders are the right size.

6. **Inference budget is a strategic constraint**: OpenVINO can unlock ensemble inference
   within 90 minutes. This should be prioritized alongside architecture changes.

7. **Non-bird taxa are a blind spot**: Insect sonotypes (28 species, 25 soundscape-only)
   and dense frog choruses occupy different spectral ranges than birds. Our mel configs
   (8–16kHz max) and bird-pretrained models may fundamentally miss these. Phase 0 error
   analysis should quantify this gap.

8. **Train-infer asymmetry works**: CE loss + sigmoid inference, framewise max + attention
   blending, overlapping windows with reconstruction — these train/infer mismatches are
   deliberate and effective across multiple winning solutions.

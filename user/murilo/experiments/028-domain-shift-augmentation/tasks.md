# Experiment 028: Domain Shift Augmentation — Tasks

## Goal

Close the 0.906→0.93+ gap by addressing domain shift through environmental noise
augmentation, secondary label supervision, soundscape oversampling, SpecAugment,
balanced sampling, and TopN post-processing. Combined recipe tested as a single
change (21 individual changes have all failed — hypothesis: they're complementary).

**Branch**: `mg/028-domain-shift-augmentation`
**Baseline**: v5j single fold = 0.906 LB (exp 019)

---

## Phase A: Diagnostic — Where are we failing?

Existing data: v5j inference already run on labeled soundscapes + focal audio.
- Logits: `/shared/inference/birdclef-2026/hgnet-b0-sed-v5j/train_soundscapes/logits.parquet`
- 4-fold: `/shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes/`
- Script: `025-iter-pseudo-labeling/diagnose_regression.py` (adapt for single-model analysis)

- [ ] A.1 Write diagnostic script (adapt from diagnose_regression.py for single-model)
- [ ] A.2 Compute per-species ROC-AUC on labeled soundscapes (75 species with GT)
- [ ] A.3 Group per-species AUC by taxonomy (Aves / Amphibia / Insecta)
- [ ] A.4 Check the "Invisible 28" (soundscape-only species) — are they predicted at all?
- [ ] A.5 Analyze confidence distributions: focal-trained vs soundscape-only species
- [ ] A.6 Document findings in `artifacts/diagnostic_analysis.md`

## Phase B: TopN=1 Post-Processing (inference-only)

- [ ] B.1 Implement TopN=1 in submission kernel:
      `prob[file, segment, class] *= max(prob[file, :, class])`
- [ ] B.2 Apply to existing v5j ONNX model (no retraining)
- [ ] B.3 Upload modified kernel + submit to LB
- [ ] B.4 Record result (expected: +0.01-0.015 vs 0.906 baseline)

## Phase C: Combined Training Recipe

### C.1: Environmental noise mixing

- [ ] C.1.1 Extract low-activity segments from labeled soundscapes as noise bank
- [ ] C.1.2 Implement noise mixing augmentation (Gaussian + pink + soundscape noise)
- [ ] C.1.3 Apply at 50% probability, SNR range -6 to +12 dB
- [ ] C.1.4 Additive only — no label change (this is noise, not signal)

### C.2: Secondary label supervision

- [ ] C.2.1 Verify secondary_labels parsing in current pipeline (4,372 focal recordings)
- [ ] C.2.2 Ensure secondary species are added to multi-hot labels with weight 1.0
- [ ] C.2.3 Verify multi-species labels work correctly with BCE loss

### C.3: Soundscape oversampling

- [ ] C.3.1 Implement oversampling factor for soundscape dataset (5-10x)
- [ ] C.3.2 Verify batch composition after oversampling

### C.4: SpecAugment

- [ ] C.4.1 Implement time masking (1-2 masks, max 40 frames)
- [ ] C.4.2 Implement frequency masking (1-2 masks, max 20 mel bins)
- [ ] C.4.3 Apply after mel computation, before spectrogram MixUp

### C.5: Sqrt-balanced sampling

- [ ] C.5.1 Compute per-species sample weights: w_i = (count_i / total)^(-0.5)
- [ ] C.5.2 Implement WeightedRandomSampler in training dataloader
- [ ] C.5.3 Verify rare species appear more frequently in training batches

### C.6: Train and evaluate

- [ ] C.6.1 Create config `configs/v12_domain_shift.yaml` (all changes combined)
- [ ] C.6.2 Fast-dev-run sanity check
- [ ] C.6.3 Full training (single fold, 90/10, 40 epochs)
- [ ] C.6.4 ONNX export
- [ ] C.6.5 Submit to LB WITHOUT TopN post-processing
- [ ] C.6.6 Submit to LB WITH TopN post-processing
- [ ] C.6.7 Record results in results.md

## Phase D: Iterate (if Phase C improves)

- [ ] D.1 4-fold training with best recipe
- [ ] D.2 4-fold ensemble submission
- [ ] D.3 Ensemble + Perch submission
- [ ] D.4 Ablation: remove individual changes to identify contributions

## Success Criteria

- [ ] Phase B (TopN only): > 0.906 LB
- [ ] Phase C (combined recipe): > 0.906 LB
- [ ] Phase C + TopN: > 0.910 LB (stretch goal)

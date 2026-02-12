# Tasks

## 1. Setup
- [x] 1.1 Add nnAudio to project dependencies (`user/anthony/pyproject.toml`)
- [ ] 1.2 Verify nnAudio mel output matches torchaudio kaldi fbank (within tolerance on a test sample)
- [x] 1.3 Scaffold experiment from exp 007 (copy configs, adapt)
- [ ] 1.4 Verify exp 007 clean re-run results (jobs 3897690/3897692) as updated baseline

## 2. Implementation

### Data Pipeline (nnAudio, sequential streaming)
- [x] 2.1 Implement `BirdSetStreamingDataModule`: loads raw audio from HF BirdSet, splits recordings into sequential non-overlapping 10s chunks
- [x] 2.2 Implement per-token label construction: overlay `detected_events` on each chunk, label=1 if event overlaps frame, else 0 → `(B, T, C)`
- [x] 2.3 Handle clip-level labels: recording-level species labels for chunks with any event, all-zeros for silence-only chunks
- [x] 2.4 Handle val data: 5s clips padded to 10s, clip-level labels only, no token labels
- [x] 2.5 Add nnAudio `MelSpectrogram` as nn.Module in model forward (GPU-accelerated)
- [x] 2.6 Handle last chunk per recording: pad to 10s with padding mask

### Attention-Pooled SED Head
- [x] 2.7 Implement `SSAMBASEDClassifier`: backbone + per-token Linear(D, C) SED head
- [x] 2.8 Implement attention pooling: softmax(sed_logits, dim=T) * sigmoid(sed_logits) → clip pred
- [x] 2.9 Handle padding mask in attention pooling (mask padded tokens with -inf before softmax)
- [x] 2.10 Load pre-trained Mamba weights (same as exp 007)

### Lightning Module
- [x] 2.10 Implement focal BCE loss with configurable γ (γ=0 reduces to plain BCE)
- [x] 2.11 Implement dual loss: L_total = L_clip (focal BCE, γ_clip) + α·L_sed (focal BCE, γ_sed, masked)
- [x] 2.12 SED loss masked to exclude padded tokens; val samples use only clip loss
- [x] 2.13 Val step: compute clip-level cmAP/AUROC/F1 from attention-pooled predictions
- [x] 2.14 Log L_clip, L_sed, L_total separately
- [x] 2.15 Freeze/unfreeze schedule (same as exp 007: freeze backbone 5 epochs)

### Config + Sbatch
- [x] 2.16 Create config-sed-128x1.yaml (128x1 patches, nnAudio params, α, γ_clip, γ_sed)
- [x] 2.17 Create sbatch/train-sed.sbatch with --run-name sed-128x1

## 3. Execution
- [x] 3.1 Fast-dev-run: verify shapes, both losses finite, no OOM, token labels align with events
- [x] 3.2 Overfit test: 4 batches, verify both losses decrease
- [ ] 3.3 Benchmark: measure throughput (samples/sec) vs exp 007 precomputed cache
- [ ] 3.4 Full training run (~14h)

## 4. Analysis
- [ ] 4.1 Compare val/cmAP and AUROC against exp 007 condition B (128x1, 0.172 cmAP)
- [ ] 4.2 Compare train-val gap: is overfitting reduced?
- [ ] 4.3 Inspect attention weights: do they concentrate on vocalization regions?
- [ ] 4.4 Assess nnAudio throughput impact
- [ ] 4.5 Document findings in results.md

---

## Diagnostic Results

### Tests (48/48 passing, job 3901455)

All 48 tests pass on GPU node:
- 13 data pipeline tests (parse events, token labels, collation, BirdSet num classes)
- 12 GPU forward tests (SED shapes, backward, bf16 mixed, padding mask, pretrained loading, freeze/unfreeze, env checks)
- 4 lightning module tests (training/val step, optimizer config, alpha=0 no-SED mode)
- 12 focal loss tests (gamma=0 ≡ BCE, focal < BCE, gradients, probs mode, masking, autocast safety)
- 7 CPU model tests (forward shapes, clip pred range, SED logits raw, padding mask, spec augment, backward, mel compute)

### Fast-dev-run (job 3901460, PASSED)

1 train batch + 1 val batch completed successfully:
- **train/loss = 0.751** (clip + SED, both finite)
- **val/loss = 0.292**, val/cmap = 0.029, val/auroc = 0.056
- Pre-trained weights: 96.9% loaded (436/438 params, patch_embed + pos_embed re-initialized)
- Backbone frozen: 434 params frozen, 221,205 / 6,995,349 trainable (3.2%)
- Data: 34,642 train chunks from 5,460 recordings, 12,000 val chunks
- Wall time: ~51s total (incl. dataset indexing)

### Overfit test (job 3901469, PASSED, 200 epochs on 4 batches)

| Metric | Epoch 0 | Epoch 199 | Interpretation |
|--------|---------|-----------|----------------|
| train/loss_epoch | 0.635 | 0.007 | Loss decreases to near-zero — model memorizes 4 batches ✓ |
| train/cmap | 0.054 | 0.190 | ≈ 4/21 classes present in 4-batch subset; macro AP averages all 21 classes |
| train/f1_macro | — | 0.190 | Same as cmAP — only ~4 species have positives in 64 samples |
| val/loss | — | 0.041 | Val on same 4 batches (overfit mode), low as expected |
| Throughput | ~1.6 batch/s train, ~7.1 batch/s val | | ~3.3s per epoch (4 train + 4 val steps) |
| Wall time | | 13m 19s total | 200 epochs × 4 batches |

The cmAP plateau at 0.190 is expected: with only 64 samples from 21 species, approximately 4 species have positive examples. The model perfectly predicts those 4 (AP≈1.0 each) while the other 17 absent species contribute 0 to the macro average → 4/21 ≈ 0.190.

### Bugs found and fixed during diagnostics

1. **F.binary_cross_entropy unsafe under bf16 autocast** — PyTorch blocks this function during autocast. Fixed by replacing with manual BCE: `-(target * log(p) + (1-target) * log(1-p))`. Also caught CUDA strict assertion on target values.
2. **ebird_code_multilabel parsed incorrectly** — BirdSet stores class indices as a list `[2, 4, 6]`, not a multi-hot encoding. Was using `enumerate()` which wrote indices as values instead of positions.
3. **GPU monitor attribute typo** — `total_mem` → `total_memory` in PyTorch CUDA device properties.
4. **Missing log transform** — nnAudio returns linear-power mel; normalization constants expect log scale. Added `torch.log(mel.clamp(min=1e-10))`.
5. **NaN in attention pooling** — All-padded tokens → softmax of all -inf → NaN. Fixed with `nan_to_num(0.0)`.
6. **Per-chunk full-file reload** — Each chunk loaded entire recording via soundfile. Fixed with partial reads: `sf.read(start, frames)`.

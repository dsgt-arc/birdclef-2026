# Tasks

## 1. Setup
- [ ] 1.1 Add nnAudio to project dependencies (`user/anthony/pyproject.toml`)
- [ ] 1.2 Verify nnAudio mel output matches torchaudio kaldi fbank (within tolerance on a test sample)
- [ ] 1.3 Scaffold experiment from exp 007 (copy configs, adapt)
- [ ] 1.4 Verify exp 007 clean re-run results (jobs 3897690/3897692) as updated baseline

## 2. Implementation

### Data Pipeline (nnAudio, sequential streaming)
- [ ] 2.1 Implement `BirdSetStreamingDataModule`: loads raw audio from HF BirdSet, splits recordings into sequential non-overlapping 10s chunks
- [ ] 2.2 Implement per-token label construction: overlay `detected_events` on each chunk, label=1 if event overlaps frame, else 0 → `(B, T, C)`
- [ ] 2.3 Handle clip-level labels: recording-level species labels for chunks with any event, all-zeros for silence-only chunks
- [ ] 2.4 Handle val data: 5s clips padded to 10s, clip-level labels only, no token labels
- [ ] 2.5 Add nnAudio `MelSpectrogram` as nn.Module in model forward (GPU-accelerated)
- [ ] 2.6 Handle last chunk per recording: pad to 10s with padding mask

### Attention-Pooled SED Head
- [ ] 2.7 Implement `SSAMBASEDClassifier`: backbone + per-token Linear(D, C) SED head
- [ ] 2.8 Implement attention pooling: softmax(sed_logits, dim=T) * sigmoid(sed_logits) → clip pred
- [ ] 2.9 Handle padding mask in attention pooling (mask padded tokens with -inf before softmax)
- [ ] 2.10 Load pre-trained Mamba weights (same as exp 007)

### Lightning Module
- [ ] 2.10 Implement focal BCE loss with configurable γ (γ=0 reduces to plain BCE)
- [ ] 2.11 Implement dual loss: L_total = L_clip (focal BCE, γ_clip) + α·L_sed (focal BCE, γ_sed, masked)
- [ ] 2.12 SED loss masked to exclude padded tokens; val samples use only clip loss
- [ ] 2.13 Val step: compute clip-level cmAP/AUROC/F1 from attention-pooled predictions
- [ ] 2.14 Log L_clip, L_sed, L_total separately
- [ ] 2.15 Freeze/unfreeze schedule (same as exp 007: freeze backbone 5 epochs)

### Config + Sbatch
- [ ] 2.16 Create config-sed-128x1.yaml (128x1 patches, nnAudio params, α, γ_clip, γ_sed)
- [ ] 2.17 Create sbatch/train-sed.sbatch with --run-name sed-128x1

## 3. Execution
- [ ] 3.1 Fast-dev-run: verify shapes, both losses finite, no OOM, token labels align with events
- [ ] 3.2 Overfit test: 4 batches, verify both losses decrease
- [ ] 3.3 Benchmark: measure throughput (samples/sec) vs exp 007 precomputed cache
- [ ] 3.4 Full training run (~14h)

## 4. Analysis
- [ ] 4.1 Compare val/cmAP and AUROC against exp 007 condition B (128x1, 0.172 cmAP)
- [ ] 4.2 Compare train-val gap: is overfitting reduced?
- [ ] 4.3 Inspect attention weights: do they concentrate on vocalization regions?
- [ ] 4.4 Assess nnAudio throughput impact
- [ ] 4.5 Document findings in results.md

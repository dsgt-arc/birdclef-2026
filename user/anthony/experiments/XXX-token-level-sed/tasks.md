# Tasks

## 1. Setup
- [ ] 1.1 Create experiment directory, fork code from exp 006
- [ ] 1.2 Explore BirdSet HSN `detected_events` field: distribution of event counts, durations, cluster IDs
- [ ] 1.3 Verify exp 006 final baseline numbers (wait for running job to finish or early-stop)

## 2. Implementation

### Data Pipeline
- [ ] 2.1 Extend `StreamingBirdSetDataset` to read `detected_events` and `event_cluster` from HF dataset
- [ ] 2.2 Implement `_events_to_token_labels()`: convert `[start_s, end_s]` pairs to `(T, num_classes)` binary tensor, filtering by dominant cluster
- [ ] 2.3 Update collate function to return `(mel_specs, token_labels, padding_mask)` where `token_labels` is `(B, T, C)` instead of `(B, C)`
- [ ] 2.4 Update precompute_mels.py to include per-token labels in cached .pt files (or compute on-the-fly in collate)
- [ ] 2.5 Handle val data: val clips don't have `detected_events`, so val collate should still produce clip-level labels `(B, C)` for evaluation

### Model
- [ ] 2.6 Create `SSAMBATokenClassifier`: same backbone, `Linear(embed_dim, num_classes)` applied per-token, no pooling
- [ ] 2.7 Support dual-mode forward: token-level output `(B, T, C)` for training, clip-level output `(B, C)` for val (via aggregation)

### Lightning Module
- [ ] 2.8 Per-token BCE loss with padding mask
- [ ] 2.9 Consider class weighting or focal loss for token-level label imbalance (vocalizations are sparse)
- [ ] 2.10 Val step: aggregate token predictions to clip-level, compute cmAP/AUROC/F1 for comparison with exp 006
- [ ] 2.11 Add token-level metrics (precision/recall) as sanity checks during training

### Config
- [ ] 2.12 Create config.yaml for exp 007 (same backbone/data params, add SED-specific settings: aggregation method, class weighting)

## 3. Execution
- [ ] 3.1 Fast-dev-run: verify shapes, loss is finite, no OOM, token labels look correct
- [ ] 3.2 Inspect a few training samples: visualize mel spectrogram + token labels overlay to confirm event alignment
- [ ] 3.3 Full training run (~12h)
- [ ] 3.4 Collect clip-level val metrics for comparison with exp 006

## 4. Analysis
- [ ] 4.1 Compare val/cmAP, AUROC, F1 against exp 006 baseline
- [ ] 4.2 Analyze per-class performance: which species benefit most from token-level supervision?
- [ ] 4.3 Inspect token-level predictions: do they temporally align with vocalizations?
- [ ] 4.4 Assess feasibility of running bambird on BirdCLEF competition data based on results
- [ ] 4.5 Document findings in results.md

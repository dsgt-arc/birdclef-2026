# Tasks

## 1. Setup
- [ ] 1.1 Wait for exp 007 SED baseline results
- [ ] 1.2 Locate Murilo's ConvNeXT checkpoint for HSN
- [ ] 1.3 Verify bambird / scikit-maad can be installed in the project venv
- [ ] 1.4 Fork code from exp 007

## 2. Pre-Computation
- [ ] 2.1 Run ConvNeXT inference over all HSN train recordings in 5s sliding windows, save per-window predictions as parquet
- [ ] 2.2 Extract bambird events for HSN train recordings (or use BirdSet's `detected_events` directly)
- [ ] 2.3 Build intersection: combine bambird temporal events + ConvNeXT species predictions into per-token soft labels
- [ ] 2.4 Cache per-token pseudo-labels alongside mel spectrograms in .pt files

## 3. Implementation

### Multi-Head Architecture
- [ ] 3.1 Add SSL head: masked token prediction (Linear(D, D) on masked positions)
- [ ] 3.2 Add clip head: masked mean pool → Linear(D, C) for recording-level weak supervision
- [ ] 3.3 Implement token masking strategy (random ~40%, avoid masking padded tokens)
- [ ] 3.4 Handle forward pass: shared backbone, three separate heads, masking applied at input level

### Multi-Task Loss
- [ ] 3.5 Implement combined loss: L = α·L_sed + β·L_ssl + γ·L_clip
- [ ] 3.6 Add loss weight hyperparameters to config
- [ ] 3.7 Log individual loss components for debugging

### Data Pipeline
- [ ] 3.8 Update dataset to load per-token pseudo-labels (soft labels from ConvNeXT intersection)
- [ ] 3.9 Update collate to return (mel_specs, token_labels, clip_labels, padding_mask)
- [ ] 3.10 Handle val data: clip-level evaluation only (no token labels for soundscape val)

## 4. Execution
- [ ] 4.1 Fast-dev-run: verify all three losses finite, shapes correct, no OOM
- [ ] 4.2 Inspect pseudo-labels: visualize mel + token labels + ConvNeXT predictions overlay
- [ ] 4.3 Ablation runs (prioritize key comparisons):
  - [ ] 4.3a SED only (exp 007 baseline)
  - [ ] 4.3b SED + SSL
  - [ ] 4.3c SED + clip
  - [ ] 4.3d SED + SSL + clip (full multi-signal)
- [ ] 4.4 Full training with best configuration

## 5. Analysis
- [ ] 5.1 Compare val/cmAP across ablations and against exp 006/007 baselines
- [ ] 5.2 Measure convergence speed: epochs to reach 0.10 cmAP for each configuration
- [ ] 5.3 Analyze whether multi-signal exceeds ConvNeXT teacher (0.48) — student-surpasses-teacher test
- [ ] 5.4 Assess which signal contributes most (ablation analysis)
- [ ] 5.5 Evaluate feasibility of scaling to full BirdCLEF competition dataset
- [ ] 5.6 Document findings in results.md

# Tasks: SSAMBA SSL Pretraining

## 1. Setup

- [ ] 1.1 Verify vendor/ssamba CUDA training path works on RTX 6000 (mamba-ssm + causal-conv1d build)
- [ ] 1.2 Confirm VRAM budget: SSAMBA-base (99M) + 500 tokens/batch on 24GB; add gradient checkpointing if needed
- [ ] 1.3 Build audio index parquet: scan 2025+2026 focal + soundscape files, 10s windows 50% overlap
- [ ] 1.4 Hold out 5% of soundscape files for SSL validation (stratified by site)
- [ ] 1.5 Verify mel config: 32kHz, n_fft=4096, n_mels=128, hop_length=320, fmin=50, fmax=16000

## 2. Implementation

- [ ] 2.1 Adapt SSAMBA patch embedding for 128×2 rectangular patches (from exp 005 work)
- [ ] 2.2 Implement SSL training: masking 40% of patches, MSE reconstruction head
- [ ] 2.3 Wire up Lightning module: AdamW optimizer, cosine LR schedule, warmup 5 epochs
- [ ] 2.4 Add TensorBoard logging: train/val MSE loss, LR, GPU utilization
- [ ] 2.5 Write sbatch script: RTX_6000, fp16, torch.compile, $TMPDIR venv

## 3. Execution

- [ ] 3.1 Run SSAMBA-small (26M) SSL training — monitor val loss convergence
- [ ] 3.2 Run SSAMBA-base (99M) SSL training — may need gradient checkpointing
- [ ] 3.3 Save best checkpoint per model (lowest val reconstruction loss)

## 4. Evaluation

- [ ] 4.1 Linear probe: freeze backbone, train Linear(embed_dim, 206) on BirdCLEF 2026 focal
- [ ] 4.2 Report cmAP and Kaggle-style AUROC; compare against exp 007 baseline (0.173 cmAP)
- [ ] 4.3 (If probe > 0.20) Run LoRA finetune using exp 102 training infrastructure
- [ ] 4.4 (If finetuned) Submit to Kaggle, record LB score

## 5. Analysis

- [ ] 5.1 Compare small vs base: does more capacity help with in-domain SSL?
- [ ] 5.2 Compare to AudioSet baseline: did domain SSL close or widen the gap?
- [ ] 5.3 Document findings in results.md, update docs/STATUS.md
- [ ] 5.4 Update docs/concepts/ssamba-audio-mamba.md with new experimental entry

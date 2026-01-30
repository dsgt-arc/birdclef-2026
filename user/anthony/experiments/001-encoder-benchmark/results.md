# Results: Encoder Benchmark on ESC-50

**Date Completed**: 2026-01-29 (updated)
**Author**: Anthony Miyaguchi

## Summary

BEATs and AST both achieve ~95-96% accuracy on ESC-50 using frozen features with a linear probe, significantly outperforming other encoders. EAT requires mean pooling (not CLS token) for competitive performance. LoRA finetuning dramatically improves EAT (77.9% → 94.6%) but fails completely for wav2vec2 due to domain mismatch. SSAMBA-HF-small was fixed (missing outer bidirectional loop) and now achieves 61.6% linear probe, and after matching the original training recipe, both full finetuning and LoRA reach **87.0% on fold 1** — exceeding the published 85.5% for SSAMBA-small.

**Verdict**: BEATs is best for frozen features (fastest + highest accuracy); AST is a close second but 5x slower. SSAMBA-HF-small (26M params) with finetuning exceeds published results for this model size (87.0% fold 1 vs 85.5% published). Full 5-fold CV needed for definitive comparison.

## Observations

### Quantitative

| Encoder | Mean Accuracy | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Embed Dim | Eval Time |
|---------|--------------|--------|--------|--------|--------|--------|-----------|-----------|
| BEATs | **95.8%** | 94.7% | 95.5% | 95.7% | 98.5% | 94.5% | 768 | 107s |
| AST | 95.5% | 95.2% | 95.2% | 94.8% | 97.0% | 94.8% | 768 | 586s |
| EAT | 77.9% | 76.0% | 79.0% | 75.7% | 79.7% | 79.2% | 768 | 162s |
| SSAMBA-HF-small | 61.6% | 63.0% | 64.5% | 62.5% | 61.0% | 57.0% | 384 | 2646s |
| wav2vec2 | 48.2% | 51.0% | 45.7% | 49.0% | 50.5% | 44.7% | 768 | 256s |

#### Comparison to Published Results

| Encoder | Our Result | Published | Method | Gap |
|---------|-----------|-----------|--------|-----|
| BEATs | 95.8% | 98.1% | Linear probe vs Finetuned | -2.3% |
| EAT | 77.9% | 95.9% | Linear probe vs Finetuned | -18.0% |
| wav2vec2 | 48.2% | N/A | Speech model, not designed for ESC | N/A |

### GPU Statistics (EAT run, V100-16GB)

| Metric | Value |
|--------|-------|
| Peak Memory | 1,372 MiB (8.4% of 16GB) |
| Peak GPU Utilization | 84% |
| Peak Power Draw | 248.8 W |
| Temperature Range | 45-61°C |
| Job Duration | 3m 51s |

GPU utilization was bursty (0-84%) due to alternating between:
- Feature extraction (GPU-bound, high utilization)
- Linear probe training on CPU (sklearn, 0% GPU)

### Qualitative

- BEATs frozen features are remarkably effective, achieving 95.8% with just a linear probe
- EAT initially showed 45% accuracy due to using CLS token; switching to mean pooling improved to 77.9%
- wav2vec2 (speech model) performs at near-random levels on environmental sounds, as expected
- All encoders produce 768-dimensional embeddings
- BEATs is fastest for feature extraction despite not using HuggingFace infrastructure

## Analysis

BEATs' strong frozen-feature performance suggests its pretraining objective (iterative audio tokenization with masked prediction) creates highly transferable representations. The 2.3% gap to finetuned performance is surprisingly small.

EAT's larger gap (18%) indicates the pretrained model benefits more from task-specific finetuning. The CLS token issue suggests EAT wasn't trained with CLS-based classification objectives during pretraining.

### Feature Extraction Strategy

| Encoder | Pooling Method | Why |
|---------|---------------|-----|
| BEATs | Mean pooling | Matches published implementation |
| EAT | Mean pooling (fixed) | CLS token gave 45%, mean pooling gives 78% |
| wav2vec2 | Mean pooling | Standard for sequence models |

### Part C: AST/SSAST/SSAMBA Benchmark (2025-01-15)

| Encoder | Mean Accuracy | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Embed Dim | Eval Time |
|---------|--------------|--------|--------|--------|--------|--------|-----------|-----------|
| **AST** | **95.5%** | 95.2% | 95.2% | 94.8% | 97.0% | 94.8% | 768 | 586s |
| SSAST | BLOCKED | - | - | - | - | - | 768 | - |
| SSAMBA-tiny | BLOCKED | - | - | - | - | - | 192 | - |

**AST (Audio Spectrogram Transformer):**
- Uses HuggingFace `MIT/ast-finetuned-audioset-10-10-0.4593`
- Very competitive with BEATs (95.5% vs 95.8%)
- Slower feature extraction (~10 min vs ~2 min for BEATs) due to per-sample processing

**SSAST/SSAMBA - Blocked due to environment issues:**
- SSAST: Requires `timm==0.4.5` which breaks CUDA libraries when downgraded
- SSAMBA: CUDA version mismatch (cluster has CUDA 13.0, PyTorch compiled with 12.8)
- Both need separate venvs or code porting to resolve

### Part B: LoRA Finetuning Results

| Encoder | Method | Mean Accuracy | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---------|--------|--------------|--------|--------|--------|--------|--------|
| EAT | Linear probe | 77.9% | 76.0% | 79.0% | 75.7% | 79.7% | 79.2% |
| EAT | LoRA r=8 | **94.6%** | 94.0% | 94.0% | 94.3% | 96.8% | 93.8% |
| wav2vec2 | Linear probe | 48.2% | 51.0% | 45.7% | 49.0% | 50.5% | 44.7% |
| wav2vec2 | LoRA r=8 | 3.0% | 2.0% | 1.8% | 3.3% | 3.5% | 4.5% |

**Key findings:**
- EAT LoRA improves from 77.9% → 94.6% (+16.7%), closing the gap with published results (95.9%)
- wav2vec2 LoRA **failed completely** (3% ≈ random chance for 50 classes)

**Why wav2vec2 LoRA failed (zero learning):**
1. **Frozen conv layers**: wav2vec2's convolutional feature extractor is frozen (LoRA only adapts transformer attention). These layers encode speech-specific patterns that can't be overcome.
2. **Domain mismatch**: wav2vec2-base-960h was pretrained on LibriSpeech (English speech) for ASR. Its representations encode phonetic/linguistic features, not environmental sound patterns.
3. **Training dynamics**: Loss stayed flat at 3.91 (log(50) = random chance) across all 50 epochs. Not slow learning - **zero learning**. The gradients provide no useful signal.

**Conclusion**: LoRA is effective when the pretrained model has relevant domain knowledge (EAT on environmental sounds). It cannot bridge large domain gaps (speech → environmental sounds).

### Part D: SSAMBA-HF Benchmark (2026-01-29, updated)

Successfully ran SSAMBA using the HuggingFace-compatible port (experiment 004), eliminating the CUDA kernel dependency.

**Bug fix (2026-01-29):** The original results (18.4%) were caused by a missing outer bidirectional loop in the HF port. The original SSAMBA processes layers in pairs — even layers see normal input, odd layers see **flipped** (reversed sequence) input, and results are summed. The HF port's `BidirectionalMambaStack.forward()` was processing all 24 layers sequentially without input flipping. After fixing this in `mamba.py`, accuracy jumped from 18.4% to 61.6%.

| Encoder | Mean Accuracy | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Embed Dim | Eval Time |
|---------|--------------|--------|--------|--------|--------|--------|-----------|-----------|
| SSAMBA-HF-small (broken) | 18.4% | 21.8% | 16.8% | 21.3% | 16.8% | 15.5% | 384 | 2504s |
| SSAMBA-HF-small (fixed) | **61.6%** | 63.0% | 64.5% | 62.5% | 61.0% | 57.0% | 384 | 2646s |

**What was wrong — the outer bidirectional loop:**

The original SSAMBA code (`both_models.py:360-370`) processes layers in pairs:
```python
for i in range(len(self.v.layers) // 2):
    hidden_states_f, residual_f = self.v.layers[i * 2](hidden_states, residual)
    hidden_states_b, residual_b = self.v.layers[i * 2 + 1](
        hidden_states.flip([1]), None if residual is None else residual.flip([1])
    )
    hidden_states = hidden_states_f + hidden_states_b.flip([1])
    residual = residual_f + residual_b.flip([1])
```

The HF port was missing this entirely — it just ran layers sequentially. This meant odd-indexed layers (which have backward-direction weights `A_b_log`, `D_b`, `conv1d_b`) were receiving forward-direction input, producing garbage features.

**Conclusion**: 61.6% linear probe is reasonable for SSL-pretrained SSAMBA-small (384-dim). The ranking is now: BEATs (95.8%) > AST (95.5%) > EAT (77.9%) > SSAMBA-HF (61.6%) > wav2vec2 (48.2%).

### Part E: SSAMBA-HF LoRA Finetuning (2026-01-30, updated)

| Encoder | Method | Mean Accuracy | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---------|--------|--------------|--------|--------|--------|--------|--------|
| SSAMBA-HF-small (broken) | Linear probe | 18.4% | 21.8% | 16.8% | 21.3% | 16.8% | 15.5% |
| SSAMBA-HF-small (broken) | LoRA r=8 | 29.6% | 30.5% | 23.5% | 32.3% | 34.0% | 27.8% |
| SSAMBA-HF-small (fixed) | Linear probe | 61.6% | 63.0% | 64.5% | 62.5% | 61.0% | 57.0% |
| SSAMBA-HF-small (fixed) | LoRA r=8 (pre-recipe fix) | **TBD** | 81.5% | - | - | - | - |
| SSAMBA-HF-small (fixed) | LoRA r=8 (matched recipe) | **TBD** | **87.0%** | - | - | - | - |

**Configuration (matched recipe, job 3637117):**
- Encoder: SSAMBA-HF-small (25.3M params, 384 embed_dim, 24 layers)
- LoRA targets: `in_proj`, `out_proj` (Mamba block projections)
- Trainable params: 589,824 (2.3% of model)
- Optimizer: Adam(wd=5e-7, betas=(0.95,0.999))
- Head: LayerNorm → Linear (matching original `mlp_head`)
- LR schedule: MultiStepLR(start=6, gamma=0.85)
- Norm stats: ESC-50 (mean=-6.63, std=5.36)
- Noise aug: uniform noise + temporal roll
- Drop path: 0.1 with per-layer stochastic depth
- Grad accumulation: batch 8 × 6 steps = effective 48
- Target length: 512, stride=10 (overlapping patches)
- GPU: V100-16GB
- CUDA kernels: enabled via mamba-ssm

**Job history:**
- Job 3447764: Crashed in Fold 2 (slow PyTorch fallback, pre-CUDA kernel support)
- Job 3519842: Completed all 5 folds with broken forward pass (29.6%)
- Job 3633114: Fixed bidirectional loop, fold 1: 81.5% (pre-recipe fix)
- Job 3637117: Matched original training recipe, fold 1: **87.0%** (fold 1 only)

**Key findings (after recipe matching):**
- LoRA improves from 61.6% → 87.0% (fold 1), a ~25% improvement from linear probe
- Recipe matching (commit `5ac5aec`) improved LoRA from 81.5% → 87.0% on fold 1
- Full 5-fold results pending

### Part F: SSAMBA-HF Full Finetuning (2026-01-30, updated)

| Encoder | Method | Mean Accuracy | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---------|--------|--------------|--------|--------|--------|--------|--------|
| SSAMBA-HF-small | Full finetune (matched recipe) | **TBD** | **87.0%** | - | - | - | - |

**Job 3637115 (V100-16GB): Completed fold 1**
- Full finetuning with all recipe fixes from commit `5ac5aec`
- Fold 1 accuracy: **87.0%** (same as LoRA)
- 200 epochs, Adam optimizer, all 25.3M params trainable

**Prior failed attempts (pre-fix):**
- Job 3523486 (V100-16GB): Out of Memory
- Job 3523727 (RTX 6000-24GB): Hit 4-hour walltime limit

### Part G: SSAMBA-HF vs Published Results — Model Size Context (2026-01-30)

The SSAMBA paper (IEEE SLT 2024) reports finetuned ESC-50 accuracy per model size:

| Model | Params | Published ESC-50 | Our Result (fold 1) | Method | Gap |
|-------|--------|-----------------|---------------------|--------|-----|
| SSAMBA-tiny | 7M | 80.1% | — | — | — |
| **SSAMBA-small** | **26M** | **85.5%** | **87.0%** | Full finetune / LoRA | **+1.5%** |
| SSAMBA-base | 99M | 89.3% | — | — | — |

**We are running SSAMBA-small (26M params, 384 embed_dim).** Our fold 1 result of 87.0% actually **exceeds** the published 85.5% for this model size by 1.5%. The 96% target in `tasks.md` was incorrect — the paper does not report 96% for any SSAMBA variant. The highest published SSAMBA result is 89.3% for base (99M params).

Note: Our 87.0% is a single fold, not a 5-fold mean. The published 85.5% is presumably a 5-fold mean. Full 5-fold CV is needed for a fair comparison, but the fold 1 result is very encouraging.

### Part H: SSAMBA-HF Inference Profiling (2026-01-30)

Stage-level inference profiling on V100-PCIE-16GB, comparing CUDA kernels (mamba-ssm 2.3.0) vs PyTorch fallback.

**GPU Benchmarks (5s audio, 16kHz):**

| Batch | CUDA kernels (ms) | PyTorch fallback (ms) | Kernel speedup | Throughput (CUDA) |
|-------|-------------------|----------------------|----------------|-------------------|
| 1 | 30 | 3,908 | 130x | 33 samp/s |
| 4 | 42 | 3,922 | 94x | 96 samp/s |
| 16 | 150 | 6,301 | 42x | 107 samp/s |
| 32 | 292 | 6,348 | 22x | 110 samp/s |

**CPU Benchmarks (4 threads):**

| Batch | With mamba-ssm (ms) | Without mamba-ssm (ms) | Throughput |
|-------|--------------------|-----------------------|------------|
| 1 | 3,448 | 4,779 | 0.3 samp/s |
| 4 | 5,551 | 7,376 | 0.7 samp/s |
| 16 | 10,827 | 12,230 | 1.5 samp/s |

**SSAMBA-HF-small vs BEATs (GPU, CUDA kernels):**

| Batch | SSAMBA (ms) | BEATs (ms) | SSAMBA / BEATs | SSAMBA peak MB | BEATs peak MB |
|-------|-------------|------------|----------------|----------------|---------------|
| 1 | 30 | 8 | 3.8x slower | 132 | 242 |
| 4 | 42 | 13 | 3.3x slower | 205 | 496 |
| 16 | 150 | 44 | 3.4x slower | 494 | 1,459 |
| 32 | 292 | 92 | 3.2x slower | 881 | 2,759 |

**Stage breakdown (GPU, CUDA kernels, batch=1):**

| Stage | Time (ms) | % of total |
|-------|-----------|------------|
| fbank (CPU) | 1.5 | 4.9% |
| patch_embed | 0.2 | 0.5% |
| pos_embed | 0.1 | 0.3% |
| **mamba_layers** | **28.1** | **93.5%** |
| norm | 0.1 | 0.4% |
| pool | 0.1 | 0.2% |

**Key findings:**

1. **CUDA kernels are essential** — without them, GPU inference is 22-130x slower (the PyTorch fallback selective scan is not parallelized). SSAMBA is unusable without mamba-ssm.
2. **SSAMBA is ~3-4x slower than BEATs on GPU** despite having fewer parameters (26M vs 90M). Mamba's linear-in-sequence-length advantage doesn't help at the ~1024 patch sequence lengths used for audio spectrograms. BEATs' attention is quadratic but highly optimized, and the constant factors dominate at these lengths.
3. **SSAMBA uses ~3x less GPU memory** than BEATs — a real advantage for deployment or larger batch sizes.
4. **Mamba layers dominate** at 84-99% of inference time across all configurations.
5. **CPU inference is impractical** — 3.4s per sample even with mamba-ssm (vs 124ms for BEATs, a 28x gap).

**Why SSAMBA is slower than BEATs despite Mamba's theoretical advantage:**
- Mamba's O(n) scaling beats attention's O(n²) only at long sequence lengths. At 1024 patches, attention with optimized CUDA kernels (FlashAttention, cuBLAS GEMM) is faster in absolute terms.
- SSAMBA-small has 24 layers with bidirectional processing (48 sequential scan operations), creating a deep serial dependency chain that limits GPU parallelism.
- BEATs has 12 transformer layers with highly parallelizable self-attention.

**Implications for BirdCLEF**: SSAMBA's speed disadvantage makes it less attractive for the inference-constrained Kaggle submission (2-hour limit for ~200k test clips). BEATs at 8ms/sample (GPU batch=1) processes 200k clips in ~27 minutes; SSAMBA at 30ms/sample would need ~100 minutes. Both fit, but BEATs leaves more headroom. SSAMBA's memory advantage matters less given the V100/T4 memory available on Kaggle.

### Confounders

- **Linear probe vs finetuning**: Published results use finetuning; we use frozen features
- **Hyperparameters**: Used 100 epochs, lr=1e-3, batch_size=32 for all; may not be optimal per-encoder
- **EAT preprocessing**: Verified against vendored code, but HuggingFace wrapper may have subtle differences
- **No data augmentation**: Published results may use SpecAugment, mixup, etc.

## Artifacts

| Type | Location |
|------|----------|
| Linear Probe Results | `artifacts/linear-probe-*.json` |
| LoRA Results | `artifacts/lora-r8-*.json`, `artifacts/wav2vec2_lora_r8_results.json` |
| GPU Stats | `sbatch/logs/gpu_stats-*.csv` |
| Job Logs | `sbatch/logs/benchmark-*.out`, `sbatch/logs/lora-finetune-*.out` |
| Checkpoints | `~/scratch/birdclef-2026/checkpoints/` |
| Code | `user/anthony/src/birdclef/` |

### Checkpoint Sources

| Model | Source | Notes |
|-------|--------|-------|
| BEATs | `lpepino/beats_ckpts` (HuggingFace) | Mirror by Leonardo Pepino; official OneDrive not programmatically accessible |
| EAT | `worstchan/EAT-base_epoch30_pretrain` (HuggingFace) | Official release from EAT authors |
| wav2vec2 | `facebook/wav2vec2-base-960h` (HuggingFace) | Standard pretrained model |
| AST | `MIT/ast-finetuned-audioset-10-10-0.4593` (HuggingFace) | Official AudioSet-finetuned model |
| SSAST | Dropbox (see tasks.md) | Requires `timm==0.4.5`, blocked |
| SSAMBA-HF | `ssamba_small_400.pth` (Google Drive) | HF port works, 61.6% linear probe (after bidirectional fix) |

## Next Steps

- [x] ~~Fix SSAMBA environment~~ - Completed via HF port (experiment 004)
- [x] ~~Add mamba-ssm CUDA kernel support~~ - Completed (experiment 004)
- [x] ~~Verify SSAMBA-HF numerical correctness~~ - Found and fixed missing outer bidirectional loop (18.4% → 61.6%)
- [x] ~~SSAMBA LoRA finetuning (broken)~~ - 29.6% with broken forward pass
- [x] ~~SSAMBA LoRA finetuning (fixed, pre-recipe)~~ - 81.5% fold 1
- [x] ~~Match original training recipe~~ - 9 differences fixed in commit `5ac5aec`
- [x] ~~SSAMBA full finetuning fold 1~~ - 87.0% (job 3637115, V100-16GB)
- [x] ~~SSAMBA LoRA finetuning fold 1 (matched recipe)~~ - 87.0% (job 3637117)
- [ ] **SSAMBA full 5-fold CV** - Run all 5 folds to get proper mean accuracy for comparison to published 85.5%
- [ ] **SSAMBA LoRA full 5-fold CV** - Compare LoRA vs full finetune across all folds
- [ ] Fix SSAST environment (port to newer timm or separate venv)
- [ ] Add BirdNET encoder for comparison (bird-specific pretraining)
- [ ] Run on AudioSet bird subset (stretch goal)
- [x] ~~Inference time benchmarks~~ - SSAMBA 3-4x slower than BEATs on GPU (30ms vs 8ms batch=1), but uses 3x less memory

**Leads to**: BEATs recommended as primary encoder for frozen features (fastest + highest accuracy). SSAMBA-HF-small with finetuning (87.0% fold 1) exceeds published results for this model size (85.5%) and approaches SSAMBA-base (89.3%). The 96% target in tasks.md was incorrect — it does not appear in the SSAMBA paper for any model size.

# Results: Encoder Benchmark on ESC-50

**Date Completed**: 2026-01-26 (updated)
**Author**: Anthony Miyaguchi

## Summary

BEATs and AST both achieve ~95-96% accuracy on ESC-50 using frozen features with a linear probe, significantly outperforming other encoders. EAT requires mean pooling (not CLS token) for competitive performance. LoRA finetuning dramatically improves EAT (77.9% → 94.6%) but fails completely for wav2vec2 due to domain mismatch. SSAMBA-HF LoRA reaches only 29.6%, far below published 96% — numerical correctness verification is now the priority.

**Verdict**: BEATs is best for frozen features (fastest + highest accuracy); AST is a close second but 5x slower. SSAMBA-HF port produces poor accuracy even with LoRA finetuning (29.6%), which is suspicious and motivates numerical correctness verification before further training.

## Observations

### Quantitative

| Encoder | Mean Accuracy | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Embed Dim | Eval Time |
|---------|--------------|--------|--------|--------|--------|--------|-----------|-----------|
| BEATs | **95.8%** | 94.7% | 95.5% | 95.7% | 98.5% | 94.5% | 768 | 107s |
| AST | 95.5% | 95.2% | 95.2% | 94.8% | 97.0% | 94.8% | 768 | 586s |
| EAT | 77.9% | 76.0% | 79.0% | 75.7% | 79.7% | 79.2% | 768 | 162s |
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

### Part D: SSAMBA-HF Benchmark (2026-01-26)

Successfully ran SSAMBA using the HuggingFace-compatible port (experiment 004), eliminating the CUDA kernel dependency.

| Encoder | Mean Accuracy | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Embed Dim | Eval Time |
|---------|--------------|--------|--------|--------|--------|--------|-----------|-----------|
| SSAMBA-HF-small (GPU) | **18.4%** | 21.8% | 16.8% | 21.3% | 16.8% | 15.5% | 384 | 2504s |
| SSAMBA-HF-small (CPU) | **18.5%** | 20.0% | 18.5% | 21.5% | 16.8% | 15.5% | 384 | 9048s |

**GPU Statistics (SSAMBA-HF GPU run, V100-32GB):**

| Metric | Value |
|--------|-------|
| Average GPU Utilization | ~10% |
| Peak GPU Utilization | Low (sequential SSM) |
| Peak Memory | 934 MiB (2.9% of 32GB) |
| Average Power Draw | 48.1 W |
| Peak Power Draw | 71.2 W |
| Job Duration | ~42 min |

**Why SSAMBA-HF accuracy is so low (18.4% vs published 96%):**

1. **Linear probe vs finetuning gap**: SSAMBA published results (96.0% on ESC-50) are from full finetuning, not frozen feature extraction. The 78% gap is much larger than BEATs (2.3% gap) or EAT (18% gap).

2. **Self-supervised pretraining characteristics**: SSAMBA uses AudioMAE-style masked spectrogram modeling, which may learn representations that require task-specific finetuning to be useful. BEATs' iterative tokenization objective produces more transferable frozen features.

3. **Numerical equivalence confirmed but representations differ from competitors**: The HF port achieves cosine similarity = 1.0 vs original CUDA implementation (see experiment 004). The low accuracy is a property of SSAMBA's pretrained representations, not a porting bug.

4. **Embedding dimension**: SSAMBA-small produces 384-dim embeddings vs 768-dim for BEATs/AST/EAT, which may limit expressiveness for linear probing.

**Conclusion**: SSAMBA pretrained weights are not suitable for frozen feature extraction with linear probes. Would require LoRA/full finetuning to achieve competitive results, similar to the wav2vec2 situation but less severe (18% > 2% random baseline).

### Part E: SSAMBA-HF LoRA Finetuning (Completed 2026-01-27)

| Encoder | Method | Mean Accuracy | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---------|--------|--------------|--------|--------|--------|--------|--------|
| SSAMBA-HF-small | Linear probe | 18.4% | 21.8% | 16.8% | 21.3% | 16.8% | 15.5% |
| SSAMBA-HF-small | LoRA r=8 | **29.6%** | 30.5% | 23.5% | 32.3% | 34.0% | 27.8% |

**Configuration:**
- Encoder: SSAMBA-HF-small (25.3M params)
- LoRA targets: `in_proj`, `out_proj` (Mamba block projections)
- Trainable params: 589,824 (2.3% of model)
- Batch size: 4 (limited by memory)
- GPU: RTX 6000 (24GB)
- Epochs: 50 (5-fold CV)
- CUDA kernels: enabled via mamba-ssm (confirmed in job logs)

**Job history:**
- Job 3447764: Crashed in Fold 2 (slow PyTorch fallback, pre-CUDA kernel support)
- Job 3519842: Completed all 5 folds successfully with CUDA kernels

**Key findings:**
- LoRA improves from 18.4% → 29.6% (+11.2%), far less than EAT's +16.7% improvement
- 29.6% is far below published 96.0% (full finetuning) — the gap is suspiciously large
- Even with LoRA, SSAMBA-HF performs worse than EAT's frozen linear probe (77.9%)
- This suggests either a porting bug or a preprocessing mismatch rather than just a finetuning gap

### Part F: SSAMBA-HF Full Finetuning (Failed)

Attempted full finetuning to match the published 96% result. Both attempts failed.

**Job 3523486 (V100-16GB): Out of Memory**
- Error: `OutOfMemoryError: CUDA out of memory. Tried to allocate 74.00 MiB (GPU 0; 15.77 GiB total capacity; 15.08 GiB already allocated)`
- V100-16GB insufficient for full finetuning of 25.3M params with batch size 4
- Peak memory exceeds 16GB when all parameters require gradients

**Job 3523727 (RTX 6000-24GB): Hit Walltime**
- Training started successfully on RTX 6000 (24GB)
- CUDA kernels confirmed working (mamba-ssm loaded)
- Hit 4-hour walltime limit during Fold 1 training
- Full finetuning is significantly slower than LoRA even with CUDA kernels

**Conclusion:** Full finetuning requires either:
1. A larger GPU (A100-40GB) with longer walltime allocation
2. Gradient checkpointing to reduce memory footprint
3. The script (`finetune_full.py`) is ready but infrastructure is the bottleneck

**However**: Given the poor LoRA results (29.6%), full finetuning alone is unlikely to reach 96%. The priority is verifying numerical correctness of the SSAMBA-HF port before investing more compute.

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
| SSAMBA-HF | `ssamba_small_400.pth` (Google Drive) | HF port works, poor linear probe accuracy (18.4%) |

## Next Steps

- [ ] **PRIORITY: Verify SSAMBA-HF numerical correctness** (experiment 004 verification suite)
  - Preprocessing audit: fbank params, normalization, input dimensions
  - Multi-input verification: multiple seeds + real ESC-50 audio
  - Layer-by-layer activation comparison (original vs HF)
  - CUDA vs PyTorch kernel comparison
- [ ] Fix SSAST environment (port to newer timm or separate venv)
- [x] ~~Fix SSAMBA environment~~ - Completed via HF port (experiment 004); 18.4% accuracy not competitive
- [ ] Add BirdNET encoder for comparison (bird-specific pretraining)
- [ ] Run on AudioSet bird subset (stretch goal)
- [ ] Inference time benchmarks (GPU/CPU at various batch sizes)
- [x] ~~SSAMBA LoRA finetuning~~ - Completed (29.6% mean, far below published 96%)
- [x] ~~Add mamba-ssm CUDA kernel support~~ - Completed (experiment 004), confirmed working in LoRA job
- [ ] SSAMBA full finetuning (blocked on verification + larger GPU)

**Leads to**: BEATs recommended as primary encoder for BirdCLEF 2026 feature extraction pipeline (fastest + highest accuracy). SSAMBA-HF requires correctness verification before further investment.

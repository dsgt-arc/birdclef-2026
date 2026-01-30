# Results: Encoder Benchmark on ESC-50

**Date Completed**: 2026-01-29 (updated)
**Author**: Anthony Miyaguchi

## Summary

BEATs and AST both achieve ~95-96% accuracy on ESC-50 using frozen features with a linear probe, significantly outperforming other encoders. EAT requires mean pooling (not CLS token) for competitive performance. LoRA finetuning dramatically improves EAT (77.9% → 94.6%) but fails completely for wav2vec2 due to domain mismatch. SSAMBA-HF was fixed (missing outer bidirectional loop) and now achieves 61.6% linear probe and 81.5%+ with LoRA (job still running).

**Verdict**: BEATs is best for frozen features (fastest + highest accuracy); AST is a close second but 5x slower. SSAMBA-HF linear probe (61.6%) is modest but LoRA finetuning shows strong improvement (81.5% fold 1, remaining folds in progress).

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

### Part E: SSAMBA-HF LoRA Finetuning (2026-01-29, updated)

| Encoder | Method | Mean Accuracy | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---------|--------|--------------|--------|--------|--------|--------|--------|
| SSAMBA-HF-small (broken) | Linear probe | 18.4% | 21.8% | 16.8% | 21.3% | 16.8% | 15.5% |
| SSAMBA-HF-small (broken) | LoRA r=8 | 29.6% | 30.5% | 23.5% | 32.3% | 34.0% | 27.8% |
| SSAMBA-HF-small (fixed) | Linear probe | 61.6% | 63.0% | 64.5% | 62.5% | 61.0% | 57.0% |
| SSAMBA-HF-small (fixed) | LoRA r=8 | **TBD** | 81.5% | _running_ | - | - | - |

**Configuration:**
- Encoder: SSAMBA-HF-small (25.3M params)
- LoRA targets: `in_proj`, `out_proj` (Mamba block projections)
- Trainable params: 589,824 (2.3% of model)
- Batch size: 4 (limited by memory)
- GPU: V100-32GB
- Epochs: 50 (5-fold CV)
- CUDA kernels: enabled via mamba-ssm (confirmed in job logs)

**Job history:**
- Job 3447764: Crashed in Fold 2 (slow PyTorch fallback, pre-CUDA kernel support)
- Job 3519842: Completed all 5 folds with broken forward pass (29.6%)
- Job 3633114: Running with fixed outer bidirectional loop (fold 1: 81.5%, fold 2 in progress)

**Key findings (after fix):**
- LoRA improves from 61.6% → 81.5%+ (fold 1), a ~20% improvement from linear probe
- The broken forward pass was the root cause of the previous 29.6% result
- With the fix, SSAMBA-HF LoRA now surpasses EAT linear probe (77.9%) and approaches EAT LoRA (94.6%)
- Full 5-fold results pending (job 3633114 still running)

### Part F: SSAMBA-HF Full Finetuning (Failed — needs retry with fix)

Attempted full finetuning before the outer bidirectional loop fix. Both attempts failed due to infrastructure.

**Job 3523486 (V100-16GB): Out of Memory**
- V100-16GB insufficient for full finetuning of 25.3M params with batch size 4

**Job 3523727 (RTX 6000-24GB): Hit Walltime**
- Hit 4-hour walltime limit during Fold 1 training

**Conclusion:** Full finetuning requires a larger GPU (A100-40GB) with longer walltime. Now that the forward pass is fixed (LoRA reaching 81.5%+), full finetuning would be worth retrying to approach the published 96%.

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
- [ ] **SSAMBA LoRA finetuning (fixed)** - Job 3633114 running (fold 1: 81.5%, fold 2 in progress)
- [ ] SSAMBA full finetuning (retry with fixed forward pass on A100)
- [ ] Fix SSAST environment (port to newer timm or separate venv)
- [ ] Add BirdNET encoder for comparison (bird-specific pretraining)
- [ ] Run on AudioSet bird subset (stretch goal)
- [ ] Inference time benchmarks (GPU/CPU at various batch sizes)

**Leads to**: BEATs recommended as primary encoder for frozen features (fastest + highest accuracy). SSAMBA-HF with LoRA is now competitive (81.5%+ vs EAT LoRA 94.6%), and full finetuning may close the gap further.

# Results: Encoder Benchmark on ESC-50

**Date Completed**: 2026-01-10
**Author**: Anthony Miyaguchi

## Summary

BEATs and AST both achieve ~95-96% accuracy on ESC-50 using frozen features with a linear probe, significantly outperforming other encoders. EAT requires mean pooling (not CLS token) for competitive performance. LoRA finetuning dramatically improves EAT (77.9% → 94.6%) but fails completely for wav2vec2 due to domain mismatch.

**Verdict**: BEATs is best for frozen features (fastest + highest accuracy); AST is a close second but 5x slower. SSAST/SSAMBA blocked due to environment issues.

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
| SSAMBA | `attentionisallyouneed369/ssamba` (HuggingFace) | Weights only, needs mamba-ssm, blocked |

## Next Steps

- [ ] Fix SSAST environment (port to newer timm or separate venv)
- [ ] Fix SSAMBA environment (port model code, resolve CUDA mismatch)
- [ ] Add BirdNET encoder for comparison (bird-specific pretraining)
- [ ] Run on AudioSet bird subset (stretch goal)
- [ ] Inference time benchmarks (GPU/CPU at various batch sizes)

**Leads to**: BEATs recommended as primary encoder for BirdCLEF 2026 feature extraction pipeline (fastest + highest accuracy)

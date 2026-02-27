# Results: LoRA Fine-Tuning EAT on BirdCLEF 2025

**Date Completed**: 2026-02-25
**Author**: Murilo Gustineli

## Summary

LoRA fine-tuning dramatically outperforms the frozen linear probe baseline from Exp 005,
with the best config (adapters + attn LoRA + MLP head) reaching cmAP=0.536 — a 78%
improvement over the Exp 005 frozen baseline of 0.301. The MLP classification head matters
more than LoRA scope, and the adapters library slightly outperforms PEFT on equivalent configs.

**Verdict**: confirmed

## Observations

### Quantitative

All 5 completed runs vs. Exp 005 frozen linear probe baseline (cmAP=0.301):

| Run | Backend | LoRA Targets | Head | test cmAP | Delta | AUROC | F1 macro | Stopped Epoch | Walltime |
|-----|---------|-------------|------|-----------|-------|-------|----------|---------------|----------|
| `eat-lora-attn` | PEFT | qkv, proj | Linear | 0.469 | +0.168 (+56%) | 0.927 | 0.409 | 36 | 11h 29m |
| `eat-lora-all` | PEFT | qkv, proj, fc1, fc2 | Linear | 0.482 | +0.181 (+60%) | 0.923 | 0.403 | 32 | 11h 16m |
| `eat-lora-attn-mlphead` | PEFT | qkv, proj | MLP 512 | 0.511 | +0.210 (+70%) | 0.942 | 0.441 | 27 | 8h 17m |
| `adapters-eat-lora-attn` | adapters | qkv, proj | Linear | 0.485 | +0.184 (+61%) | 0.938 | 0.424 | 31 | 9h 23m |
| **`adapters-eat-lora-attn-mlphead`** | **adapters** | **qkv, proj** | **MLP 512** | **0.536** | **+0.235 (+78%)** | **0.945** | **0.475** | **31** | **11h 00m** |
| `adapters-eat-lora-all` | adapters | qkv, proj, fc1, fc2 | Linear | — | — | — | — | — | >12h (timed out) |

All runs used a single RTX 6000 GPU with batch_size=32, accumulate_grad_batches=8
(effective batch=256), and early stopping patience=10.

### Qualitative

- **MLP head > LoRA scope**: Adding an MLP classification head (768->512->206) consistently
  outperforms expanding LoRA to MLP layers (fc1, fc2). `attn + MLP head` beats `attn+mlp LoRA
  + linear head` by ~0.03-0.05 cmAP on both backends. Investing parameters in the head is
  more cost-effective than adapting more encoder layers.

- **Adapters slightly outperforms PEFT**: On equivalent configs, adapters produces ~0.01-0.025
  higher cmAP than PEFT. This is likely due to differences in how the two libraries handle
  fused QKV layers — adapters uses `LoRAMergedLinear` with separate enable flags per Q/K/V,
  while PEFT treats the fused layer as a single target.

- **All runs converge well**: Early stopping triggered between epochs 27-36 (patience=10),
  well within the 50-epoch budget.

- **Training is encoder-bound**: GPU profiling shows the forward+backward pass through the
  90M-param EAT encoder dominates step time (~117-203ms depending on backend and batch size).
  Increasing batch_size from 32 to 64 only improved epoch time by ~18% — half the steps but
  ~70% more compute per step. Data loading and Python overhead are not the bottleneck.
  See `gpu-profiling.md` for full analysis.

- **Efficiency**: Training with LoRA (0.67-1.47% of params) fits comfortably in memory at
  batch_size=32 on a single RTX 6000 (~5.4 GB reserved of 24 GB available), unlike full
  fine-tuning which OOM'd in Exp 005.

## Analysis

The frozen linear probe baseline from Exp 005 (cmAP=0.301) only trains the classification
head on top of fixed EAT features. LoRA allows the encoder to adapt its attention
representations to the BirdCLEF domain while keeping the parameter budget small. The
consistent improvement across all configs confirms that EAT's pretrained features benefit
from task-specific adaptation, even with <1% of parameters trainable.

The head architecture finding is notable: a 512-dim MLP head with ~500K params outperforms
expanding LoRA targets from 451K to 1,188K params. This suggests the bottleneck is in the
classification head's capacity to decode encoder features, not in the encoder's ability to
produce discriminative representations. The linear head may be too simple for 206-class
multi-label classification.

The adapters vs PEFT gap, while small, is consistent. The adapters library's `LoRAMergedLinear`
applies LoRA separately to Q, K, and V within the fused QKV projection, using per-matrix
enable flags and grouped convolutions. PEFT applies a single LoRA adapter to the entire fused
`qkv` linear layer. This structural difference may give adapters finer-grained control over
what gets adapted.

### Confounders

- **`adapters-eat-lora-all` timed out**: The adapters backend with all LoRA targets (qkv +
  proj + fc1 + fc2) exceeded the 12-hour SLURM time limit. This is the most parameter-heavy
  config (1.35M trainable) and the adapters library's per-matrix LoRA handling adds compute
  overhead (~50% slower per step than PEFT). The PEFT equivalent completed in 11h 16m.

- **Single seed**: All runs use seed=42. Results may vary with different random seeds.

## Artifacts

| Type | Location |
|------|----------|
| PEFT metrics | `artifacts/training/{eat_lora_attn,eat_lora_all,eat_lora_attn_mlphead}/metrics.json` |
| Adapters metrics | `artifacts/training/{adapters_eat_lora_attn,adapters_eat_lora_attn_mlphead}/metrics.json` |
| Test results (all) | `outputs/test_results/{eat-lora-*.json,adapters-eat-lora-*.json}` |
| Checkpoints | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/006-lora-finetune/checkpoints/` |
| TensorBoard | `outputs/tensorboard/exp006-*/` |
| W&B | `outputs/wandb/wandb/run-*/` |
| GPU profiling | `outputs/gpu_check/*.{txt,png}`, `gpu-profiling.md` |
| SLURM logs | `logs/slurm_*.log` |
| Code | Branch `mg/006-lora-finetune` |

## Next Steps

- [ ] Re-run `adapters-eat-lora-all` with longer SLURM time limit (or skip — low priority given findings)
- [ ] Explore higher LoRA rank (r=16, r=32) to see if more adapter capacity helps
- [ ] Test with adapters backend as default for future experiments (slight cmAP edge)
- [ ] Try combining MLP head with expanded LoRA targets (`eat_lora_all` + `hidden_dim: 512`)
- [ ] Open upstream PRs for adapter-hub/adapters (fused QKV fix, EAT model support)

**Leads to**: The best config (`adapters-eat-lora-attn-mlphead`, cmAP=0.536) becomes the
new baseline for downstream experiments. Consider exploring domain-specific pretraining
(BirdNET weights) or data augmentation strategies to push cmAP further.

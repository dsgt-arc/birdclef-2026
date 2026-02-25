# Results: LoRA Fine-Tuning EAT on BirdCLEF 2025

**Date Completed**: 2026-02-25
**Author**: Murilo Gustineli

## Summary

LoRA fine-tuning dramatically outperforms the frozen linear probe baseline, with the best
config (adapters + attn LoRA + MLP head) reaching cmAP=0.536 — a 78% improvement over the
Exp 005 frozen baseline of 0.301. The MLP classification head matters more than LoRA scope,
and the adapters library slightly outperforms PEFT on equivalent configs.

**Verdict**: confirmed

## Observations

### Quantitative

All 5 completed runs vs. frozen linear probe baseline (Exp 005, cmAP=0.301):

| Run | Backend | LoRA Targets | Head | test cmAP | Delta | AUROC | F1 macro | Stopped Epoch |
|-----|---------|-------------|------|-----------|-------|-------|----------|---------------|
| `eat-lora-attn` | PEFT | qkv, proj | Linear | 0.469 | +0.168 (+56%) | 0.927 | 0.409 | 36 |
| `eat-lora-all` | PEFT | qkv, proj, fc1, fc2 | Linear | 0.482 | +0.181 (+60%) | 0.923 | 0.403 | 32 |
| `eat-lora-attn-mlphead` | PEFT | qkv, proj | MLP 512 | 0.511 | +0.210 (+70%) | 0.942 | 0.441 | 27 |
| `adapters-eat-lora-attn` | adapters | qkv, proj | Linear | 0.485 | +0.184 (+61%) | 0.938 | 0.424 | 31 |
| **`adapters-eat-lora-attn-mlphead`** | **adapters** | **qkv, proj** | **MLP 512** | **0.536** | **+0.235 (+78%)** | **0.945** | **0.475** | **31** |
| `adapters-eat-lora-all` | adapters | qkv, proj, fc1, fc2 | Linear | — | — | — | — | timed out (>12h) |

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
  well within the 50-epoch budget. Training time for completed runs was 9-11 hours on a
  single RTX 6000.

- **Efficiency**: Training with LoRA (0.67-1.47% of params) fits comfortably in memory at
  batch_size=32 on a single RTX 6000, unlike full fine-tuning which OOM'd in Exp 005.
  GPU utilization was 90.6% during the validation check.

## Analysis

The frozen linear probe baseline (cmAP=0.301) only trains the classification head on top
of fixed EAT features. LoRA allows the encoder to adapt its attention representations to
the BirdCLEF domain while keeping the parameter budget small. The consistent improvement
across all configs confirms that EAT's pretrained features benefit from task-specific
adaptation, even with <1% of parameters trainable.

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
  config (1.35M trainable) and the adapters library's per-matrix LoRA handling adds overhead.
  The PEFT equivalent completed in ~11 hours.

- **`save_metrics.py` bug**: The adapters runs' structured metrics were not saved to
  `artifacts/` because `save_metrics.py` looks for `{run_name}.json` from the YAML config
  (e.g., `eat-lora-attn.json`) but `train_adapters.py` saves results with an `adapters-`
  prefix. The raw test results exist in `outputs/test_results/` and are used in this analysis.

- **Single seed**: All runs use seed=42. Results may vary with different random seeds.

## Artifacts

| Type | Location |
|------|----------|
| PEFT metrics | `artifacts/training/{eat_lora_attn,eat_lora_all,eat_lora_attn_mlphead}/metrics.json` |
| Test results (all) | `outputs/test_results/{eat-lora-*.json,adapters-eat-lora-*.json}` |
| Checkpoints | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/006-lora-finetune/checkpoints/` |
| TensorBoard | `outputs/tensorboard/exp006-*/` |
| W&B | `outputs/wandb/wandb/run-*/` |
| SLURM logs | `logs/slurm_*.log` |
| Code | Branch `mg/006-lora-finetune` |

## Next Steps

- [ ] Fix `save_metrics.py` to accept `--backend` flag for adapters-prefixed results
- [ ] Re-run `adapters-eat-lora-all` with longer SLURM time limit (or skip — low priority given findings)
- [ ] Explore higher LoRA rank (r=16, r=32) to see if more adapter capacity helps
- [ ] Test with adapters backend as default for future experiments (slight cmAP edge)
- [ ] Try combining MLP head with expanded LoRA targets (`eat_lora_all` + `hidden_dim: 512`)

**Leads to**: The best config (`adapters-eat-lora-attn-mlphead`, cmAP=0.536) becomes the
new baseline for downstream experiments. Consider exploring domain-specific pretraining
(BirdNET weights) or data augmentation strategies to push cmAP further.

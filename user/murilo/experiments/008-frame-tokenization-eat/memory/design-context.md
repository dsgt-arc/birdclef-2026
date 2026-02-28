# Experiment 008: Frame Tokenization for EAT — Design Context

## Research Question
Does full-frequency frame tokenization compensate for losing pretrained patch embeddings in EAT?

## Downstream Goal
Select the best teacher model for pseudo-label generation (future weak supervision experiment).

## Why This Experiment
- Exp 004: frame tokenization (128×4) improved cmAP +37% over 16×16 when training ViT from scratch
- Exp 006: EAT LoRA best = cmAP 0.536 (adapters) / 0.482 (PEFT all-layer)
- Exp 007: ConvNeXt LoRA = cmAP 0.527 (only 1.7% below EAT best)
- Question: can frame tokenization from exp 004 improve EAT from exp 006?

## Core Tension
- Frame tokenization worked in exp 004 because training was FROM SCRATCH
- EAT's value comes from PRETRAINED weights (AudioSet self-supervised)
- Changing patch shape (16×16 → W×128) requires random init of patch embedding
- Trade-off: better tokenization vs loss of pretrained patch representations

## Axis Convention (CRITICAL)
- **EAT layout**: `(B, 1, time, freq)` = `(B, 1, 512, 128)` — time is dim 2, freq is dim 3
- **Exp 004 FrameViT layout**: `(B, 1, freq, time)` = `(B, 1, 128, 500)` — freq is dim 2, time is dim 3
- **EAT kernel for full-freq frames**: `Conv2d(1, 768, (W, 128), (W, 128))` — (W_time, 128_freq)
- **Exp 004 kernel**: `Conv2d(1, 768, (128, W), (128, W))` — (128_freq, W_time) — same concept, flipped axes
- Config naming uses `W×128` (time×freq) to match EAT's axis order

## Run Matrix (A=baseline, B=control, C=frame variants)

### Phase 1 (4 runs)
- **A: baseline** — Pretrained 16×16 (frozen) + LoRA r=8 → reproduces exp 006 PEFT
- **B: control** — Random 16×16 (full train) + LoRA r=8 → isolates pretraining loss
- **C1a: frame-2×128** — Random 2×128 (full train) + LoRA r=8 → 256 tokens, 20ms resolution
- **C1b: frame-4×128** — Random 4×128 (full train) + LoRA r=8 → 128 tokens, 40ms resolution

### Phase 2 (2 runs, best frame size from Phase 1)
- **C2: frame + higher rank** — LoRA r=16 → more adapter capacity
- **C3: frame + partial unfreeze** — LoRA r=8 + blocks 0-1 fully trained → bridge layers

## Interpretation Guide
- C > B > A → frame tokenization beats even pretrained patches (surprising)
- A > C > B → frame helps vs random squares, but pretrained still wins (expected)
- A > B ≈ C → frame shape doesn't matter; pretraining is what counts
- A > B > C → frame tokenization actively hurts

## Success Criteria
- **Confirm**: best C config cmAP > Config B (frame shape helps beyond retraining)
- **Reject**: best C config cmAP ≤ Config B (no frame advantage)
- **Bonus**: any C config cmAP > Config A (overcomes pretraining loss)

## Implementation Notes
- PEFT backend only (simpler; winning config can be re-run with adapters later)
- Patch embed: fully trained (~196K for 2×128, ~393K for 4×128), NOT LoRA
- Transformer: pretrained + LoRA on ["qkv", "proj"] (same as exp 006)
- Training: max_epochs=100, patience=15 (extended for random patch embed convergence)
- batch_size=32, accum=8 (effective=256), lr=3e-4, seed=42
- C3 partial unfreeze: blocks 0-1 fully trained, blocks 2-11 LoRA only

## Exp 004 Frame Tokenization Reference
- Implementation: `user/murilo/src/birdclef/models/frame_vit.py`
- Uses timm's `patch_size` tuple to change Conv2d kernel/stride
- 1-channel adaptation: average pretrained 3-ch weights → 1-ch (for pretrained models)
- For random init: no weight adaptation needed
- Configs tested: 128×1 (500 tokens), 128×2 (250 tokens), 128×4 (125 tokens) — in (freq, time) order
- Best: 128×4 on Hawaiian dataset, but 128×2 was close — need to test both on BirdCLEF 2025
- NOTE: exp 004 uses (freq, time) axis order; EAT uses (time, freq) — kernel dims are swapped

## Files Created
- `user/murilo/experiments/008-frame-tokenization-eat/proposal.md` — full proposal
- `user/murilo/experiments/008-frame-tokenization-eat/tasks.md` — 25 tasks, 6 phases
- `user/murilo/experiments/008-frame-tokenization-eat/pyproject.toml` — PEFT as direct dep
- `src` symlink → `../../src` (shared user/murilo/src/)
- Empty dirs: artifacts/, config/, sbatch/, logs/, outputs/

## Next Steps (for next session)
1. Start implementation: `FramePatchEmbed` with (W, 128) kernel, positional encoding regen, `build_eat_frame_model()`
2. Create YAML configs for all 6 runs
3. Create SLURM scripts and save_metrics.py
4. Fast-dev-run validation
5. Submit Phase 1 runs

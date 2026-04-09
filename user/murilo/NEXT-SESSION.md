<!-- Handoff prompt for the next Claude Code session. Copy-paste as opening message. -->

We'll be working from user/murilo/ in this session. I'm continuing from the previous session where we built the Perch + SED ensemble (exp 023) and established the three source-of-truth docs.

## What happened last session

1. **Exp 023: Perch + SED Ensemble — 0.929 LB (new best overall)**
   - Built single Kaggle CPU kernel combining Perch ONNX + SED v5j-4fold-pp2
   - Config-driven submission system: YAML configs in `configs/`, `push.sh <config_name>` injects override
   - Weight grid: w=0.3 (0.927), w=0.5 (0.927), w=0.7 (0.926) — flat curve
   - **MLP-only (proto_weight=0.0): 0.929 LB** — dropping ProtoSSM improved by +0.002
   - No PP: 0.922 (−0.005) — confirms PP is valuable
   - TF Perch + single SED submitted but likely to timeout (~120 min est. for 600 files)

2. **Documentation overhaul** — Three source-of-truth docs established:
   - `user/murilo/EXPERIMENTS.md` — per-experiment methodology (updated through exp 023)
   - `user/murilo/MODEL-RUNS.md` — all model scores and checkpoints
   - `user/murilo/FINDINGS.md` — cross-cutting insights and patterns (new)

3. **Branch**: `mg/023-perch-sed-ensemble` — PR #25

## Current scoreboard

| Model | LB | Notes |
|-------|-----|-------|
| **Perch+SED MLP-only (exp 023)** | **0.929** | **Current best — ONNX Perch + SED 4-fold, proto_weight=0.0** |
| Perch+SED w=0.5 (exp 023) | 0.927 | ProtoSSM+MLP blend |
| Perch ProtoSSM v18 TF (exp 020) | 0.925 | Best individual model |
| SED v5j-4fold-pp2 (exp 021) | 0.917 | Best SED |
| Competition #1 | 0.939 | Target — gap: 0.010 |

## What needs to happen now

### Immediate: Check TF+single result
- TF Perch + SED single-fold + MLP-only was submitted but may timeout
- If it succeeds: could score 0.932+ (TF recovers 0.005 ONNX penalty)
- If it times out: need to optimize TF timing or abandon TF path for ensemble

### High-value next steps (from FINDINGS.md remaining gap analysis)

**Option A: MLP-only weight grid** (zero effort)
- MLP-only used w=0.5. Try w=0.3, 0.7 with MLP-only to find optimal blend
- Configs to create: `ensemble_mlp_only_w03.yaml`, `ensemble_mlp_only_w07.yaml`

**Option B: Knowledge distillation from ensemble** (medium effort)
- Use the 0.929 ensemble as teacher to train a single stronger SED model
- SED-to-SED distillation might work where clip-level distillation (exp 015) failed
- Risk: exp 015 showed pseudo-labels hurt, but architecture was different

**Option C: Better post-processing for ensemble** (medium effort)
- Current: each pipeline runs own PP, then average
- Alternative: ensemble raw scores, then apply single unified PP
- No-PP scored 0.922 (−0.005), so PP helps but may not be optimal

**Option D: Perch embedding probes on SED embeddings** (medium effort)
- Train linear probes on extracted v5j embeddings (1024-dim)
- 4-fold structure enables proper OOF evaluation
- Could reveal if SED embeddings carry complementary information

**Option E: Framewise inference** (from exp 023 discussion)
- Use framewise max instead of LSE pool at inference time
- Nikita (1st place 2025) commented out attention, used framewise max
- Could be done as inference-only change (no retraining)

## Key learned patterns (from FINDINGS.md)

- **Only input representation and inference-time changes improve LB** — 15+ training recipe changes, 0 successes
- **Offline metrics are anti-correlated with LB** — don't trust val/auroc for model selection
- **Simpler models beat complex ones in ensemble** — MLP-only > ProtoSSM+MLP (+0.002)
- **PP is worth +0.005** — both pipelines' PP contribute positively
- **Weight doesn't matter much** — 0.926-0.927 across w=0.3/0.5/0.7

## Key context files

- `user/murilo/EXPERIMENTS.md` — per-experiment summaries (001-023)
- `user/murilo/MODEL-RUNS.md` — all model scores
- `user/murilo/FINDINGS.md` — cross-cutting insights and patterns
- `user/murilo/experiments/023-perch-sed-ensemble/` — current experiment
  - `kaggle_kernel/submission_ensemble.py` — combined kernel (~2,200 lines)
  - `configs/` — YAML configs for each submission variant
  - `kaggle_kernel/push.sh <config_name>` — config-driven Kaggle push
  - `results.md` — all submission scores
- `user/murilo/experiments/020-perch-onnx/` — Perch ProtoSSM v18 source
- `user/murilo/experiments/010-submission-pipeline/` — SED submission source
